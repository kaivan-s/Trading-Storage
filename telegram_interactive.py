"""
Interactive Telegram bot — users message a stock name, get a full report.

Commands:
    /start              — welcome message
    /help               — list commands
    /r SYMBOL           — stock report (alias: /report, or just type a name)
    /sector NAME        — sector overview
    /heatmap            — sector rotation map
    /flow               — money flow: into vs out of sectors
    /triggers           — stocks within 2% of breakout trigger
    /delivery           — unusual delivery % (institutional activity)
    /changed            — what changed since yesterday

Architecture:
    Telegram sends updates to POST /api/telegram/webhook.
    This module processes the update and replies inline.
    The bot talks to the same in-memory engine that powers the dashboard.
"""

from __future__ import annotations

import os
import requests
import numpy as np
import pandas as pd

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
PAID_CHAT = os.environ.get("TELEGRAM_PAID_CHAT", "")
API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# --------------------------------------------------------------------------
# Freemium gating
# --------------------------------------------------------------------------
# Stock reports (/r, plain text)  → FREE, unlimited
# Market views (/heatmap, /flow, /triggers, /delivery, /changed)
#   → FREE: 3 per day
#   → PAID: unlimited
#
# "Paid" = member of the paid Telegram channel. No separate DB needed.
# --------------------------------------------------------------------------

FREE_DAILY_LIMIT = 3
_PAID_COMMANDS = {"/heatmap", "/flow", "/triggers", "/delivery", "/changed", "/today"}

# In-memory usage tracker: {user_id: {"date": "YYYY-MM-DD", "count": int}}
_usage: dict[int, dict] = {}

# Cache paid status for 10 min to avoid hammering the API
_paid_cache: dict[int, tuple[float, bool]] = {}
_PAID_CACHE_TTL = 600


def _is_paid(user_id: int) -> bool:
    """Check if a user is a member of the paid channel."""
    import time as _time

    if not PAID_CHAT or not BOT_TOKEN:
        return False

    # Check cache
    cached = _paid_cache.get(user_id)
    if cached and (_time.time() - cached[0]) < _PAID_CACHE_TTL:
        return cached[1]

    try:
        r = requests.get(f"{API}/getChatMember", params={
            "chat_id": PAID_CHAT,
            "user_id": user_id,
        }, timeout=10)
        if r.ok:
            status = r.json().get("result", {}).get("status", "")
            is_member = status in ("member", "administrator", "creator")
            _paid_cache[user_id] = (_time.time(), is_member)
            return is_member
    except Exception as exc:
        print(f"[tg-bot] paid check failed: {exc}")

    _paid_cache[user_id] = (_time.time(), False)
    return False


def _check_limit(user_id: int, command: str) -> bool:
    """
    Returns True if the user can use this command.
    Returns False if they've hit the free daily limit.
    """
    from datetime import date as _date

    # Stock reports are always free
    if command not in _PAID_COMMANDS:
        return True

    # Paid users have no limit
    if _is_paid(user_id):
        return True

    # Free user: check daily usage
    today = _date.today().isoformat()
    usage = _usage.get(user_id, {})
    if usage.get("date") != today:
        usage = {"date": today, "count": 0}

    if usage["count"] >= FREE_DAILY_LIMIT:
        return False

    usage["count"] += 1
    _usage[user_id] = usage
    return True


def _remaining(user_id: int) -> int | None:
    """How many free market views left today. None if paid."""
    from datetime import date as _date

    if _is_paid(user_id):
        return None

    today = _date.today().isoformat()
    usage = _usage.get(user_id, {})
    if usage.get("date") != today:
        return FREE_DAILY_LIMIT
    return max(0, FREE_DAILY_LIMIT - usage.get("count", 0))


def _limit_msg() -> str:
    return (
        "🔒 <b>Daily limit reached</b>\n\n"
        "Free users get 3 market views per day.\n"
        "Stock reports (<code>TRENT</code>, <code>RELIANCE</code>) are always free.\n\n"
        "🔓 <b>Unlock unlimited access:</b>\n"
        "Join the Pro channel for real-time circuit alerts "
        "and unlimited market views.\n\n"
        "DM @morrow_desk_admin for access."
    )


# Bot commands shown in Telegram's / menu
BOT_COMMANDS = [
    {"command": "today", "description": "📅 What matters today — top picks + insight"},
    {"command": "heatmap", "description": "🗺️ Sector rotation map"},
    {"command": "flow", "description": "🔄 Money flowing into/out of sectors"},
    {"command": "triggers", "description": "🎯 Stocks near breakout trigger"},
    {"command": "delivery", "description": "📦 Unusual institutional delivery"},
    {"command": "changed", "description": "📋 What changed since yesterday"},
    {"command": "sector", "description": "🏭 Sector overview — /sector Retailing"},
    {"command": "help", "description": "📖 All commands"},
]

KLASS_EMOJI = {
    "CROSSING": "🟢", "PULLBACK": "🟡", "BASE": "⚪",
    "CROSSING_UNVERIFIED": "🟠", "DOWN": "🔴",
    "DISQUALIFIED": "⛔", "NEGLECT": "⚫", "NONE": "⚫",
}


def _reply(chat_id: int, text: str, parse_mode: str = "HTML",
           buttons: list[list[dict]] | None = None) -> bool:
    """
    Send a message, optionally with inline keyboard buttons.

    buttons format: [[{"text": "Label", "callback_data": "cmd"}], ...]
    Each inner list is one row of buttons.
    """
    if not BOT_TOKEN:
        return False
    payload: dict = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "link_preview_options": {"is_disabled": True},
    }
    if buttons:
        payload["reply_markup"] = {"inline_keyboard": buttons}
    try:
        r = requests.post(f"{API}/sendMessage", json=payload, timeout=15)
        if not r.ok:
            print(f"[tg-bot] reply failed: {r.status_code} {r.text[:200]}")
        return r.ok
    except Exception as exc:
        print(f"[tg-bot] reply error: {exc}")
        return False


def _answer_callback(callback_id: str, text: str = "") -> bool:
    """Acknowledge a button press (removes the loading spinner)."""
    if not BOT_TOKEN:
        return False
    try:
        requests.post(f"{API}/answerCallbackQuery", json={
            "callback_query_id": callback_id,
            "text": text,
        }, timeout=10)
        return True
    except Exception:
        return False


def _send_long(chat_id: int, text: str, buttons: list[list[dict]] | None = None):
    """Split long messages; attach buttons only to the last chunk."""
    if len(text) <= 4000:
        _reply(chat_id, text, buttons=buttons)
        return
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3900:
            chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)
    for i, chunk in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        _reply(chat_id, chunk, buttons=buttons if is_last else None)


# Inline button layouts used across commands
MAIN_MENU_BUTTONS = [
    [
        {"text": "📅 Today's Brief", "callback_data": "/today"},
    ],
    [
        {"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
        {"text": "🔄 Flow", "callback_data": "/flow"},
        {"text": "🎯 Triggers", "callback_data": "/triggers"},
    ],
    [
        {"text": "📦 Delivery", "callback_data": "/delivery"},
        {"text": "📋 Changed", "callback_data": "/changed"},
    ],
]


def _stock_buttons(symbol: str, sector: str | None = None) -> list[list[dict]]:
    """Buttons shown after a stock report."""
    row1 = []
    if sector:
        row1.append({"text": f"🏭 {sector[:20]}", "callback_data": f"/sector {sector}"})
    row1.append({"text": "🎯 Triggers", "callback_data": "/triggers"})
    row2 = [
        {"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
        {"text": "🔄 Flow", "callback_data": "/flow"},
    ]
    return [row1, row2]


# --------------------------------------------------------------------------
# /start and /help
# --------------------------------------------------------------------------

def handle_start(chat_id: int):
    _reply(chat_id, (
        "👋 <b>Welcome to Morrow Desk</b>\n\n"
        "I analyse NSE stocks using sector structure, coil patterns, "
        "volume, delivery and momentum.\n\n"
        "📝 <b>Just type any stock name</b> — <code>TRENT</code>, "
        "<code>RELIANCE</code>, <code>Tata Motors</code>\n\n"
        "Or tap a button below to explore the market 👇"
    ), buttons=MAIN_MENU_BUTTONS)


def handle_help(chat_id: int):
    _reply(chat_id, (
        "📖 <b>Commands</b>\n\n"
        "<b>Stock report:</b>\n"
        "  Just type a symbol: <code>TRENT</code>\n"
        "  Also works: <code>Tata Motors</code>\n\n"
        "<b>Sector:</b>\n"
        "  <code>/sector Retailing</code>\n\n"
        "Tap any button below, or use the ≡ menu 👇"
    ), buttons=MAIN_MENU_BUTTONS)


# --------------------------------------------------------------------------
# /r SYMBOL — stock report (quick mode by default, detail on tap)
# --------------------------------------------------------------------------

def _fmt_report_quick(data: dict) -> str:
    """4-line quick verdict — what 80% of users need."""
    if not data.get("found"):
        near = data.get("near", [])
        if near:
            suggestions = "\n".join(
                f"  • <code>{r['symbol']}</code> — {r.get('sector', '')}"
                for r in near[:6]
            )
            return (f"🔍 <b>{data.get('query', '?')}</b> not found.\n\n"
                    f"Did you mean:\n{suggestions}")
        return f"🔍 <b>{data.get('query', '?')}</b> — not found in the universe."

    sym = data["symbol"]
    sector = data.get("sector") or ""
    klass = data.get("sector_klass") or ""
    phase = data.get("phase_label") or data.get("phase", "")
    m = data.get("metrics") or {}
    plan = data.get("plan") or {}
    why = data.get("why") or ""

    adj = m.get("adj")
    e = KLASS_EMOJI.get(klass, "")
    phase_emoji = {
        "Broke out": "🚀", "Potential": "🎯", "Coiled": "⚡",
        "Near miss": "👀", "At the high": "📈", "Volume break": "💥",
        "Watching": "⏳",
    }.get(phase, "•")

    # Line 1: symbol + price
    lines = [f"📊 <b>{sym}</b> — ₹{adj:,.1f}" if adj else f"📊 <b>{sym}</b>"]

    # Line 2: verdict (phase)
    lines.append(f"\n{phase_emoji} <b>{phase}</b>")

    # Line 3: key context — trigger distance, sector, or action
    trigger = m.get("trigger") or m.get("prior_trigger")
    to_trig = m.get("to_trigger")
    cmf = m.get("cmf")

    context_parts = []
    if to_trig is not None and trigger:
        if abs(to_trig) < 0.02:
            context_parts.append(f"{to_trig * 100:.1f}% from trigger ₹{trigger:,.0f}")
        else:
            context_parts.append(f"Trigger ₹{trigger:,.0f} ({to_trig * 100:+.1f}%)")
    if sector and klass:
        context_parts.append(f"{sector} {e}{klass}")
    if cmf is not None:
        flow_word = "accumulation" if cmf > 0.05 else "distribution" if cmf < -0.05 else ""
        if flow_word:
            context_parts.append(flow_word)
    if context_parts:
        lines.append(" · ".join(context_parts))

    # Line 4: the plain English summary — trimmed to first sentence
    if why:
        first_sentence = why.split(". ")[0] + "." if ". " in why else why[:200]
        lines.append(f"\n💡 <i>{first_sentence}</i>")

    # Trade plan one-liner if available
    stop, target = plan.get("stop"), plan.get("target")
    if stop and target and adj:
        lines.append(f"\n📐 Stop ₹{stop:,.0f} · Target ₹{target:,.0f}")

    return "\n".join(lines)


def _fmt_report_full(data: dict) -> str:
    """Detailed report with all metrics — shown when user taps 'Full report'."""
    if not data.get("found"):
        return _fmt_report_quick(data)

    sym = data["symbol"]
    m = data.get("metrics") or {}
    plan = data.get("plan") or {}
    shape = data.get("shape") or {}
    sector = data.get("sector") or "—"
    klass = data.get("sector_klass") or "—"
    adj = m.get("adj")

    lines = [f"📊 <b>{sym} — Full Report</b>"]
    lines.append(f"As of {data.get('as_of', '?')}\n")

    # Price + Trend
    if adj:
        lines.append(f"💰 ₹{adj:,.1f}")
    ema50, ema200 = m.get("ema50"), m.get("ema200")
    if adj and ema50 and ema200:
        if adj > ema50 > ema200:
            lines.append("✅ Uptrend (above EMA50 & 200)")
        elif adj > ema200:
            lines.append("🟡 Above EMA200, below EMA50")
        else:
            lines.append("🔴 Downtrend (below both EMAs)")

    # Sector
    lines.append(f"\n🏭 <b>{sector}</b> {KLASS_EMOJI.get(klass, '')} {klass}")
    if shape:
        parts = []
        t = shape.get("T")
        if t is not None:
            parts.append(f"T:{t:.0f}%")
        b = shape.get("B")
        if b is not None:
            parts.append(f"B:{b:.0f}%")
        cmf_s = shape.get("cmf")
        if cmf_s is not None:
            parts.append(f"CMF:{cmf_s:+.2f}")
        rs = shape.get("rs")
        if rs is not None:
            parts.append(f"RS:{rs:+.1f}")
        if parts:
            lines.append("   " + " · ".join(parts))

    # Technical
    lines.append(f"\n📈 <b>Technical</b>")
    trigger = m.get("trigger") or m.get("prior_trigger")
    to_trig = m.get("to_trigger")
    if trigger:
        dist = f" ({to_trig * 100:+.1f}%)" if to_trig is not None else ""
        lines.append(f"   Trigger: ₹{trigger:,.1f}{dist}")
    coil = m.get("coil")
    if coil is not None:
        lines.append(f"   Coil: {coil:.1f}")
    rsi = m.get("rsi")
    if rsi is not None:
        lines.append(f"   RSI: {rsi:.0f}")
    pos_hi = m.get("pos_hi")
    if pos_hi is not None:
        lines.append(f"   Position: {pos_hi * 100:.0f}% of 85-day range")

    # Volume & Flow
    lines.append(f"\n📊 <b>Volume & Flow</b>")
    vol_x = m.get("vol_expand")
    if vol_x is not None:
        lines.append(f"   Volume: {vol_x:.1f}× avg")
    med_to = m.get("median_turnover")
    if med_to is not None:
        cr = med_to / 100
        lines.append(f"   Turnover: ₹{cr:.0f}cr/day" if cr >= 1 else f"   Turnover: ₹{cr:.1f}cr/day")
    cmf = m.get("cmf")
    if cmf is not None:
        flow = "accumulation" if cmf > 0.05 else "distribution" if cmf < -0.05 else "neutral"
        lines.append(f"   CMF: {cmf:+.2f} ({flow})")
    deliv = m.get("deliv_pct")
    if deliv is not None:
        lines.append(f"   Delivery: {deliv:.0f}%")

    # Trade plan
    action = plan.get("action_label")
    if action:
        lines.append(f"\n🔔 <b>{action}</b>")
    stop, target, rr = plan.get("stop"), plan.get("target"), plan.get("rr")
    if stop is not None:
        lines.append(f"   Stop: ₹{stop:,.1f}")
    if target is not None:
        lines.append(f"   Target: ₹{target:,.1f}")
    if rr is not None:
        lines.append(f"   R:R = 1:{rr:.1f}")

    # Setup / breakout
    setup = data.get("setup")
    if setup and setup.get("why"):
        lines.append(f"\n💡 {setup['why'][:200]}")
    breakout = data.get("breakout")
    if breakout and breakout.get("why"):
        lines.append(f"\n💥 {breakout['why'][:200]}")

    # Full summary
    why = data.get("why")
    if why:
        lines.append(f"\n📝 <i>{why[:400]}</i>")

    lines.append("\n<i>Observational analysis — not a recommendation.</i>")
    return "\n".join(lines)


def handle_report(chat_id: int, query: str, engine, full: bool = False):
    if not query:
        _reply(chat_id, "Just type a stock name — <code>TRENT</code>, <code>RELIANCE</code>")
        return
    with engine._lock:
        if engine.status != "ready" or engine.stocks is None:
            _reply(chat_id, "⏳ Engine still loading. Try again in a few minutes.")
            return
    try:
        data = engine.stock(query)
    except Exception as exc:
        _reply(chat_id, f"❌ Error: {str(exc)[:200]}")
        return

    sym = data.get("symbol", query.upper())
    sector = data.get("sector") if data.get("found") else None

    if full:
        msg = _fmt_report_full(data)
        btns = _stock_buttons(sym, sector) if data.get("found") else MAIN_MENU_BUTTONS
    else:
        msg = _fmt_report_quick(data)
        # Quick mode: show "Full report" button + sector
        btns = []
        if data.get("found"):
            row1 = [{"text": "📊 Full report", "callback_data": f"/detail {sym}"}]
            if sector:
                row1.append({"text": f"🏭 {sector[:18]}", "callback_data": f"/sector {sector}"})
            btns.append(row1)
            btns.append([
                {"text": "🎯 Triggers", "callback_data": "/triggers"},
                {"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
            ])
        else:
            btns = MAIN_MENU_BUTTONS

    _send_long(chat_id, msg, buttons=btns)


# --------------------------------------------------------------------------
# /sector NAME
# --------------------------------------------------------------------------

def _fmt_sector(data: dict) -> str:
    if not data.get("found"):
        near = data.get("near", [])
        reason = data.get("reason", "")
        if reason:
            return f"🏭 {reason}"
        if near:
            suggestions = "\n".join(f"  • <code>{s}</code>" for s in near[:6])
            return f"🔍 Sector not found. Similar:\n{suggestions}"
        return "🔍 Sector not found."

    name = data.get("sector", "?")
    klass = data.get("klass") or "—"
    shape = data.get("shape") or {}
    buys = data.get("buys") or []
    members = data.get("constituents") or []

    lines = [f"🏭 <b>{name}</b>"]
    lines.append(f"{KLASS_EMOJI.get(klass, '•')} <b>{klass}</b>")
    note = data.get("note")
    if note:
        lines.append(f"<i>{note}</i>")

    lines.append("")
    for key, label in [("T", "📈 Breadth"), ("B", "📊 Width"),
                        ("cmf", "💧 CMF"), ("rs", "💪 RS")]:
        v = shape.get(key)
        if v is not None:
            lines.append(f"{label}: {v:+.2f}" if key in ("cmf", "rs") else f"{label}: {v:.0f}%")

    verdict = shape.get("verdict")
    if verdict:
        lines.append(f"\n📋 <b>Verdict:</b> {verdict}")

    if buys:
        lines.append(f"\n🎯 <b>Setups ({len(buys)}):</b>")
        for r in buys[:5]:
            coil = r.get("coil")
            c = f" · coil {coil:.1f}" if coil else ""
            lines.append(f"  <code>{r.get('symbol', '?')}</code>{c}")
        if len(buys) > 5:
            lines.append(f"  +{len(buys) - 5} more")

    if members:
        lines.append(f"\n👥 <b>Top constituents:</b>")
        for r in members[:8]:
            ret = r.get("ret")
            ret_s = f" {ret * 100:+.1f}%" if ret is not None else ""
            lines.append(f"  {r.get('symbol', '?')}{ret_s}")

    lines.append("\n<i>Sector analysis — not a recommendation.</i>")
    return "\n".join(lines)


def handle_sector(chat_id: int, query: str, engine):
    if not query:
        _reply(chat_id, "Usage: /sector NAME\nExample: <code>/sector Retailing</code>")
        return
    _reply(chat_id, f"🔍 Looking up sector <b>{query}</b>...")
    with engine._lock:
        if engine.status != "ready":
            _reply(chat_id, "⏳ Engine still loading. Try again in a few minutes.")
            return
    try:
        data = engine.sector(query)
    except Exception as exc:
        _reply(chat_id, f"❌ Error: {str(exc)[:200]}")
        return
    if data is None:
        _reply(chat_id, "⏳ Engine not ready.")
        return
    btns = [
        [{"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
         {"text": "🔄 Flow", "callback_data": "/flow"}],
    ]
    _send_long(chat_id, _fmt_sector(data), buttons=btns)


# --------------------------------------------------------------------------
# /today — "just tell me what matters"
# --------------------------------------------------------------------------

def handle_today(chat_id: int, engine):
    with engine._lock:
        if engine.status != "ready":
            _reply(chat_id, "⏳ Engine still loading.")
            return
        scan_rows = engine.scan_rows.copy() if engine.scan_rows is not None and not engine.scan_rows.empty else pd.DataFrame()
        coil_rows = engine.coil_rows.copy() if engine.coil_rows is not None and not engine.coil_rows.empty else pd.DataFrame()
        buys = engine.buys.copy() if engine.buys is not None and not engine.buys.empty else pd.DataFrame()
        coil_stocks = engine.coil_stocks
        as_of = engine.as_of

    lines = [f"📅 <b>Today's Brief</b>", f"As of {as_of}\n"]

    # 1. Top actionable — stocks closest to trigger in buy setups
    actionable = []
    pool = buys if not buys.empty else coil_rows
    if not pool.empty and "to_trigger" in pool.columns:
        near = pool[pool["to_trigger"].notna() & (pool["to_trigger"] <= 0.03)]
        near = near.sort_values("to_trigger").head(3)
        for _, r in near.iterrows():
            sym = r.get("symbol", "?")
            adj = r.get("adj", 0)
            trig = r.get("trigger", 0)
            to_t = r.get("to_trigger", 0)
            sec = r.get("sector", "")
            klass = ""
            if not scan_rows.empty:
                sr = scan_rows[scan_rows["sector"] == sec]
                if not sr.empty:
                    klass = sr.iloc[0].get("klass", "")
            actionable.append((sym, adj, trig, to_t, sec, klass))

    if actionable:
        lines.append("🎯 <b>Actionable</b>")
        for sym, adj, trig, to_t, sec, klass in actionable:
            e = KLASS_EMOJI.get(klass, "")
            lines.append(f"  <b>{sym}</b> ₹{adj:,.0f} → trigger ₹{trig:,.0f} ({to_t * 100:.1f}% away)")
            lines.append(f"  {sec} {e}{klass}")
        lines.append("")
    else:
        lines.append("🎯 No stocks near breakout trigger today.")
        lines.append("<i>The market isn't always offering setups — that's okay.</i>\n")

    # 2. One insight — strongest sector flow or unusual delivery
    if not scan_rows.empty and "cmf" in scan_rows.columns:
        top_cmf = scan_rows[scan_rows["cmf"].notna()].sort_values("cmf", ascending=False)
        if not top_cmf.empty:
            best = top_cmf.iloc[0]
            sec = best["sector"]
            cmf = best["cmf"]
            klass = best.get("klass", "")
            n_adv = int(best.get("n_adv", 0))
            n_stocks = int(best.get("n_stocks", 1))
            adv_pct = n_adv / n_stocks * 100 if n_stocks > 0 else 0
            if cmf > 0.05:
                lines.append("💡 <b>Insight</b>")
                lines.append(f"Strongest money flow: <b>{sec}</b> (CMF {cmf:+.2f})")
                lines.append(f"{adv_pct:.0f}% of stocks advancing · {KLASS_EMOJI.get(klass, '')}{klass}")
                # Check for streak — how many days CMF has been positive
                if coil_stocks is not None and not coil_stocks.empty:
                    try:
                        # Quick check from panel data isn't straightforward, so keep it simple
                        lines.append(f"<i>Positive CMF = institutions accumulating this sector</i>")
                    except Exception:
                        pass
                lines.append("")

            # Also mention worst outflow
            worst = top_cmf.iloc[-1]
            if worst["cmf"] < -0.05:
                lines.append(f"⚠️ Outflow: <b>{worst['sector']}</b> (CMF {worst['cmf']:+.2f})")
                lines.append(f"<i>Money leaving — avoid new positions here</i>")
                lines.append("")

    # 3. Market regime
    if not scan_rows.empty:
        total = len(scan_rows)
        crossing = int((scan_rows["klass"] == "CROSSING").sum())
        pullback = int((scan_rows["klass"] == "PULLBACK").sum())
        bullish_pct = (crossing + pullback) / max(total, 1) * 100
        regime = "BULLISH" if bullish_pct >= 40 else "CAUTIOUS"
        emoji = "🟢" if regime == "BULLISH" else "🟡"
        lines.append(f"📊 <b>Market:</b> {emoji} {regime}")
        lines.append(f"   {crossing} crossing · {pullback} pullback · {bullish_pct:.0f}% in uptrend")

    # Coil pool stat
    if not coil_rows.empty:
        lines.append(f"   {len(coil_rows)} stocks coiled (compressed near highs)")

    lines.append("\n<i>What the data shows today — not a recommendation.</i>")

    btns = [
        [{"text": "🎯 All triggers", "callback_data": "/triggers"},
         {"text": "🗺️ Heatmap", "callback_data": "/heatmap"}],
        [{"text": "📦 Delivery", "callback_data": "/delivery"},
         {"text": "🔄 Flow", "callback_data": "/flow"}],
    ]
    _send_long(chat_id, "\n".join(lines), buttons=btns)


# --------------------------------------------------------------------------
# /heatmap — sector rotation map
# --------------------------------------------------------------------------

def _fmt_sector_row(r) -> str:
    """One-line sector summary for heatmap."""
    sec = r["sector"]
    t = r.get("T", 0)
    cmf = r.get("cmf", 0)
    n_adv = int(r.get("n_adv", 0))
    n_stocks = int(r.get("n_stocks", 1))
    adv_pct = n_adv / n_stocks * 100 if n_stocks > 0 else 0
    cmf_arrow = "↑" if pd.notna(cmf) and cmf > 0.05 else "↓" if pd.notna(cmf) and cmf < -0.05 else ""
    cmf_s = f"CMF {cmf:+.2f}{cmf_arrow}" if pd.notna(cmf) else ""
    return f"  <b>{sec}</b> · T:{t:.0f}% · {adv_pct:.0f}% adv · {cmf_s}"


def handle_heatmap(chat_id: int, engine, full: bool = False):
    with engine._lock:
        if engine.status != "ready" or engine.scan_rows is None or engine.scan_rows.empty:
            _reply(chat_id, "⏳ Engine still loading. Try again in a few minutes.")
            return
        df = engine.scan_rows.copy()
        as_of = engine.as_of

    total = len(df)
    crossing = df[df["klass"] == "CROSSING"]
    pullback = df[df["klass"] == "PULLBACK"]
    n_crossing = len(crossing)
    n_pullback = len(pullback)
    n_actionable = n_crossing + n_pullback
    regime = "BULLISH" if n_actionable / max(total, 1) >= 0.4 else "CAUTIOUS"
    emoji = "🟢" if regime == "BULLISH" else "🟡"

    if full:
        # ── Full view: every sector grouped ──
        lines = [f"🗺️ <b>All Sectors</b>", f"As of {as_of}\n"]

        groups = [
            ("🟢 CROSSING", crossing),
            ("🟡 PULLBACK", pullback),
            ("⚪ BASE", df[df["klass"] == "BASE"]),
            ("🟠 UNVERIFIED", df[df["klass"] == "CROSSING_UNVERIFIED"]),
            ("🔴 DOWN", df[df["klass"].isin(["DOWN", "NONE", "NEGLECT"])]),
            ("⛔ DISQUALIFIED", df[df["klass"] == "DISQUALIFIED"]),
        ]
        for label, grp in groups:
            if grp.empty:
                continue
            lines.append(f"<b>{label}</b> ({len(grp)})")
            for _, r in grp.sort_values("T", ascending=False).iterrows():
                lines.append(_fmt_sector_row(r))
            lines.append("")

        lines.append(f"{emoji} <b>{regime}</b> · {n_actionable}/{total} sectors in uptrend")
        lines.append("\n<i>Sector rotation — not a recommendation.</i>")
        btns = [
            [{"text": "🔄 Flow", "callback_data": "/flow"},
             {"text": "🎯 Triggers", "callback_data": "/triggers"}],
        ]
        _send_long(chat_id, "\n".join(lines), buttons=btns)
        return

    # ── Concise view: only actionable sectors ──
    lines = [f"🗺️ <b>Sector Heatmap</b>", f"As of {as_of}\n"]
    lines.append(f"{emoji} <b>Market: {regime}</b> — {n_actionable}/{total} sectors in uptrend\n")

    if not crossing.empty:
        lines.append(f"🟢 <b>CROSSING</b> ({n_crossing}) — uptrend confirmed")
        for _, r in crossing.sort_values("T", ascending=False).iterrows():
            lines.append(_fmt_sector_row(r))
        lines.append("")

    if not pullback.empty:
        lines.append(f"🟡 <b>PULLBACK</b> ({n_pullback}) — buy zone")
        for _, r in pullback.sort_values("T", ascending=False).iterrows():
            lines.append(_fmt_sector_row(r))
        lines.append("")

    if n_actionable == 0:
        lines.append("No sectors in CROSSING or PULLBACK right now.")
        lines.append("<i>The market isn't always offering setups — that's okay.</i>")

    # Quick counts for the rest
    rest = {
        "Base": len(df[df["klass"] == "BASE"]),
        "Down": len(df[df["klass"].isin(["DOWN", "NONE", "NEGLECT"])]),
        "Disqualified": len(df[df["klass"] == "DISQUALIFIED"]),
    }
    rest_parts = [f"{v} {k.lower()}" for k, v in rest.items() if v > 0]
    if rest_parts:
        lines.append(f"📊 Also: {' · '.join(rest_parts)}")

    lines.append("\n<i>Sector rotation — not a recommendation.</i>")
    btns = [
        [{"text": "📋 Show all sectors", "callback_data": "/heatmap_full"}],
        [{"text": "🔄 Flow", "callback_data": "/flow"},
         {"text": "🎯 Triggers", "callback_data": "/triggers"}],
    ]
    _send_long(chat_id, "\n".join(lines), buttons=btns)


# --------------------------------------------------------------------------
# /flow — money flow into vs out of sectors
# --------------------------------------------------------------------------

def handle_flow(chat_id: int, engine):
    with engine._lock:
        if engine.status != "ready" or engine.scan_rows is None or engine.scan_rows.empty:
            _reply(chat_id, "⏳ Engine still loading.")
            return
        df = engine.scan_rows.copy()
        as_of = engine.as_of

    lines = [f"🔄 <b>Sector Money Flow</b>", f"As of {as_of}\n"]

    # Sort by CMF: most positive first (inflows), most negative last (outflows)
    df = df[df["cmf"].notna()].sort_values("cmf", ascending=False)

    if df.empty:
        _reply(chat_id, "No CMF data available.")
        return

    # Inflows
    inflows = df[df["cmf"] > 0.03]
    if not inflows.empty:
        lines.append("💰 <b>INFLOWS</b> (CMF positive — accumulation)")
        for _, r in inflows.iterrows():
            e = KLASS_EMOJI.get(r["klass"], "•")
            strength = "█" * min(8, max(1, int(r["cmf"] * 40)))
            n_adv = int(r.get("n_adv", 0))
            n_stocks = int(r.get("n_stocks", 1))
            adv = n_adv / n_stocks * 100 if n_stocks else 0
            lines.append(f"  {e} <b>{r['sector']}</b>")
            lines.append(f"    {strength} CMF {r['cmf']:+.2f} · {adv:.0f}% advancing · {r['klass']}")
        lines.append("")

    # Neutral
    neutral = df[(df["cmf"] >= -0.03) & (df["cmf"] <= 0.03)]
    if not neutral.empty:
        lines.append(f"⚖️ <b>NEUTRAL</b> ({len(neutral)} sectors)")
        for _, r in neutral.head(5).iterrows():
            lines.append(f"  {r['sector']} · CMF {r['cmf']:+.2f}")
        if len(neutral) > 5:
            lines.append(f"  +{len(neutral) - 5} more")
        lines.append("")

    # Outflows
    outflows = df[df["cmf"] < -0.03].sort_values("cmf")
    if not outflows.empty:
        lines.append("🚨 <b>OUTFLOWS</b> (CMF negative — distribution)")
        for _, r in outflows.iterrows():
            e = KLASS_EMOJI.get(r["klass"], "•")
            strength = "█" * min(8, max(1, int(abs(r["cmf"]) * 40)))
            lines.append(f"  {e} <b>{r['sector']}</b>")
            lines.append(f"    {strength} CMF {r['cmf']:+.2f} · {r['klass']}")
        lines.append("")

    lines.append("💡 <i>CMF (Chaikin Money Flow) measures buying vs selling pressure "
                 "weighted by volume. Positive = accumulation, negative = distribution.</i>")
    btns = [
        [{"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
         {"text": "🎯 Triggers", "callback_data": "/triggers"},
         {"text": "📦 Delivery", "callback_data": "/delivery"}],
    ]
    _send_long(chat_id, "\n".join(lines), buttons=btns)


# --------------------------------------------------------------------------
# /triggers — stocks near breakout
# --------------------------------------------------------------------------

def handle_triggers(chat_id: int, engine):
    with engine._lock:
        if engine.status != "ready" or engine.coil_rows is None:
            _reply(chat_id, "⏳ Engine still loading.")
            return
        coils = engine.coil_rows.copy() if not engine.coil_rows.empty else pd.DataFrame()
        buys = engine.buys.copy() if engine.buys is not None and not engine.buys.empty else pd.DataFrame()
        as_of = engine.as_of

    # Combine coils and buys, prefer buys
    pool = buys if not buys.empty else coils
    if pool.empty:
        _reply(chat_id, "No coiled stocks right now. The market may not be offering this setup.")
        return

    # Filter to stocks within 2% of trigger
    pool = pool[pool["to_trigger"].notna() & (pool["to_trigger"] <= 0.02)].copy()
    pool = pool.sort_values("to_trigger")

    if pool.empty:
        _reply(chat_id, "📭 No stocks within 2% of their trigger right now.\n\n"
                        "<i>When a coiled stock is within 1-2% of its 20-day high, "
                        "one strong session can close through it — that's the breakout confirmation.</i>")
        return

    lines = [f"🎯 <b>Trigger Watch — {len(pool)} stocks near breakout</b>"]
    lines.append(f"As of {as_of}\n")

    for _, r in pool.head(15).iterrows():
        sym = r.get("symbol", "?")
        adj = r.get("adj", 0)
        trig = r.get("trigger", 0)
        to_trig = r.get("to_trigger", 0)
        sector = r.get("sector", "")
        coil = r.get("coil")
        cmf = r.get("cmf")
        rsi = r.get("rsi")

        pct = to_trig * 100
        if pct < 0.5:
            proximity = "🔴 <b>AT TRIGGER</b>"
        elif pct < 1.0:
            proximity = "🟠 within 1%"
        else:
            proximity = "🟡 within 2%"

        lines.append(f"  <b>{sym}</b> — {proximity}")
        lines.append(f"    ₹{adj:,.1f} → trigger ₹{trig:,.1f} ({pct:.1f}% away)")

        detail_parts = [sector]
        if coil:
            detail_parts.append(f"coil {coil:.1f}")
        if cmf is not None and not (isinstance(cmf, float) and (np.isnan(cmf) or np.isinf(cmf))):
            detail_parts.append(f"CMF {cmf:+.2f}")
        lines.append(f"    {' · '.join(detail_parts)}")

        # Is it in the buy setup list?
        is_buy = not buys.empty and sym in buys["symbol"].values
        if is_buy:
            lines.append("    ✅ In buy setup list")
        lines.append("")

    if len(pool) > 15:
        lines.append(f"  +{len(pool) - 15} more on the dashboard")

    lines.append("💡 <i>A close above the trigger on rising volume = confirmed "
                 "breakout. The trigger is the 20-day high — the price above which "
                 "the base has been overcome.</i>")
    btns = [
        [{"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
         {"text": "📦 Delivery", "callback_data": "/delivery"},
         {"text": "🔄 Flow", "callback_data": "/flow"}],
    ]
    _send_long(chat_id, "\n".join(lines), buttons=btns)


# --------------------------------------------------------------------------
# /delivery — unusual delivery activity
# --------------------------------------------------------------------------

def handle_delivery(chat_id: int, engine):
    with engine._lock:
        if engine.status != "ready" or engine.coil_stocks is None:
            _reply(chat_id, "⏳ Engine still loading.")
            return
        stocks = engine.coil_stocks.copy()
        as_of = engine.as_of

    # Get the latest day
    latest_date = stocks["date"].max()
    today = stocks[stocks["date"] == latest_date].copy()
    if today.empty:
        _reply(chat_id, "No data available.")
        return

    # Need deliv_pct column
    if "deliv_pct" not in today.columns:
        _reply(chat_id, "Delivery data not available in the current panel.")
        return

    # Compute each stock's 20-day average delivery
    results = []
    for _, row in today.iterrows():
        sym = row["symbol"]
        dp = row.get("deliv_pct")
        if pd.isna(dp) or dp <= 0:
            continue
        hist = stocks[(stocks["symbol"] == sym) & (stocks["date"] < latest_date)]
        if len(hist) < 10:
            continue
        avg_dp = hist["deliv_pct"].tail(20).mean()
        if pd.isna(avg_dp) or avg_dp <= 0:
            continue
        ratio = dp / avg_dp
        if ratio >= 1.3 and dp >= 40:
            results.append({
                "symbol": sym,
                "deliv_pct": dp,
                "avg_deliv": avg_dp,
                "ratio": ratio,
                "ret": float(row.get("ret", 0)) if pd.notna(row.get("ret")) else 0,
                "sector": row.get("sector", ""),
                "turnover": float(row.get("turnover", 0)) if pd.notna(row.get("turnover")) else 0,
            })

    if not results:
        _reply(chat_id, "📭 No unusual delivery activity today.\n\n"
                        "<i>Unusual delivery = delivery % significantly above the stock's "
                        "own 20-day average. High delivery means physical settlement, "
                        "which typically indicates institutional positions.</i>")
        return

    # Sort by ratio, take top
    results.sort(key=lambda x: -x["ratio"])

    # Split into accumulation (price up) and distribution (price down)
    accum = [r for r in results if r["ret"] > 0.005]
    distrib = [r for r in results if r["ret"] < -0.005]

    lines = [f"📦 <b>Unusual Delivery</b>"]
    lines.append(f"As of {as_of}\n")

    if accum:
        lines.append(f"💰 <b>ACCUMULATION</b> (high delivery + price rising)")
        lines.append(f"<i>Institutions may be building positions</i>\n")
        for r in accum[:10]:
            cr = r["turnover"] / 1e7
            lines.append(f"  <b>{r['symbol']}</b>")
            lines.append(f"    Delivery {r['deliv_pct']:.0f}% (avg {r['avg_deliv']:.0f}%) · "
                         f"{r['ratio']:.1f}× normal")
            lines.append(f"    {r['ret'] * 100:+.1f}% · {r['sector']} · ₹{cr:.0f}cr turnover")
            lines.append("")

    if distrib:
        lines.append(f"🚨 <b>DISTRIBUTION</b> (high delivery + price falling)")
        lines.append(f"<i>Institutions may be exiting positions</i>\n")
        for r in distrib[:10]:
            cr = r["turnover"] / 1e7
            lines.append(f"  <b>{r['symbol']}</b>")
            lines.append(f"    Delivery {r['deliv_pct']:.0f}% (avg {r['avg_deliv']:.0f}%) · "
                         f"{r['ratio']:.1f}× normal")
            lines.append(f"    {r['ret'] * 100:+.1f}% · {r['sector']} · ₹{cr:.0f}cr turnover")
            lines.append("")

    lines.append(f"📊 {len(results)} stocks with unusual delivery today")
    lines.append("\n💡 <i>Delivery % = shares actually transferred to demat accounts "
                 "(not squared off intraday). High delivery on rising prices suggests "
                 "institutional buying. NSE-unique metric.</i>")
    btns = [
        [{"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
         {"text": "🎯 Triggers", "callback_data": "/triggers"},
         {"text": "📋 Changed", "callback_data": "/changed"}],
    ]
    _send_long(chat_id, "\n".join(lines), buttons=btns)


# --------------------------------------------------------------------------
# /changed — what changed since yesterday
# --------------------------------------------------------------------------

def handle_changed(chat_id: int, engine):
    with engine._lock:
        if engine.status != "ready":
            _reply(chat_id, "⏳ Engine still loading.")
            return
        scan_rows = engine.scan_rows.copy() if engine.scan_rows is not None and not engine.scan_rows.empty else pd.DataFrame()
        buys = engine.buys.copy() if engine.buys is not None and not engine.buys.empty else pd.DataFrame()
        coil_rows = engine.coil_rows.copy() if engine.coil_rows is not None and not engine.coil_rows.empty else pd.DataFrame()
        as_of = engine.as_of

    # Load previous sector scans from DB
    import db
    sector_changes = []
    new_setups = []
    lost_setups = []

    try:
        prev_scans = db.get_all_sectors_latest()
        if prev_scans:
            prev_map = {}
            for r in prev_scans:
                sd = str(r.get("scan_date", ""))
                if sd != str(as_of):
                    prev_map[r["sector"]] = r.get("klass", "")

            if prev_map and not scan_rows.empty:
                for _, r in scan_rows.iterrows():
                    sec = r["sector"]
                    new_k = r["klass"]
                    old_k = prev_map.get(sec)
                    if old_k and old_k != new_k:
                        # Determine if upgrade or downgrade
                        order = {"CROSSING": 0, "PULLBACK": 1, "CROSSING_UNVERIFIED": 2,
                                 "BASE": 3, "NONE": 4, "NEGLECT": 5, "DOWN": 6, "DISQUALIFIED": 7}
                        old_o = order.get(old_k, 5)
                        new_o = order.get(new_k, 5)
                        arrow = "↗️" if new_o < old_o else "↘️"
                        sector_changes.append((sec, old_k, new_k, arrow))
    except Exception as exc:
        print(f"[tg-bot] changed: prev scans error: {exc}")

    # New setups vs previous day's setups from DB
    try:
        prev_setups = db.get_setups(None)
        if prev_setups:
            prev_syms = {r["symbol"] for r in prev_setups if r.get("symbol")}
            if not buys.empty:
                curr_syms = set(buys["symbol"].astype(str))
                new_setups = sorted(curr_syms - prev_syms)
                lost_setups = sorted(prev_syms - curr_syms)
    except Exception:
        pass

    lines = [f"📋 <b>What Changed</b>"]
    lines.append(f"As of {as_of}\n")

    anything = False

    if sector_changes:
        anything = True
        lines.append("<b>Sector moves:</b>")
        for sec, old_k, new_k, arrow in sector_changes:
            e_old = KLASS_EMOJI.get(old_k, "•")
            e_new = KLASS_EMOJI.get(new_k, "•")
            lines.append(f"  {arrow} <b>{sec}</b>: {e_old}{old_k} → {e_new}{new_k}")
        lines.append("")

    if new_setups:
        anything = True
        lines.append("<b>New setups entered:</b>")
        for sym in new_setups[:10]:
            buy_row = buys[buys["symbol"] == sym].iloc[0] if not buys.empty and sym in buys["symbol"].values else None
            sec = buy_row["sector"] if buy_row is not None and "sector" in buy_row.index else ""
            coil = f" · coil {buy_row['coil']:.1f}" if buy_row is not None and "coil" in buy_row.index and pd.notna(buy_row["coil"]) else ""
            lines.append(f"  ⚡ <code>{sym}</code> — {sec}{coil}")
        if len(new_setups) > 10:
            lines.append(f"  +{len(new_setups) - 10} more")
        lines.append("")

    if lost_setups:
        anything = True
        lines.append("<b>Setups removed:</b>")
        for sym in lost_setups[:10]:
            lines.append(f"  ❌ <code>{sym}</code>")
        if len(lost_setups) > 10:
            lines.append(f"  +{len(lost_setups) - 10} more")
        lines.append("")

    # Coil stats
    if not coil_rows.empty:
        n_coil = len(coil_rows)
        near_trig = coil_rows[coil_rows["to_trigger"].notna() & (coil_rows["to_trigger"] <= 0.02)]
        lines.append(f"<b>Coil pool:</b> {n_coil} stocks coiled, {len(near_trig)} within 2% of trigger")
        anything = True

    if not scan_rows.empty:
        crossing = int((scan_rows["klass"] == "CROSSING").sum())
        pullback = int((scan_rows["klass"] == "PULLBACK").sum())
        total = len(scan_rows)
        lines.append(f"<b>Sectors:</b> {crossing} crossing · {pullback} pullback · {total} total")
        anything = True

    if not anything:
        lines.append("No significant changes detected.\n"
                     "<i>This can happen on quiet days or when the previous "
                     "session's data isn't saved yet.</i>")

    lines.append("\n<i>Changes compared to the most recent saved session.</i>")
    _send_long(chat_id, "\n".join(lines), buttons=MAIN_MENU_BUTTONS)


# --------------------------------------------------------------------------
# Webhook dispatcher
# --------------------------------------------------------------------------

def _dispatch(chat_id: int, user_id: int, text: str, engine) -> None:
    """Route a text command (from message or button callback)."""
    lower = text.lower().strip()
    cmd = lower.split()[0] if lower else ""

    # Free commands — always available
    if lower == "/start":
        handle_start(chat_id)
        return
    if lower == "/help":
        handle_help(chat_id)
        return

    # Paid-gated market views — check limit
    if cmd in _PAID_COMMANDS:
        if not _check_limit(user_id, cmd):
            _reply(chat_id, _limit_msg(), buttons=[
                [{"text": "📊 Try a stock report (free)", "callback_data": "/help"}],
            ])
            return

    if lower == "/heatmap":
        handle_heatmap(chat_id, engine)
    elif lower == "/heatmap_full":
        handle_heatmap(chat_id, engine, full=True)
    elif lower == "/flow":
        handle_flow(chat_id, engine)
    elif lower == "/triggers":
        handle_triggers(chat_id, engine)
    elif lower == "/delivery":
        handle_delivery(chat_id, engine)
    elif lower == "/changed":
        handle_changed(chat_id, engine)
    elif lower == "/today":
        handle_today(chat_id, engine)
    elif lower.startswith("/detail "):
        query = text.split(maxsplit=1)[1] if " " in text else ""
        handle_report(chat_id, query.strip(), engine, full=True)
    elif lower.startswith("/r ") or lower.startswith("/report "):
        query = text.split(maxsplit=1)[1] if " " in text else ""
        handle_report(chat_id, query.strip(), engine)
    elif lower.startswith("/sector "):
        query = text.split(maxsplit=1)[1] if " " in text else ""
        handle_sector(chat_id, query.strip(), engine)
    elif text.startswith("/"):
        _reply(chat_id, "Unknown command. Try /help", buttons=MAIN_MENU_BUTTONS)
    else:
        # Plain text = stock lookup (always free)
        handle_report(chat_id, text.strip(), engine)

    # Show remaining count for free users after a gated command
    if cmd in _PAID_COMMANDS:
        remaining = _remaining(user_id)
        if remaining is not None and remaining <= 2:
            note = (f"💡 {remaining} free market view{'s' if remaining != 1 else ''} "
                    f"left today. Stock reports are always free.")
            _reply(chat_id, note)


# Deduplication: track recent update IDs to avoid processing retries
_recent_updates: dict[int, float] = {}
_DEDUP_TTL = 300  # 5 min


def _is_duplicate(update_id: int) -> bool:
    """Return True if we already processed this update."""
    import time as _time
    now = _time.time()

    # Clean old entries
    stale = [k for k, t in _recent_updates.items() if now - t > _DEDUP_TTL]
    for k in stale:
        del _recent_updates[k]

    if update_id in _recent_updates:
        return True
    _recent_updates[update_id] = now
    return False


def process_update(update: dict, engine) -> None:
    """Process one Telegram update from the webhook."""

    # Deduplicate: Telegram retries if our response was slow
    update_id = update.get("update_id")
    if update_id and _is_duplicate(update_id):
        return

    # Handle button presses (callback_query)
    cb = update.get("callback_query")
    if cb:
        chat_id = cb.get("message", {}).get("chat", {}).get("id")
        user_id = cb.get("from", {}).get("id") or chat_id
        data = cb.get("data", "")
        cb_id = cb.get("id", "")
        if chat_id and data:
            _answer_callback(cb_id)
            _dispatch(chat_id, user_id, data, engine)
        return

    # Handle text messages
    msg = update.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = msg.get("chat", {}).get("id")
    user_id = msg.get("from", {}).get("id") or chat_id

    if not chat_id or not text:
        return

    # Only respond in private DMs
    chat_type = msg.get("chat", {}).get("type", "private")
    if chat_type not in ("private",):
        return

    _dispatch(chat_id, user_id, text, engine)


# --------------------------------------------------------------------------
# Webhook management
# --------------------------------------------------------------------------

def set_commands() -> bool:
    """Register the bot command menu with Telegram."""
    if not BOT_TOKEN:
        return False
    try:
        r = requests.post(f"{API}/setMyCommands", json={
            "commands": BOT_COMMANDS,
        }, timeout=15)
        print(f"[telegram] setMyCommands: {r.status_code}")
        return r.ok
    except Exception as exc:
        print(f"[telegram] setMyCommands failed: {exc}")
        return False


def set_webhook(url: str) -> bool:
    if not BOT_TOKEN:
        return False
    try:
        r = requests.post(f"{API}/setWebhook", json={
            "url": url,
            "allowed_updates": ["message", "callback_query"],
        }, timeout=15)
        print(f"[telegram] setWebhook: {r.status_code} {r.text[:200]}")
        # Also register the command menu
        set_commands()
        return r.ok
    except Exception as exc:
        print(f"[telegram] setWebhook failed: {exc}")
        return False


def delete_webhook() -> bool:
    if not BOT_TOKEN:
        return False
    try:
        r = requests.post(f"{API}/deleteWebhook", timeout=15)
        return r.ok
    except Exception:
        return False
