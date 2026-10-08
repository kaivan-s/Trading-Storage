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
API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Bot commands shown in Telegram's / menu
BOT_COMMANDS = [
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
        {"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
        {"text": "🔄 Flow", "callback_data": "/flow"},
        {"text": "🎯 Triggers", "callback_data": "/triggers"},
    ],
    [
        {"text": "📦 Delivery", "callback_data": "/delivery"},
        {"text": "📋 Changed", "callback_data": "/changed"},
        {"text": "📖 Help", "callback_data": "/help"},
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
# /r SYMBOL — stock report
# --------------------------------------------------------------------------

def _fmt_report(data: dict) -> str:
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
    sector = data.get("sector") or "—"
    klass = data.get("sector_klass") or "—"
    phase = data.get("phase_label") or data.get("phase", "—")
    m = data.get("metrics") or {}
    plan = data.get("plan") or {}
    shape = data.get("shape") or {}

    lines = [f"📊 <b>{sym}</b>"]
    lines.append(f"As of {data.get('as_of', '?')}\n")

    adj = m.get("adj")
    if adj:
        lines.append(f"💰 Price: <b>₹{adj:,.1f}</b>")

    e = KLASS_EMOJI.get(klass, "•")
    lines.append(f"\n🏭 <b>Sector:</b> {sector}")
    lines.append(f"   {e} {klass}")

    if shape:
        t = shape.get("T")
        b = shape.get("B")
        rs = shape.get("rs")
        if t is not None:
            lines.append(f"   Breadth (T): {t:.0f}% {'📈' if t > 50 else '📉'}")
        if b is not None:
            lines.append(f"   Width (B): {b:.0f}%")
        if rs is not None:
            lines.append(f"   RS: {rs:+.1f}")

    phase_emoji = {
        "Broke out": "🚀", "Potential": "🎯", "Coiled": "⚡",
        "Near miss": "👀", "At the high": "📈", "Volume break": "💥",
        "Watching": "⏳",
    }.get(phase, "•")
    lines.append(f"\n{phase_emoji} <b>Status:</b> {phase}")

    lines.append(f"\n📈 <b>Technical:</b>")
    ema50, ema200 = m.get("ema50"), m.get("ema200")
    if adj and ema50 and ema200:
        if adj > ema50 > ema200:
            lines.append("   ✅ Above EMA50 & EMA200 (uptrend)")
        elif adj > ema200:
            lines.append("   🟡 Above EMA200, below EMA50")
        else:
            lines.append("   🔴 Below both EMAs (downtrend)")

    trigger = m.get("trigger") or m.get("prior_trigger")
    to_trig = m.get("to_trigger")
    if trigger:
        dist = f" ({to_trig * 100:+.1f}%)" if to_trig is not None else ""
        lines.append(f"   🎯 Trigger: ₹{trigger:,.1f}{dist}")

    coil = m.get("coil")
    if coil is not None:
        lines.append(f"   ⚡ Coil score: {coil:.1f}")
    rsi = m.get("rsi")
    if rsi is not None:
        lines.append(f"   {'🔴' if rsi > 70 else '🟢' if rsi < 30 else '•'} RSI: {rsi:.0f}")
    pos_hi = m.get("pos_hi")
    if pos_hi is not None:
        lines.append(f"   📏 Position: {pos_hi * 100:.0f}% of 85-day range")

    lines.append(f"\n📊 <b>Volume & Flow:</b>")
    vol_x = m.get("vol_expand")
    if vol_x is not None:
        lines.append(f"   Volume: {vol_x:.1f}× 20-day avg")
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

    action = plan.get("action_label")
    if action:
        lines.append(f"\n🔔 <b>Structure:</b> {action}")
    stop, target, rr = plan.get("stop"), plan.get("target"), plan.get("rr")
    if stop is not None:
        lines.append(f"   Stop: ₹{stop:,.1f}")
    if target is not None:
        lines.append(f"   2R target: ₹{target:,.1f}")
    if rr is not None:
        lines.append(f"   R:R = 1:{rr:.1f}")

    setup = data.get("setup")
    if setup and setup.get("why"):
        lines.append(f"\n💡 <b>Setup:</b> {setup['why'][:200]}")
    breakout = data.get("breakout")
    if breakout and breakout.get("why"):
        lines.append(f"\n💥 <b>Breakout:</b> {breakout['why'][:200]}")

    why = data.get("why")
    if why:
        lines.append(f"\n📝 <b>Summary:</b>\n<i>{why[:400]}</i>")

    lines.append("\n<i>Observational analysis — not a recommendation.</i>")
    return "\n".join(lines)


def handle_report(chat_id: int, query: str, engine):
    if not query:
        _reply(chat_id, "Usage: /r SYMBOL\nExample: <code>/r TRENT</code>")
        return
    _reply(chat_id, f"🔍 Looking up <b>{query.upper()}</b>...")
    with engine._lock:
        if engine.status != "ready" or engine.stocks is None:
            _reply(chat_id, "⏳ Engine still loading. Try again in a few minutes.")
            return
    try:
        data = engine.stock(query)
    except Exception as exc:
        _reply(chat_id, f"❌ Error: {str(exc)[:200]}")
        return
    # Contextual buttons after the report
    sector = data.get("sector") if data.get("found") else None
    btns = _stock_buttons(data.get("symbol", ""), sector) if data.get("found") else MAIN_MENU_BUTTONS
    _send_long(chat_id, _fmt_report(data), buttons=btns)


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
# /heatmap — sector rotation map
# --------------------------------------------------------------------------

def handle_heatmap(chat_id: int, engine):
    with engine._lock:
        if engine.status != "ready" or engine.scan_rows is None or engine.scan_rows.empty:
            _reply(chat_id, "⏳ Engine still loading. Try again in a few minutes.")
            return
        df = engine.scan_rows.copy()
        as_of = engine.as_of

    lines = [f"🗺️ <b>Sector Heatmap</b>", f"As of {as_of}\n"]

    # Group by classification
    groups = {
        "🟢 CROSSING": df[df["klass"] == "CROSSING"],
        "🟡 PULLBACK": df[df["klass"] == "PULLBACK"],
        "⚪ BASE": df[df["klass"] == "BASE"],
        "🟠 UNVERIFIED": df[df["klass"] == "CROSSING_UNVERIFIED"],
        "🔴 DOWN": df[df["klass"].isin(["DOWN", "NONE", "NEGLECT"])],
        "⛔ DISQUALIFIED": df[df["klass"] == "DISQUALIFIED"],
    }

    total = len(df)
    for label, grp in groups.items():
        if grp.empty:
            continue
        lines.append(f"<b>{label}</b> ({len(grp)} sectors)")
        grp_sorted = grp.sort_values("T", ascending=False)
        for _, r in grp_sorted.iterrows():
            sec = r["sector"]
            t = r.get("T", 0)
            cmf = r.get("cmf", 0)
            n_adv = int(r.get("n_adv", 0))
            n_stocks = int(r.get("n_stocks", 1))
            adv_pct = n_adv / n_stocks * 100 if n_stocks > 0 else 0
            bar = "█" * max(1, int(t / 10)) if pd.notna(t) else ""
            cmf_s = f"CMF {cmf:+.2f}" if pd.notna(cmf) else ""
            lines.append(f"  {sec}")
            lines.append(f"    {bar} T:{t:.0f}% · {adv_pct:.0f}% adv · {cmf_s}")
        lines.append("")

    # Summary
    crossing = len(groups["🟢 CROSSING"])
    pullback = len(groups["🟡 PULLBACK"])
    down = len(groups["🔴 DOWN"])
    regime = "BULLISH" if (crossing + pullback) / max(total, 1) >= 0.4 else "CAUTIOUS"
    lines.append(f"📊 <b>Market regime: {regime}</b>")
    lines.append(f"   {crossing} crossing · {pullback} pullback · {down} down")
    lines.append(f"   {(crossing + pullback) / max(total, 1) * 100:.0f}% of sectors in uptrend")

    lines.append("\n<i>Sector rotation analysis — not a recommendation.</i>")
    btns = [
        [{"text": "🔄 Flow", "callback_data": "/flow"},
         {"text": "🎯 Triggers", "callback_data": "/triggers"},
         {"text": "📋 Changed", "callback_data": "/changed"}],
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

def _dispatch(chat_id: int, text: str, engine) -> None:
    """Route a text command (from message or button callback)."""
    lower = text.lower().strip()

    if lower == "/start":
        handle_start(chat_id)
    elif lower == "/help":
        handle_help(chat_id)
    elif lower == "/heatmap":
        handle_heatmap(chat_id, engine)
    elif lower == "/flow":
        handle_flow(chat_id, engine)
    elif lower == "/triggers":
        handle_triggers(chat_id, engine)
    elif lower == "/delivery":
        handle_delivery(chat_id, engine)
    elif lower == "/changed":
        handle_changed(chat_id, engine)
    elif lower.startswith("/r ") or lower.startswith("/report "):
        query = text.split(maxsplit=1)[1] if " " in text else ""
        handle_report(chat_id, query.strip(), engine)
    elif lower.startswith("/sector "):
        query = text.split(maxsplit=1)[1] if " " in text else ""
        handle_sector(chat_id, query.strip(), engine)
    elif text.startswith("/"):
        _reply(chat_id, "Unknown command. Try /help", buttons=MAIN_MENU_BUTTONS)
    else:
        handle_report(chat_id, text.strip(), engine)


def process_update(update: dict, engine) -> None:
    """Process one Telegram update from the webhook."""

    # Handle button presses (callback_query)
    cb = update.get("callback_query")
    if cb:
        chat_id = cb.get("message", {}).get("chat", {}).get("id")
        data = cb.get("data", "")
        cb_id = cb.get("id", "")
        if chat_id and data:
            _answer_callback(cb_id)
            _dispatch(chat_id, data, engine)
        return

    # Handle text messages
    msg = update.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = msg.get("chat", {}).get("id")

    if not chat_id or not text:
        return

    # Only respond in private DMs
    chat_type = msg.get("chat", {}).get("type", "private")
    if chat_type not in ("private",):
        return

    _dispatch(chat_id, text, engine)


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
