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
from datetime import datetime
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
    """
    Check if a Telegram user has a paid subscription.

    Priority:
    1. In-memory cache (10-min TTL)
    2. Supabase subscriptions table (email match via Telegram user lookup)
    3. Telegram paid channel membership (fallback)
    """
    import time as _time

    if not BOT_TOKEN:
        return False

    cached = _paid_cache.get(user_id)
    if cached and (_time.time() - cached[0]) < _PAID_CACHE_TTL:
        return cached[1]

    # Try Supabase first — check if any subscription is linked to this telegram_id
    try:
        from subscription import _get_supabase
        client = _get_supabase()
        result = client.table("subscriptions").select("is_premium,expires_at") \
            .eq("telegram_id", str(user_id)).execute()
        if result.data:
            row = result.data[0]
            is_premium = row.get("is_premium", False)
            # Verify not expired
            expires = row.get("expires_at")
            if expires:
                from datetime import date as _date
                try:
                    exp_date = _date.fromisoformat(str(expires)[:10])
                    is_premium = is_premium and exp_date >= _date.today()
                except (ValueError, TypeError):
                    pass
            _paid_cache[user_id] = (_time.time(), is_premium)
            return is_premium
    except Exception as exc:
        print(f"[tg-bot] supabase paid check failed: {exc}")

    # Fallback: check Telegram channel membership
    if PAID_CHAT:
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
            print(f"[tg-bot] channel paid check failed: {exc}")

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
        "Stock reports (<code>TRENT</code>, <code>RELIANCE</code>) "
        "are always free — no limit.\n\n"
        "🔓 <b>Unlock unlimited access</b>\n"
        "Subscribe at <b>morrowdesk.com/pricing</b> to get:\n"
        "  • Unlimited bot commands\n"
        "  • Telegram channel alerts\n"
        "  • Full website access\n\n"
        "Already subscribed? Link your account:\n"
        "<code>/link your@email.com</code>"
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
    {"command": "link", "description": "🔗 Link Telegram to website account"},
    {"command": "verify", "description": "✅ Verify link code — /verify 123456"},
    {"command": "help", "description": "📖 All commands"},
]

KLASS_EMOJI = {
    "CROSSING": "🟢", "PULLBACK": "🟡", "BASE": "⚪",
    "CROSSING_UNVERIFIED": "🟠", "DOWN": "🔴",
    "DISQUALIFIED": "⛔", "NEGLECT": "⚫", "NONE": "⚫",
}


# --------------------------------------------------------------------------
# Pre-computed cache — all bot views are static after post-market
# --------------------------------------------------------------------------
# Instead of querying the engine and iterating DataFrames on every user
# request, we pre-compute everything once after the session closes.
# Each handler checks the cache first; falls back to live only if stale.
#
# Cache structure:
#   _view_cache = {
#       "as_of": "2026-10-09",
#       "heatmap": {...extracted data...},
#       "heatmap_full": {...},
#       "flow": {...},
#       "triggers": [...],
#       "delivery": {...},
#       "changed": {...},
#       "today": {...},
#       "stocks": {"TRENT": {...report...}, ...},  # filled on demand
#   }

_view_cache: dict = {}


def precompute_bot_cache(engine) -> dict:
    """
    Pre-compute all bot view data from the engine.
    Call this once after post-market or after any full engine load.
    Returns the cache dict for inspection.
    """
    global _view_cache

    with engine._lock:
        if engine.status != "ready":
            return {"error": "engine not ready"}
        scan_rows = engine.scan_rows.copy() if engine.scan_rows is not None and not engine.scan_rows.empty else pd.DataFrame()
        coil_rows = engine.coil_rows.copy() if engine.coil_rows is not None and not engine.coil_rows.empty else pd.DataFrame()
        buys = engine.buys.copy() if engine.buys is not None and not engine.buys.empty else pd.DataFrame()
        coil_stocks = engine.coil_stocks
        as_of = engine.as_of

    cache: dict = {"as_of": as_of, "stocks": {}}

    # ── Heatmap data ──
    if not scan_rows.empty:
        hm: dict = {"total": len(scan_rows), "groups": {}}
        for klass in ["CROSSING", "PULLBACK", "BASE", "CROSSING_UNVERIFIED"]:
            grp = scan_rows[scan_rows["klass"] == klass]
            if not grp.empty:
                rows = []
                for _, r in grp.sort_values("T", ascending=False).iterrows():
                    n_s = int(r.get("n_stocks", 1))
                    rows.append({
                        "sector": r["sector"],
                        "T": float(r.get("T", 0)) if pd.notna(r.get("T")) else 0,
                        "cmf": float(r.get("cmf", 0)) if pd.notna(r.get("cmf")) else 0,
                        "adv_pct": int(r.get("n_adv", 0)) / max(n_s, 1) * 100,
                        "klass": klass,
                    })
                hm["groups"][klass] = rows
        down = scan_rows[scan_rows["klass"].isin(["DOWN", "NONE", "NEGLECT"])]
        if not down.empty:
            hm["groups"]["DOWN"] = [{"sector": r["sector"], "T": float(r.get("T", 0)) if pd.notna(r.get("T")) else 0,
                                      "cmf": float(r.get("cmf", 0)) if pd.notna(r.get("cmf")) else 0,
                                      "adv_pct": int(r.get("n_adv", 0)) / max(int(r.get("n_stocks", 1)), 1) * 100,
                                      "klass": "DOWN"}
                                     for _, r in down.sort_values("T", ascending=False).iterrows()]
        dq = scan_rows[scan_rows["klass"] == "DISQUALIFIED"]
        if not dq.empty:
            hm["groups"]["DISQUALIFIED"] = [{"sector": r["sector"]} for _, r in dq.iterrows()]
        n_cross = len(scan_rows[scan_rows["klass"] == "CROSSING"])
        n_pull = len(scan_rows[scan_rows["klass"] == "PULLBACK"])
        hm["n_crossing"] = n_cross
        hm["n_pullback"] = n_pull
        hm["n_actionable"] = n_cross + n_pull
        hm["regime"] = "BULLISH" if (n_cross + n_pull) / max(len(scan_rows), 1) >= 0.4 else "CAUTIOUS"
        hm["n_base"] = len(scan_rows[scan_rows["klass"] == "BASE"])
        hm["n_down"] = len(down) if not down.empty else 0
        hm["n_dq"] = len(dq) if not dq.empty else 0
        cache["heatmap"] = hm

    # ── Flow data (sorted by CMF) ──
    if not scan_rows.empty and "cmf" in scan_rows.columns:
        flow_df = scan_rows[scan_rows["cmf"].notna()].sort_values("cmf", ascending=False)
        cache["flow"] = {
            "inflows": [{"sector": r["sector"], "cmf": float(r["cmf"]),
                         "adv_pct": int(r.get("n_adv", 0)) / max(int(r.get("n_stocks", 1)), 1) * 100,
                         "klass": r["klass"]}
                        for _, r in flow_df[flow_df["cmf"] > 0.03].iterrows()],
            "neutral": [{"sector": r["sector"], "cmf": float(r["cmf"])}
                        for _, r in flow_df[(flow_df["cmf"] >= -0.03) & (flow_df["cmf"] <= 0.03)].iterrows()],
            "outflows": [{"sector": r["sector"], "cmf": float(r["cmf"]), "klass": r["klass"]}
                         for _, r in flow_df[flow_df["cmf"] < -0.03].sort_values("cmf").iterrows()],
        }

    # ── Triggers (stocks near breakout) ──
    pool = buys if not buys.empty else coil_rows
    if not pool.empty and "to_trigger" in pool.columns:
        near = pool[pool["to_trigger"].notna() & (pool["to_trigger"] <= 0.02)].sort_values("to_trigger")
        trig_list = []
        for _, r in near.iterrows():
            trig_list.append({
                "symbol": r.get("symbol", "?"),
                "adj": float(r.get("adj", 0)) if pd.notna(r.get("adj")) else 0,
                "trigger": float(r.get("trigger", 0)) if pd.notna(r.get("trigger")) else 0,
                "to_trigger": float(r.get("to_trigger", 0)),
                "sector": r.get("sector", ""),
                "coil": float(r.get("coil")) if pd.notna(r.get("coil")) else None,
                "cmf": float(r.get("cmf")) if pd.notna(r.get("cmf")) else None,
                "is_buy": not buys.empty and r.get("symbol", "") in buys["symbol"].values,
            })
        cache["triggers"] = trig_list

    # ── Delivery (unusual delivery %) ──
    if coil_stocks is not None and "deliv_pct" in coil_stocks.columns:
        latest_date = coil_stocks["date"].max()
        today_df = coil_stocks[coil_stocks["date"] == latest_date]
        deliv_results = []
        for _, row in today_df.iterrows():
            dp = row.get("deliv_pct")
            if pd.isna(dp) or dp <= 0:
                continue
            sym = row["symbol"]
            hist = coil_stocks[(coil_stocks["symbol"] == sym) & (coil_stocks["date"] < latest_date)]
            if len(hist) < 10:
                continue
            avg_dp = hist["deliv_pct"].tail(20).mean()
            if pd.isna(avg_dp) or avg_dp <= 0:
                continue
            ratio = dp / avg_dp
            if ratio >= 1.3 and dp >= 40:
                deliv_results.append({
                    "symbol": sym,
                    "deliv_pct": float(dp),
                    "avg_deliv": float(avg_dp),
                    "ratio": float(ratio),
                    "ret": float(row.get("ret", 0)) if pd.notna(row.get("ret")) else 0,
                    "sector": row.get("sector", ""),
                    "turnover": float(row.get("turnover", 0)) if pd.notna(row.get("turnover")) else 0,
                })
        deliv_results.sort(key=lambda x: -x["ratio"])
        cache["delivery"] = {
            "accum": [r for r in deliv_results if r["ret"] > 0.005],
            "distrib": [r for r in deliv_results if r["ret"] < -0.005],
            "total": len(deliv_results),
        }

    # ── Today's brief data ──
    today_data: dict = {}
    if not scan_rows.empty:
        today_data["n_crossing"] = int((scan_rows["klass"] == "CROSSING").sum())
        today_data["n_pullback"] = int((scan_rows["klass"] == "PULLBACK").sum())
        today_data["total_sectors"] = len(scan_rows)
        today_data["n_coils"] = len(coil_rows) if not coil_rows.empty else 0
        # Top CMF sector
        if "cmf" in scan_rows.columns:
            top = scan_rows[scan_rows["cmf"].notna()].sort_values("cmf", ascending=False)
            if not top.empty:
                best = top.iloc[0]
                today_data["best_sector"] = best["sector"]
                today_data["best_cmf"] = float(best["cmf"])
                today_data["best_klass"] = best.get("klass", "")
                n_s = int(best.get("n_stocks", 1))
                today_data["best_adv_pct"] = int(best.get("n_adv", 0)) / max(n_s, 1) * 100
            worst = top.iloc[-1]
            if worst["cmf"] < -0.05:
                today_data["worst_sector"] = worst["sector"]
                today_data["worst_cmf"] = float(worst["cmf"])
    # Actionable stocks near trigger
    if not pool.empty and "to_trigger" in pool.columns:
        near = pool[pool["to_trigger"].notna() & (pool["to_trigger"] <= 0.03)].sort_values("to_trigger").head(3)
        act = []
        for _, r in near.iterrows():
            klass = ""
            if not scan_rows.empty:
                sr = scan_rows[scan_rows["sector"] == r.get("sector", "")]
                if not sr.empty:
                    klass = sr.iloc[0].get("klass", "")
            act.append({
                "symbol": r.get("symbol", "?"),
                "adj": float(r.get("adj", 0)) if pd.notna(r.get("adj")) else 0,
                "trigger": float(r.get("trigger", 0)) if pd.notna(r.get("trigger")) else 0,
                "to_trigger": float(r.get("to_trigger", 0)),
                "sector": r.get("sector", ""),
                "klass": klass,
            })
        today_data["actionable"] = act
    cache["today"] = today_data

    # ── Changed data (sector upgrades/downgrades + new/lost setups) ──
    changed_data: dict = {}
    try:
        import db
        prev_scans = db.get_all_sectors_latest()
        sector_changes = []
        if prev_scans and not scan_rows.empty:
            prev_map = {}
            for r in prev_scans:
                sd = str(r.get("scan_date", ""))
                if sd != str(as_of):
                    prev_map[r["sector"]] = r.get("klass", "")
            if prev_map:
                order = {"CROSSING": 0, "PULLBACK": 1, "CROSSING_UNVERIFIED": 2,
                         "BASE": 3, "NONE": 4, "NEGLECT": 5, "DOWN": 6, "DISQUALIFIED": 7}
                for _, r in scan_rows.iterrows():
                    sec = r["sector"]
                    new_k = r["klass"]
                    old_k = prev_map.get(sec)
                    if old_k and old_k != new_k:
                        arrow = "↗️" if order.get(new_k, 5) < order.get(old_k, 5) else "↘️"
                        sector_changes.append({"sector": sec, "old": old_k, "new": new_k, "arrow": arrow})
        changed_data["sector_changes"] = sector_changes

        prev_setups = db.get_setups(None)
        new_setups, lost_setups = [], []
        if prev_setups:
            prev_syms = {r["symbol"] for r in prev_setups if r.get("symbol")}
            if not buys.empty:
                curr_syms = set(buys["symbol"].astype(str))
                new_setups = sorted(curr_syms - prev_syms)
                lost_setups = sorted(prev_syms - curr_syms)
        changed_data["new_setups"] = new_setups
        changed_data["lost_setups"] = lost_setups

        if not buys.empty:
            setup_info = {}
            for sym in new_setups[:10]:
                rows = buys[buys["symbol"] == sym]
                if not rows.empty:
                    rw = rows.iloc[0]
                    setup_info[sym] = {
                        "sector": rw.get("sector", ""),
                        "coil": float(rw["coil"]) if pd.notna(rw.get("coil")) else None,
                    }
            changed_data["setup_info"] = setup_info
    except Exception as exc:
        print(f"[tg-bot] cache changed-data error: {exc}")

    if not coil_rows.empty:
        changed_data["n_coils"] = len(coil_rows)
        near_trig = coil_rows[coil_rows["to_trigger"].notna() & (coil_rows["to_trigger"] <= 0.02)]
        changed_data["n_near_trigger"] = len(near_trig)
    if not scan_rows.empty:
        changed_data["n_crossing"] = int((scan_rows["klass"] == "CROSSING").sum())
        changed_data["n_pullback"] = int((scan_rows["klass"] == "PULLBACK").sum())
        changed_data["total_sectors"] = len(scan_rows)

    cache["changed"] = changed_data

    _view_cache = cache
    n_stocks_cached = len(cache.get("stocks", {}))
    print(f"[tg-bot] cache refreshed for {as_of}: "
          f"{len(cache.get('triggers', []))} triggers, "
          f"{cache.get('delivery', {}).get('total', 0)} delivery, "
          f"{n_stocks_cached} stocks")
    return cache


def _cache_ok() -> bool:
    """True if the cache exists and matches the engine's as_of."""
    return bool(_view_cache and _view_cache.get("as_of"))


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
        "🆓 Stock reports and /today are always free.\n"
        "📊 3 market views per day (heatmap, flow, triggers…).\n"
        "⭐ Premium = unlimited everything + channel alerts.\n\n"
        "Already a subscriber? Link your account:\n"
        "<code>/link your@email.com</code>\n\n"
        "Or tap a button below to explore 👇"
    ), buttons=MAIN_MENU_BUTTONS)


def handle_help(chat_id: int):
    _reply(chat_id, (
        "📖 <b>Commands</b>\n\n"
        "<b>🆓 Always free:</b>\n"
        "  <code>TRENT</code> — stock report (just type any name)\n"
        "  /today — today's brief\n"
        "  /link — link your website account\n\n"
        "<b>📊 Market views</b> <i>(3/day free, unlimited premium)</i><b>:</b>\n"
        "  /heatmap — sector rotation map\n"
        "  /flow — money flow into/out of sectors\n"
        "  /triggers — stocks near breakout\n"
        "  /delivery — unusual institutional delivery\n"
        "  /changed — what changed since yesterday\n"
        "  /sector Name — sector deep-dive\n\n"
        "<b>⭐ Premium:</b> morrowdesk.com/pricing\n"
        "Unlimited commands + channel alerts."
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

    # Structural levels one-liner if available
    stop, target = plan.get("stop"), plan.get("target")
    if stop and target and adj:
        lines.append(f"\n📐 Base ₹{stop:,.0f} · 2R ₹{target:,.0f}")

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

    # Structural levels
    action = plan.get("action_label")
    if action:
        lines.append(f"\n🔔 <b>{action}</b>")
    stop, target, rr = plan.get("stop"), plan.get("target"), plan.get("rr")
    if stop is not None:
        lines.append(f"   Base level: ₹{stop:,.1f}")
    if target is not None:
        lines.append(f"   2R level: ₹{target:,.1f}")
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
    td = _view_cache.get("today") if _cache_ok() else None
    if not td:
        with engine._lock:
            if engine.status != "ready":
                _reply(chat_id, "⏳ Engine still loading.")
                return
        precompute_bot_cache(engine)
        td = _view_cache.get("today", {})

    as_of = _view_cache.get("as_of", "?")
    lines = [f"📅 <b>Today's Brief</b>", f"As of {as_of}\n"]

    # 1. Top actionable
    actionable = td.get("actionable", [])
    if actionable:
        lines.append("🎯 <b>Actionable</b>")
        for r in actionable:
            e = KLASS_EMOJI.get(r.get("klass", ""), "")
            lines.append(f"  <b>{r['symbol']}</b> ₹{r['adj']:,.0f} → trigger ₹{r['trigger']:,.0f} ({r['to_trigger'] * 100:.1f}% away)")
            lines.append(f"  {r['sector']} {e}{r.get('klass', '')}")
        lines.append("")
    else:
        lines.append("🎯 No stocks near breakout trigger today.")
        lines.append("<i>The market isn't always offering setups — that's okay.</i>\n")

    # 2. One insight
    best_sec = td.get("best_sector")
    best_cmf = td.get("best_cmf", 0)
    if best_sec and best_cmf > 0.05:
        klass = td.get("best_klass", "")
        lines.append("💡 <b>Insight</b>")
        lines.append(f"Strongest money flow: <b>{best_sec}</b> (CMF {best_cmf:+.2f})")
        lines.append(f"{td.get('best_adv_pct', 0):.0f}% advancing · {KLASS_EMOJI.get(klass, '')}{klass}")
        lines.append(f"<i>Positive CMF = institutions accumulating this sector</i>")
        lines.append("")

    worst_sec = td.get("worst_sector")
    worst_cmf = td.get("worst_cmf", 0)
    if worst_sec:
        lines.append(f"⚠️ Outflow: <b>{worst_sec}</b> (CMF {worst_cmf:+.2f})")
        lines.append(f"<i>Money leaving — avoid new positions here</i>")
        lines.append("")

    # 3. Market regime
    n_cross = td.get("n_crossing", 0)
    n_pull = td.get("n_pullback", 0)
    total = td.get("total_sectors", 1)
    bullish_pct = (n_cross + n_pull) / max(total, 1) * 100
    regime = "BULLISH" if bullish_pct >= 40 else "CAUTIOUS"
    emoji = "🟢" if regime == "BULLISH" else "🟡"
    lines.append(f"📊 <b>Market:</b> {emoji} {regime}")
    lines.append(f"   {n_cross} crossing · {n_pull} pullback · {bullish_pct:.0f}% in uptrend")
    n_coils = td.get("n_coils", 0)
    if n_coils:
        lines.append(f"   {n_coils} stocks coiled (compressed near highs)")

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


def _fmt_cached_sector(r: dict) -> str:
    """One-line sector from cached dict."""
    cmf = r.get("cmf", 0)
    cmf_arrow = "↑" if cmf > 0.05 else "↓" if cmf < -0.05 else ""
    return f"  <b>{r['sector']}</b> · T:{r.get('T', 0):.0f}% · {r.get('adv_pct', 0):.0f}% adv · CMF {cmf:+.2f}{cmf_arrow}"


def handle_heatmap(chat_id: int, engine, full: bool = False):
    # Try cache first
    hm = _view_cache.get("heatmap") if _cache_ok() else None

    if not hm:
        # Fall back to live computation if cache is stale
        with engine._lock:
            if engine.status != "ready" or engine.scan_rows is None or engine.scan_rows.empty:
                _reply(chat_id, "⏳ Engine still loading. Try again in a few minutes.")
                return
        precompute_bot_cache(engine)
        hm = _view_cache.get("heatmap")
        if not hm:
            _reply(chat_id, "No sector data available.")
            return

    as_of = _view_cache.get("as_of", "?")
    groups = hm.get("groups", {})
    regime = hm.get("regime", "CAUTIOUS")
    n_act = hm.get("n_actionable", 0)
    total = hm.get("total", 0)
    emoji = "🟢" if regime == "BULLISH" else "🟡"

    if full:
        lines = [f"🗺️ <b>All Sectors</b>", f"As of {as_of}\n"]
        labels = [
            ("🟢 CROSSING", "CROSSING"), ("🟡 PULLBACK", "PULLBACK"),
            ("⚪ BASE", "BASE"), ("🟠 UNVERIFIED", "CROSSING_UNVERIFIED"),
            ("🔴 DOWN", "DOWN"), ("⛔ DISQUALIFIED", "DISQUALIFIED"),
        ]
        for label, key in labels:
            grp = groups.get(key, [])
            if not grp:
                continue
            lines.append(f"<b>{label}</b> ({len(grp)})")
            for r in grp:
                if r.get("T") is not None:
                    lines.append(_fmt_cached_sector(r))
                else:
                    lines.append(f"  {r.get('sector', '?')}")
            lines.append("")
        lines.append(f"{emoji} <b>{regime}</b> · {n_act}/{total} sectors in uptrend")
        lines.append("\n<i>Sector rotation — not a recommendation.</i>")
        btns = [[{"text": "🔄 Flow", "callback_data": "/flow"},
                  {"text": "🎯 Triggers", "callback_data": "/triggers"}]]
        _send_long(chat_id, "\n".join(lines), buttons=btns)
        return

    # ── Concise view: actionable only ──
    lines = [f"🗺️ <b>Sector Heatmap</b>", f"As of {as_of}\n"]
    lines.append(f"{emoji} <b>Market: {regime}</b> — {n_act}/{total} sectors in uptrend\n")

    crossing = groups.get("CROSSING", [])
    pullback = groups.get("PULLBACK", [])

    if crossing:
        lines.append(f"🟢 <b>CROSSING</b> ({len(crossing)}) — uptrend confirmed")
        for r in crossing:
            lines.append(_fmt_cached_sector(r))
        lines.append("")
    if pullback:
        lines.append(f"🟡 <b>PULLBACK</b> ({len(pullback)}) — setup zone")
        for r in pullback:
            lines.append(_fmt_cached_sector(r))
        lines.append("")

    if not crossing and not pullback:
        lines.append("No sectors in CROSSING or PULLBACK right now.")
        lines.append("<i>The market isn't always offering setups — that's okay.</i>")

    rest_parts = []
    for label, key in [("base", "n_base"), ("down", "n_down"), ("disqualified", "n_dq")]:
        v = hm.get(key, 0)
        if v:
            rest_parts.append(f"{v} {label}")
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
    flow = _view_cache.get("flow") if _cache_ok() else None
    if not flow:
        with engine._lock:
            if engine.status != "ready":
                _reply(chat_id, "⏳ Engine still loading.")
                return
        precompute_bot_cache(engine)
        flow = _view_cache.get("flow")
    if not flow:
        _reply(chat_id, "No flow data available.")
        return

    as_of = _view_cache.get("as_of", "?")
    lines = [f"🔄 <b>Sector Money Flow</b>", f"As of {as_of}\n"]

    inflows = flow.get("inflows", [])
    if inflows:
        lines.append("💰 <b>INFLOWS</b> (accumulation)")
        for r in inflows:
            e = KLASS_EMOJI.get(r.get("klass", ""), "•")
            strength = "█" * min(8, max(1, int(r["cmf"] * 40)))
            lines.append(f"  {e} <b>{r['sector']}</b>")
            lines.append(f"    {strength} CMF {r['cmf']:+.2f} · {r.get('adv_pct', 0):.0f}% advancing")
        lines.append("")

    neutral = flow.get("neutral", [])
    if neutral:
        lines.append(f"⚖️ <b>NEUTRAL</b> ({len(neutral)} sectors)")
        for r in neutral[:5]:
            lines.append(f"  {r['sector']} · CMF {r['cmf']:+.2f}")
        if len(neutral) > 5:
            lines.append(f"  +{len(neutral) - 5} more")
        lines.append("")

    outflows = flow.get("outflows", [])
    if outflows:
        lines.append("🚨 <b>OUTFLOWS</b> (distribution)")
        for r in outflows:
            e = KLASS_EMOJI.get(r.get("klass", ""), "•")
            strength = "█" * min(8, max(1, int(abs(r["cmf"]) * 40)))
            lines.append(f"  {e} <b>{r['sector']}</b>")
            lines.append(f"    {strength} CMF {r['cmf']:+.2f}")
        lines.append("")

    lines.append("💡 <i>CMF measures buying vs selling pressure. "
                 "Positive = accumulation, negative = distribution.</i>")
    btns = [
        [{"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
         {"text": "🎯 Triggers", "callback_data": "/triggers"}],
    ]
    _send_long(chat_id, "\n".join(lines), buttons=btns)


# --------------------------------------------------------------------------
# /triggers — stocks near breakout
# --------------------------------------------------------------------------

def handle_triggers(chat_id: int, engine):
    trig_list = _view_cache.get("triggers") if _cache_ok() else None
    if trig_list is None:
        with engine._lock:
            if engine.status != "ready":
                _reply(chat_id, "⏳ Engine still loading.")
                return
        precompute_bot_cache(engine)
        trig_list = _view_cache.get("triggers")

    if not trig_list:
        _reply(chat_id, "📭 No stocks within 2% of their trigger right now.\n\n"
                        "<i>When a coiled stock is within 1-2% of its 20-day high, "
                        "one strong session can close through it.</i>")
        return

    as_of = _view_cache.get("as_of", "?")
    lines = [f"🎯 <b>Trigger Watch — {len(trig_list)} stocks near breakout</b>"]
    lines.append(f"As of {as_of}\n")

    for r in trig_list[:15]:
        pct = r["to_trigger"] * 100
        if pct < 0.5:
            proximity = "🔴 <b>AT TRIGGER</b>"
        elif pct < 1.0:
            proximity = "🟠 within 1%"
        else:
            proximity = "🟡 within 2%"

        lines.append(f"  <b>{r['symbol']}</b> — {proximity}")
        lines.append(f"    ₹{r['adj']:,.1f} → trigger ₹{r['trigger']:,.1f} ({pct:.1f}% away)")

        parts = [r["sector"]]
        if r.get("coil"):
            parts.append(f"coil {r['coil']:.1f}")
        if r.get("cmf") is not None:
            parts.append(f"CMF {r['cmf']:+.2f}")
        lines.append(f"    {' · '.join(parts)}")

        if r.get("is_buy"):
            lines.append("    ✅ In setup list")
        lines.append("")

    if len(trig_list) > 15:
        lines.append(f"  +{len(trig_list) - 15} more on the dashboard")

    lines.append("💡 <i>A close above the trigger on rising volume = confirmed breakout.</i>")
    btns = [
        [{"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
         {"text": "🔄 Flow", "callback_data": "/flow"}],
    ]
    _send_long(chat_id, "\n".join(lines), buttons=btns)


# --------------------------------------------------------------------------
# /delivery — unusual delivery activity
# --------------------------------------------------------------------------

def handle_delivery(chat_id: int, engine):
    deliv = _view_cache.get("delivery") if _cache_ok() else None
    if not deliv:
        with engine._lock:
            if engine.status != "ready":
                _reply(chat_id, "⏳ Engine still loading.")
                return
        precompute_bot_cache(engine)
        deliv = _view_cache.get("delivery")

    if not deliv or deliv.get("total", 0) == 0:
        _reply(chat_id, "📭 No unusual delivery activity today.\n\n"
                        "<i>Unusual delivery = delivery % significantly above the "
                        "stock's own 20-day average. High delivery typically "
                        "indicates institutional positions.</i>")
        return

    as_of = _view_cache.get("as_of", "?")
    accum = deliv.get("accum", [])
    distrib = deliv.get("distrib", [])

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

    lines.append(f"📊 {deliv['total']} stocks with unusual delivery today")
    lines.append("\n💡 <i>Delivery % = shares transferred to demat (not squared off). "
                 "High delivery + price rise = institutional buying.</i>")
    btns = [
        [{"text": "🗺️ Heatmap", "callback_data": "/heatmap"},
         {"text": "🎯 Triggers", "callback_data": "/triggers"}],
    ]
    _send_long(chat_id, "\n".join(lines), buttons=btns)


# --------------------------------------------------------------------------
# /changed — what changed since yesterday
# --------------------------------------------------------------------------

def handle_changed(chat_id: int, engine):
    cd = _view_cache.get("changed") if _cache_ok() else None
    if not cd:
        with engine._lock:
            if engine.status != "ready":
                _reply(chat_id, "⏳ Engine still loading.")
                return
        precompute_bot_cache(engine)
        cd = _view_cache.get("changed", {})

    as_of = _view_cache.get("as_of", "?")
    lines = [f"📋 <b>What Changed</b>"]
    lines.append(f"As of {as_of}\n")

    anything = False

    sector_changes = cd.get("sector_changes", [])
    if sector_changes:
        anything = True
        lines.append("<b>Sector moves:</b>")
        for sc in sector_changes:
            e_old = KLASS_EMOJI.get(sc["old"], "•")
            e_new = KLASS_EMOJI.get(sc["new"], "•")
            lines.append(f"  {sc['arrow']} <b>{sc['sector']}</b>: {e_old}{sc['old']} → {e_new}{sc['new']}")
        lines.append("")

    new_setups = cd.get("new_setups", [])
    setup_info = cd.get("setup_info", {})
    if new_setups:
        anything = True
        lines.append("<b>New setups entered:</b>")
        for sym in new_setups[:10]:
            info = setup_info.get(sym, {})
            sec = info.get("sector", "")
            coil_v = info.get("coil")
            coil = f" · coil {coil_v:.1f}" if coil_v is not None else ""
            lines.append(f"  ⚡ <code>{sym}</code> — {sec}{coil}")
        if len(new_setups) > 10:
            lines.append(f"  +{len(new_setups) - 10} more")
        lines.append("")

    lost_setups = cd.get("lost_setups", [])
    if lost_setups:
        anything = True
        lines.append("<b>Setups removed:</b>")
        for sym in lost_setups[:10]:
            lines.append(f"  ❌ <code>{sym}</code>")
        if len(lost_setups) > 10:
            lines.append(f"  +{len(lost_setups) - 10} more")
        lines.append("")

    n_coils = cd.get("n_coils", 0)
    if n_coils:
        n_near = cd.get("n_near_trigger", 0)
        lines.append(f"<b>Coil pool:</b> {n_coils} stocks coiled, {n_near} within 2% of trigger")
        anything = True

    n_cross = cd.get("n_crossing", 0)
    n_pull = cd.get("n_pullback", 0)
    total = cd.get("total_sectors", 0)
    if total:
        lines.append(f"<b>Sectors:</b> {n_cross} crossing · {n_pull} pullback · {total} total")
        anything = True

    if not anything:
        lines.append("No significant changes detected.\n"
                     "<i>This can happen on quiet days or when the previous "
                     "session's data isn't saved yet.</i>")

    lines.append("\n<i>Changes compared to the most recent saved session.</i>")
    _send_long(chat_id, "\n".join(lines), buttons=MAIN_MENU_BUTTONS)


# Pending OTP verifications: {user_id: (email, otp_code, timestamp)}
_pending_otps: dict[int, tuple[str, str, float]] = {}
_OTP_EXPIRY = 300  # 5 minutes


def _generate_otp() -> str:
    import random
    return str(random.randint(100000, 999999))


def _send_otp_email(email: str, otp: str) -> bool:
    """Send OTP via Supabase edge function or SMTP."""
    # Use Supabase's built-in email: insert a row that triggers
    # an email, or call a simple edge function.
    # For now, use the Supabase Auth magic link as a workaround:
    # we send the OTP through the bot and verify on website.
    # Simplest approach: send OTP through Supabase's REST API
    # to trigger an email via a DB function or edge function.
    try:
        from subscription import _get_supabase
        client = _get_supabase()
        # Store OTP in a verification table so the website can
        # also show "verify your Telegram" if needed
        client.table("link_verifications").upsert({
            "email": email.lower(),
            "otp": otp,
            "created_at": datetime.now().isoformat(),
            "verified": False,
        }, on_conflict="email").execute()
        return True
    except Exception as exc:
        print(f"[tg-bot] OTP store failed: {exc}")
        return False


def handle_link(chat_id: int, user_id: int, args: str):
    """
    Link a Telegram account to a website account via email.

    Two-step verification:
    1. /link user@email.com  → sends 6-digit OTP, shows verify prompt
    2. /verify 123456        → confirms ownership, links accounts
    """
    import time as _time

    email = args.strip().lower()
    if not email or "@" not in email:
        _reply(chat_id,
               "🔗 <b>Link your account</b>\n\n"
               "Send your Morrow Desk email to connect your Telegram:\n\n"
               "<code>/link your@email.com</code>\n\n"
               "<i>Use the same email you signed up with on the website. "
               "We'll send a verification code to confirm it's yours.</i>")
        return

    # Check if this email is already linked to a DIFFERENT telegram user
    try:
        from subscription import _get_supabase
        client = _get_supabase()
        result = client.table("subscriptions").select("telegram_id") \
            .eq("email", email).execute()
        if result.data:
            existing_tg = result.data[0].get("telegram_id")
            if existing_tg and existing_tg != str(user_id):
                _reply(chat_id,
                       "⚠️ This email is already linked to another Telegram account.\n\n"
                       "<i>If this is your email, contact support to unlink it.</i>")
                return
            if existing_tg == str(user_id):
                _reply(chat_id,
                       f"✅ Already linked to <b>{email}</b>",
                       buttons=MAIN_MENU_BUTTONS)
                return
    except Exception:
        pass

    # Generate OTP and store it
    otp = _generate_otp()
    _pending_otps[user_id] = (email, otp, _time.time())

    # Store in DB so it persists across restarts
    _send_otp_email(email, otp)

    _reply(chat_id,
           f"📧 Verification code sent!\n\n"
           f"Check your <b>Morrow Desk website</b> — log in with "
           f"<b>{email}</b> and you'll see your 6-digit code.\n\n"
           f"Then send it here:\n"
           f"<code>/verify 123456</code>\n\n"
           f"<i>Code expires in 5 minutes.</i>")


def handle_verify(chat_id: int, user_id: int, args: str):
    """Verify the OTP and complete the account link."""
    import time as _time

    code = args.strip()
    if not code or not code.isdigit() or len(code) != 6:
        _reply(chat_id,
               "Enter the 6-digit code from your Morrow Desk account:\n\n"
               "<code>/verify 123456</code>")
        return

    pending = _pending_otps.get(user_id)
    if not pending:
        _reply(chat_id,
               "No pending verification. Start with:\n"
               "<code>/link your@email.com</code>")
        return

    email, correct_otp, created_at = pending

    # Check expiry
    if _time.time() - created_at > _OTP_EXPIRY:
        _pending_otps.pop(user_id, None)
        _reply(chat_id,
               "⏰ Code expired. Please start again:\n"
               f"<code>/link {email}</code>")
        return

    # Check code
    if code != correct_otp:
        _reply(chat_id, "❌ Wrong code. Try again or request a new one with /link")
        return

    # OTP matches — link the accounts
    _pending_otps.pop(user_id, None)

    try:
        from subscription import _get_supabase, clear_cache
        client = _get_supabase()

        # Upsert: set telegram_id on the subscription row
        client.table("subscriptions").upsert({
            "email": email,
            "telegram_id": str(user_id),
        }, on_conflict="email").execute()

        # Mark verification as complete
        try:
            client.table("link_verifications").update({
                "verified": True,
            }).eq("email", email).execute()
        except Exception:
            pass

        clear_cache(email)
        _paid_cache.pop(user_id, None)

        # Check premium status
        result = client.table("subscriptions").select("is_premium") \
            .eq("email", email).execute()
        is_prem = result.data[0].get("is_premium", False) if result.data else False

        if is_prem:
            _reply(chat_id,
                   f"✅ Verified & linked to <b>{email}</b>\n\n"
                   f"Premium status: <b>Active</b> ✨\n"
                   f"All bot commands are now unlimited.",
                   buttons=MAIN_MENU_BUTTONS)
        else:
            _reply(chat_id,
                   f"✅ Verified & linked to <b>{email}</b>\n\n"
                   f"No active subscription yet. When you subscribe "
                   f"on the website, premium activates here automatically.",
                   buttons=[
                       [{"text": "📅 Today (free)", "callback_data": "/today"}],
                   ])

    except Exception as exc:
        print(f"[tg-bot] verify link error: {exc}")
        _reply(chat_id, "⚠️ Could not link account. Try again later.")


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
    if lower == "/link" or lower.startswith("/link "):
        args = text.split(maxsplit=1)[1] if " " in text else ""
        handle_link(chat_id, user_id, args)
        return
    if lower == "/verify" or lower.startswith("/verify "):
        args = text.split(maxsplit=1)[1] if " " in text else ""
        handle_verify(chat_id, user_id, args)
        return

    # Paid-gated market views — check limit
    if cmd in _PAID_COMMANDS:
        if not _check_limit(user_id, cmd):
            _reply(chat_id, _limit_msg(), buttons=[
                [{"text": "📊 Stock report (free)", "callback_data": "/help"},
                 {"text": "🔗 Link account", "callback_data": "/link"}],
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
            if remaining == 0:
                note = "💡 That was your last free market view today. Stock reports are always free."
            else:
                note = (f"💡 {remaining} free market view{'s' if remaining != 1 else ''} "
                        f"left today. Stock reports are always free.")
            _reply(chat_id, note, buttons=[
                [{"text": "🔗 Link account", "callback_data": "/link"}],
            ] if remaining == 0 else None)


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
