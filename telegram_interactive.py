"""
Interactive Telegram bot — users message a stock name, get a full report.

Commands:
    /start          — welcome message
    /help           — list commands
    /r SYMBOL       — stock report (alias: /report)
    /sector NAME    — sector overview
    SYMBOL          — plain text treated as stock lookup

Architecture:
    Telegram sends updates to POST /api/telegram/webhook.
    This module processes the update and replies inline.
    The bot talks to the same in-memory engine that powers the dashboard.
"""

from __future__ import annotations

import os
import re
import requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API = f"https://api.telegram.org/bot{BOT_TOKEN}"


def _reply(chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
    if not BOT_TOKEN:
        return False
    try:
        r = requests.post(f"{API}/sendMessage", json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "link_preview_options": {"is_disabled": True},
        }, timeout=15)
        return r.ok
    except Exception:
        return False


# --------------------------------------------------------------------------
# Command handlers
# --------------------------------------------------------------------------

def handle_start(chat_id: int):
    _reply(chat_id, (
        "👋 <b>Welcome to Morrow Desk</b>\n\n"
        "I analyse NSE stocks using sector structure, coil patterns, "
        "volume, delivery and momentum.\n\n"
        "<b>Try it:</b>\n"
        "  Send any stock name — <code>TRENT</code>, <code>RELIANCE</code>, <code>AXISCADES</code>\n\n"
        "<b>Commands:</b>\n"
        "  /r SYMBOL — full stock report\n"
        "  /sector NAME — sector overview\n"
        "  /help — this message\n\n"
        "<i>All analysis is observational, not a recommendation.</i>"
    ))


def handle_help(chat_id: int):
    _reply(chat_id, (
        "📖 <b>Commands</b>\n\n"
        "  <b>Stock report:</b>\n"
        "  Just type a symbol: <code>TRENT</code>\n"
        "  Or: /r TRENT\n\n"
        "  <b>Sector overview:</b>\n"
        "  /sector Retailing\n\n"
        "  <b>Other:</b>\n"
        "  /start — welcome\n"
        "  /help — this message\n\n"
        "💡 You can type company names too — <code>Tata Motors</code> "
        "works as well as <code>TATAMOTORS</code>."
    ))


def _fmt_report(data: dict) -> str:
    """Format an analyze.build() report for Telegram."""
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

    # Price
    adj = m.get("adj")
    if adj:
        lines.append(f"💰 Price: <b>₹{adj:,.1f}</b>")

    # Sector
    klass_emoji = {
        "CROSSING": "🟢", "PULLBACK": "🟡", "BASE": "⚪",
        "CROSSING_UNVERIFIED": "🟠", "DOWN": "🔴", "DISQUALIFIED": "⛔",
    }.get(klass, "•")
    lines.append(f"\n🏭 <b>Sector:</b> {sector}")
    lines.append(f"   {klass_emoji} {klass}")

    # Sector shape details
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

    # Phase
    phase_emoji = {
        "Broke out": "🚀", "Potential": "🎯", "Coiled": "⚡",
        "Near miss": "👀", "At the high": "📈", "Volume break": "💥",
        "Watching": "⏳",
    }.get(phase, "•")
    lines.append(f"\n{phase_emoji} <b>Status:</b> {phase}")

    # Technical
    lines.append(f"\n📈 <b>Technical:</b>")
    ema50 = m.get("ema50")
    ema200 = m.get("ema200")
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
        rsi_emoji = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "•"
        lines.append(f"   {rsi_emoji} RSI: {rsi:.0f}")

    pos_hi = m.get("pos_hi")
    if pos_hi is not None:
        lines.append(f"   📏 Position: {pos_hi * 100:.0f}% of 85-day range")

    # Volume & flow
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

    # Trade plan
    action = plan.get("action_label")
    if action:
        lines.append(f"\n🔔 <b>Structure:</b> {action}")
    stop = plan.get("stop")
    target = plan.get("target")
    rr = plan.get("rr")
    if stop is not None:
        lines.append(f"   Stop: ₹{stop:,.1f}")
    if target is not None:
        lines.append(f"   2R target: ₹{target:,.1f}")
    if rr is not None:
        lines.append(f"   R:R = 1:{rr:.1f}")

    # Setup / breakout context
    setup = data.get("setup")
    if setup and setup.get("why"):
        lines.append(f"\n💡 <b>Setup:</b> {setup['why'][:200]}")

    breakout = data.get("breakout")
    if breakout and breakout.get("why"):
        lines.append(f"\n💥 <b>Breakout:</b> {breakout['why'][:200]}")

    # Plain English summary
    why = data.get("why")
    if why:
        lines.append(f"\n📝 <b>Summary:</b>\n<i>{why[:400]}</i>")

    lines.append("\n<i>Observational analysis — not a recommendation.</i>")
    return "\n".join(lines)


def _fmt_sector(data: dict) -> str:
    """Format a sector report for Telegram."""
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
    note = data.get("note") or ""
    shape = data.get("shape") or {}
    buys = data.get("buys") or []
    members = data.get("constituents") or []

    klass_emoji = {
        "CROSSING": "🟢", "PULLBACK": "🟡", "BASE": "⚪",
        "CROSSING_UNVERIFIED": "🟠", "DOWN": "🔴", "DISQUALIFIED": "⛔",
    }.get(klass, "•")

    lines = [f"🏭 <b>{name}</b>"]
    lines.append(f"{klass_emoji} <b>{klass}</b>")
    if note:
        lines.append(f"<i>{note}</i>")

    t = shape.get("T")
    b = shape.get("B")
    cmf = shape.get("cmf")
    rs = shape.get("rs")
    lines.append("")
    if t is not None:
        lines.append(f"📈 Breadth: {t:.0f}%")
    if b is not None:
        lines.append(f"📊 Width: {b:.0f}%")
    if cmf is not None:
        lines.append(f"💧 CMF: {cmf:+.2f}")
    if rs is not None:
        lines.append(f"💪 RS: {rs:+.1f}")

    verdict = shape.get("verdict")
    if verdict:
        lines.append(f"\n📋 <b>Verdict:</b> {verdict}")

    if buys:
        lines.append(f"\n🎯 <b>Setups ({len(buys)}):</b>")
        for r in buys[:5]:
            sym = r.get("symbol", "?")
            coil = r.get("coil")
            coil_s = f" · coil {coil:.1f}" if coil else ""
            lines.append(f"  <code>{sym}</code>{coil_s}")
        if len(buys) > 5:
            lines.append(f"  +{len(buys) - 5} more")

    if members:
        lines.append(f"\n👥 <b>Top constituents:</b>")
        for r in members[:8]:
            sym = r.get("symbol", "?")
            ret = r.get("ret")
            ret_s = f" {ret * 100:+.1f}%" if ret is not None else ""
            lines.append(f"  {sym}{ret_s}")

    lines.append("\n<i>Sector analysis — not a recommendation.</i>")
    return "\n".join(lines)


def handle_report(chat_id: int, query: str, engine):
    """Look up a stock and reply with the full report."""
    if not query:
        _reply(chat_id, "Usage: /r SYMBOL\nExample: <code>/r TRENT</code>")
        return

    _reply(chat_id, f"🔍 Looking up <b>{query.upper()}</b>...")

    with engine._lock:
        if engine.status != "ready" or engine.stocks is None:
            _reply(chat_id, "⏳ The engine is still loading. Try again in a few minutes.")
            return

    try:
        data = engine.stock(query)
    except Exception as exc:
        _reply(chat_id, f"❌ Error: {str(exc)[:200]}")
        return

    msg = _fmt_report(data)
    # Telegram has a 4096 char limit per message
    if len(msg) > 4000:
        _reply(chat_id, msg[:4000] + "\n\n<i>... truncated</i>")
    else:
        _reply(chat_id, msg)


def handle_sector(chat_id: int, query: str, engine):
    """Look up a sector and reply with the overview."""
    if not query:
        _reply(chat_id, "Usage: /sector NAME\nExample: <code>/sector Retailing</code>")
        return

    _reply(chat_id, f"🔍 Looking up sector <b>{query}</b>...")

    with engine._lock:
        if engine.status != "ready":
            _reply(chat_id, "⏳ The engine is still loading. Try again in a few minutes.")
            return

    try:
        data = engine.sector(query)
    except Exception as exc:
        _reply(chat_id, f"❌ Error: {str(exc)[:200]}")
        return

    if data is None:
        _reply(chat_id, "⏳ Engine not ready.")
        return

    msg = _fmt_sector(data)
    if len(msg) > 4000:
        _reply(chat_id, msg[:4000] + "\n\n<i>... truncated</i>")
    else:
        _reply(chat_id, msg)


# --------------------------------------------------------------------------
# Webhook dispatcher
# --------------------------------------------------------------------------

def process_update(update: dict, engine) -> None:
    """
    Process one Telegram update. Called from the webhook endpoint.

    Handles:
      /start, /help — info
      /r SYMBOL, /report SYMBOL — stock report
      /sector NAME — sector overview
      plain text — treated as stock lookup
    """
    msg = update.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = msg.get("chat", {}).get("id")

    if not chat_id or not text:
        return

    # Ignore messages from channels/groups — bot is for DMs
    chat_type = msg.get("chat", {}).get("type", "private")
    if chat_type not in ("private",):
        return

    lower = text.lower()

    if lower == "/start":
        handle_start(chat_id)
    elif lower == "/help":
        handle_help(chat_id)
    elif lower.startswith("/r ") or lower.startswith("/report "):
        query = text.split(maxsplit=1)[1] if " " in text else ""
        handle_report(chat_id, query.strip(), engine)
    elif lower.startswith("/sector "):
        query = text.split(maxsplit=1)[1] if " " in text else ""
        handle_sector(chat_id, query.strip(), engine)
    elif text.startswith("/"):
        _reply(chat_id, "Unknown command. Try /help")
    else:
        # Plain text — treat as stock lookup
        handle_report(chat_id, text.strip(), engine)


def set_webhook(url: str) -> bool:
    """Tell Telegram to send updates to our webhook URL."""
    if not BOT_TOKEN:
        return False
    try:
        r = requests.post(f"{API}/setWebhook", json={
            "url": url,
            "allowed_updates": ["message"],
        }, timeout=15)
        print(f"[telegram] setWebhook: {r.status_code} {r.text[:200]}")
        return r.ok
    except Exception as exc:
        print(f"[telegram] setWebhook failed: {exc}")
        return False


def delete_webhook() -> bool:
    """Remove the webhook (switch back to getUpdates polling)."""
    if not BOT_TOKEN:
        return False
    try:
        r = requests.post(f"{API}/deleteWebhook", timeout=15)
        return r.ok
    except Exception:
        return False
