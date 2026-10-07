"""
Telegram alert bot for the circuit scanner and broader market scanners.

Environment variables (set in .env or EC2 environment):
    TELEGRAM_BOT_TOKEN    — from @BotFather
    TELEGRAM_FREE_CHAT    — public channel ID or @username (EOD alerts)
    TELEGRAM_PAID_CHAT    — private channel chat ID (real-time alerts)

Usage:
    # Post intraday circuit alert to PAID channel
    python telegram_bot.py circuit

    # Post EOD summary to BOTH channels
    python telegram_bot.py eod

    # Post morning scorecard to BOTH channels
    python telegram_bot.py morning

    # Post scanner alert to PAID channel
    python telegram_bot.py scanners

Called automatically by the cron endpoints after each scan.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
FREE_CHAT = os.environ.get("TELEGRAM_FREE_CHAT", "")   # @channel or chat_id
PAID_CHAT = os.environ.get("TELEGRAM_PAID_CHAT", "")   # chat_id (negative)

API = f"https://api.telegram.org/bot{BOT_TOKEN}"


def _send(chat_id: str, text: str, silent: bool = False) -> bool:
    """Send a message to a Telegram chat/channel."""
    if not BOT_TOKEN or not chat_id:
        print(f"[telegram] skipped (no token or chat_id)")
        return False
    try:
        r = requests.post(f"{API}/sendMessage", json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_notification": silent,
        }, timeout=15)
        if not r.ok:
            print(f"[telegram] error {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as exc:
        print(f"[telegram] send failed: {exc}")
        return False


def send_paid(text: str, silent: bool = False) -> bool:
    """Send to the paid (real-time) channel."""
    return _send(PAID_CHAT, text, silent)


def send_free(text: str, silent: bool = False) -> bool:
    """Send to the free (EOD) channel."""
    return _send(FREE_CHAT, text, silent)


def send_both(text: str, silent: bool = False) -> tuple[bool, bool]:
    """Send to both channels."""
    return send_free(text, silent), send_paid(text, silent)


# --------------------------------------------------------------------------
# Message formatters
# --------------------------------------------------------------------------

def _qty_fmt(v) -> str:
    if v is None:
        return "—"
    n = int(v)
    if n >= 1e7:
        return f"{n / 1e7:.1f}cr"
    if n >= 1e5:
        return f"{n / 1e5:.1f}L"
    return f"{n:,}"


def format_circuit_alert(data: dict) -> str | None:
    """
    Format an intraday circuit scan for Telegram.
    Returns None if nothing worth alerting.
    """
    summary = data.get("summary", {})
    total = sum(summary.values())
    if total == 0:
        return None

    scan_time = data.get("latest_scan", "?")
    as_of = data.get("as_of", "?")
    lines = [f"🔔 <b>Circuit Scan — {scan_time} IST</b>"]
    lines.append(f"📅 {as_of}\n")

    latest = data.get("latest", [])

    # Group by status
    at_circuit = [r for r in latest if r.get("status") == "at_circuit"]
    locked = [r for r in latest if r.get("status") == "locked"]
    approaching = [r for r in latest if r.get("status") == "approaching"]
    heating = [r for r in latest if r.get("status") == "heating"]

    if at_circuit:
        lines.append("🟢 <b>AT CIRCUIT</b> (sellers present)")
        for r in at_circuit[:10]:
            band = f"{int(r.get('band', 0) * 100)}%" if r.get("band") else "?"
            turn = f"₹{r.get('med_turn20', 0) / 100:.0f}cr" if r.get("med_turn20") else ""
            lines.append(f"  <b>{r['symbol']}</b> · {band} band · ₹{r.get('ltp', '?')} · {turn}")
        lines.append("")

    if locked:
        lines.append("🔒 <b>LOCKED</b> (queue only)")
        for r in locked[:10]:
            band = f"{int(r.get('band', 0) * 100)}%" if r.get("band") else "?"
            buy_q = _qty_fmt(r.get("total_buy_qty"))
            lines.append(f"  <b>{r['symbol']}</b> · {band} band · ₹{r.get('ltp', '?')} · {buy_q} buyers")
        lines.append("")

    if approaching:
        lines.append("🟡 <b>APPROACHING</b> (within 1%)")
        for r in approaching[:8]:
            band = f"{int(r.get('band', 0) * 100)}%" if r.get("band") else "?"
            dist = r.get("distance_to_circuit")
            dist_str = f"{dist * 100:.1f}% away" if dist is not None else ""
            fill = "✓ sellers" if r.get("fillable") else "no sellers"
            lines.append(f"  <b>{r['symbol']}</b> · {band} band · {dist_str} · {fill}")
        lines.append("")

    if heating:
        lines.append(f"🔥 <b>HEATING</b> ({len(heating)} stocks)")
        for r in heating[:5]:
            lines.append(f"  {r['symbol']} +{r.get('pchange', 0):.1f}%")
        if len(heating) > 5:
            lines.append(f"  ... and {len(heating) - 5} more")
        lines.append("")

    # Summary line
    parts = []
    if summary.get("at_circuit", 0):
        parts.append(f"{summary['at_circuit']} at circuit")
    if summary.get("locked", 0):
        parts.append(f"{summary['locked']} locked")
    if summary.get("approaching", 0):
        parts.append(f"{summary['approaching']} approaching")
    if summary.get("heating", 0):
        parts.append(f"{summary['heating']} heating")
    lines.append(f"📊 {' · '.join(parts)}")

    scans = data.get("scans", [])
    lines.append(f"🔄 Scan {len(scans)} of the day")

    return "\n".join(lines)


def format_eod_summary(carry_data: dict) -> str:
    """Format the end-of-day carry summary for both channels."""
    lines = [f"📋 <b>Circuit Carry — EOD Summary</b>"]
    as_of = carry_data.get("as_of", "?")
    lines.append(f"📅 {as_of}\n")

    latest = carry_data.get("latest", [])
    if not latest:
        lines.append("No stocks at circuit today.")
        return "\n".join(lines)

    source = carry_data.get("source", "eod")
    lines.append(f"<b>{len(latest)} stocks</b> closed at circuit today"
                 f" ({source} snapshot)\n")

    for r in latest[:20]:
        band = f"{int(float(r.get('band', 0)) * 100)}%" if r.get("band") else "?"
        fill = "✅" if r.get("fillable") else "🔒"
        lines.append(f"{fill} <b>{r.get('symbol', '?')}</b> · {band} · "
                     f"₹{r.get('ltp', '?')}")

    if len(latest) > 20:
        lines.append(f"... and {len(latest) - 20} more")

    # Pending count
    pending = carry_data.get("pending", 0)
    if pending:
        lines.append(f"\n⏳ {pending} stocks awaiting next-session results")

    lines.append("\n<i>Observational data, not a recommendation.</i>")
    return "\n".join(lines)


def format_morning_scorecard(carry_data: dict) -> str | None:
    """
    How yesterday's circuit stocks opened. Sent at 9:15 AM.
    Uses the daily history from the carry payload.
    """
    daily = carry_data.get("daily", [])
    if not daily:
        return None

    # Most recent scored day
    scored = [d for d in daily if d.get("mean_btst") is not None]
    if not scored:
        return None

    d = scored[0]
    lines = [f"☀️ <b>Morning Scorecard</b>"]
    lines.append(f"📅 Results for circuit list of {d['as_of']}\n")
    lines.append(f"📊 <b>{d.get('hits', 0)}/{d.get('n', 0)}</b> reached +4% next session")
    lines.append(f"📈 Mean move: <b>{d.get('mean_btst', 0) * 100:+.1f}%</b>")
    lines.append(f"🏆 Best: {d.get('best', 0) * 100:+.1f}%  "
                 f"📉 Worst: {d.get('worst', 0) * 100:+.1f}%")

    # Overall stats
    summary = carry_data.get("summary", [])
    all_bucket = next((s for s in summary if s.get("bucket") == "all"), None)
    if all_bucket:
        lines.append(f"\n📋 <b>Overall track record</b> ({all_bucket.get('sessions', '?')} sessions)")
        lines.append(f"  Reached +4%: {all_bucket.get('hit4', 0) * 100:.0f}%")
        lines.append(f"  Mean move: {all_bucket.get('mean_btst', 0) * 100:+.1f}%")

    lines.append("\n<i>Historical observations, not a performance claim.</i>")
    return "\n".join(lines)


def format_scanners(data: dict) -> str | None:
    """Format the broader scanner results."""
    vol = data.get("unusual_volume", [])
    movers = data.get("big_movers", [])
    sectors = data.get("sector_pulse", [])
    scan_time = data.get("scan_time", "?")

    if not vol and not movers:
        return None

    lines = [f"📡 <b>Market Pulse — {scan_time} IST</b>\n"]

    if movers:
        lines.append(f"🚀 <b>Big Movers</b> ({len(movers)} stocks up 5%+)")
        for r in movers[:8]:
            lines.append(f"  <b>{r['symbol']}</b> +{r['pchange']:.1f}% · "
                         f"₹{r['ltp']:.0f} · {r['sector']}")
        if len(movers) > 8:
            lines.append(f"  ... +{len(movers) - 8} more")
        lines.append("")

    if vol:
        lines.append(f"📊 <b>Unusual Volume</b> ({len(vol)} stocks at 2x+ normal)")
        for r in vol[:8]:
            lines.append(f"  <b>{r['symbol']}</b> {r['vol_ratio']}x vol · "
                         f"+{r['pchange']:.1f}% · {r['sector']}")
        if len(vol) > 8:
            lines.append(f"  ... +{len(vol) - 8} more")
        lines.append("")

    if sectors:
        hot = [s for s in sectors if s["avg_change"] > 0.5][:5]
        if hot:
            lines.append("🌡️ <b>Hot Sectors</b>")
            for s in hot:
                lines.append(f"  {s['sector']} · {s['adv_pct']:.0f}% advancing · "
                             f"avg +{s['avg_change']:.1f}% · "
                             f"top: {s['best_stock']} +{s['best_change']:.1f}%")

    lines.append("\n<i>What's moving today — informational, not a recommendation.</i>")
    return "\n".join(lines)


def format_eod_scanners(data: dict) -> str | None:
    """Format EOD scanners (streaks + breakouts)."""
    streaks = data.get("momentum_streaks", [])
    breakouts = data.get("breakouts_52w", [])

    if not streaks and not breakouts:
        return None

    lines = [f"📋 <b>EOD Watchlists — {data.get('as_of', '?')}</b>\n"]

    if breakouts:
        lines.append(f"📈 <b>52-Week Breakouts</b> ({len(breakouts)} stocks)")
        for r in breakouts[:10]:
            lines.append(f"  <b>{r['symbol']}</b> ₹{r['close']:.0f} · "
                         f"{r['from_high']:+.1f}% from high · "
                         f"{r['vol_ratio']}x vol · {r['sector']}")
        if len(breakouts) > 10:
            lines.append(f"  ... +{len(breakouts) - 10} more")
        lines.append("")

    if streaks:
        lines.append(f"🔥 <b>Momentum Streaks</b> ({len(streaks)} stocks, 3+ up days)")
        for r in streaks[:10]:
            lines.append(f"  <b>{r['symbol']}</b> ₹{r['close']:.0f} · "
                         f"{r['streak']} days · +{r['cum_return']:.1f}% · {r['sector']}")
        if len(streaks) > 10:
            lines.append(f"  ... +{len(streaks) - 10} more")

    lines.append("\n<i>Informational watchlists, not recommendations.</i>")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python telegram_bot.py [circuit|eod|morning|scanners|eod-scanners|test]")
        return

    cmd = sys.argv[1]

    if cmd == "test":
        ok = _send(PAID_CHAT or FREE_CHAT,
                    "✅ <b>Bot connected!</b>\nAlerts will appear here.")
        print(f"Test message: {'sent' if ok else 'FAILED'}")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if cmd == "circuit":
        import carry
        data = carry.intraday_payload()
        msg = format_circuit_alert(data)
        if msg:
            send_paid(msg)
            print(f"[telegram] circuit alert sent ({len(msg)} chars)")
        else:
            print("[telegram] nothing to alert")

    elif cmd == "eod":
        import carry
        data = carry.payload()
        msg = format_eod_summary(data)
        send_both(msg)
        print(f"[telegram] EOD summary sent to both channels")

    elif cmd == "morning":
        import carry
        data = carry.payload()
        msg = format_morning_scorecard(data)
        if msg:
            send_both(msg)
            print(f"[telegram] morning scorecard sent")
        else:
            print("[telegram] no scored data for scorecard")

    elif cmd == "scanners":
        import scanners
        data = scanners.intraday_all()
        msg = format_scanners(data)
        if msg:
            send_paid(msg)
            print(f"[telegram] scanner alert sent ({len(msg)} chars)")

    elif cmd == "eod-scanners":
        import scanners
        data = scanners.eod_all()
        msg = format_eod_scanners(data)
        if msg:
            send_both(msg)
            print(f"[telegram] EOD scanners sent")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
