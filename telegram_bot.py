"""
Smart Telegram alerts for Morrow Desk.

Design principles:
  1. Alert on CHANGE, not on a timer — don't resend the same stocks.
  2. Context beats data — one sentence explaining what the number means.
  3. Fewer, better messages — 3-4/day paid, 2/day + 1/week free.
  4. Accountability — every morning shows exactly what happened.

Alert schedule:
  PAID channel (real-time):
    ⚡ Circuit flash   — only when a NEW stock reaches circuit or status changes
    📡 Midday pulse    — 12:30 IST, ONE consolidated scanner summary
  BOTH channels:
    ☀️ Morning brief   — 9:15, yesterday's scorecard with per-stock results
    📋 EOD wrap        — 15:45, final circuit list + day's standout moves
    📊 Weekly digest   — Saturday 10:00, week's results + cumulative track record

Environment variables:
    TELEGRAM_BOT_TOKEN    — from @BotFather
    TELEGRAM_FREE_CHAT    — public channel (@username or chat_id)
    TELEGRAM_PAID_CHAT    — private channel (chat_id, negative number)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
FREE_CHAT = os.environ.get("TELEGRAM_FREE_CHAT", "")
PAID_CHAT = os.environ.get("TELEGRAM_PAID_CHAT", "")

API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# In-memory state: what we already alerted this session.
# Reset when the date changes.
_alerted_date: str = ""
_alerted_symbols: dict[str, str] = {}  # symbol → last status sent


def _send(chat_id: str, text: str, silent: bool = False) -> bool:
    if not BOT_TOKEN or not chat_id:
        print("[telegram] skipped (no token or chat_id)")
        return False
    try:
        r = requests.post(f"{API}/sendMessage", json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_notification": silent,
            "link_preview_options": {"is_disabled": True},
        }, timeout=15)
        if not r.ok:
            print(f"[telegram] error {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as exc:
        print(f"[telegram] send failed: {exc}")
        return False


def send_paid(text: str, silent: bool = False) -> bool:
    return _send(PAID_CHAT, text, silent)

def send_free(text: str, silent: bool = False) -> bool:
    return _send(FREE_CHAT, text, silent)

def send_both(text: str, silent: bool = False) -> tuple[bool, bool]:
    return send_free(text, silent), send_paid(text, silent)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _qty(v) -> str:
    if v is None: return "—"
    n = int(v)
    if n >= 1e7: return f"{n / 1e7:.1f}cr"
    if n >= 1e5: return f"{n / 1e5:.1f}L"
    return f"{n:,}"

def _turn(med_turn20) -> str:
    if not med_turn20: return ""
    cr = float(med_turn20) / 100
    return f"₹{cr:.0f}cr" if cr >= 1 else f"₹{cr:.1f}cr"

def _band_context(band: float) -> str:
    """One-line context about how this band has historically performed."""
    b = int(round(float(band) * 100)) if band else 0
    # Raw pattern base rates (all closes on the band, fill not assumed). Shown
    # as context only — whether you could actually buy is the open question.
    if b == 5:
        return "5% band — most common; raw pattern reached +4% ~69% of the time (fill not guaranteed)"
    if b == 10:
        return "10% band — raw pattern reached +4% ~74% of the time (fill not guaranteed)"
    if b == 20:
        return "20% band — strongest; raw pattern reached +4% ~86% of the time (fill not guaranteed)"
    return ""

def _turnover_context(med_turn20) -> str:
    """Turnover tier insight from the volume study."""
    if not med_turn20: return ""
    cr = float(med_turn20) / 100
    if cr < 5:
        return "Lower turnover — historically the strongest continuation (83% hit rate)"
    if cr < 20:
        return "Mid-range turnover — 74% hit rate historically"
    return "Higher turnover — 69% hit rate historically, still positive"


# --------------------------------------------------------------------------
# Smart circuit flash — only on state changes
# --------------------------------------------------------------------------

def smart_circuit_alert(data: dict) -> str | None:
    """
    Compare current scan to what was already alerted.
    Only produce a message if there are NEW stocks or STATUS CHANGES.
    """
    global _alerted_date, _alerted_symbols

    as_of = data.get("as_of", "")
    if as_of != _alerted_date:
        _alerted_date = as_of
        _alerted_symbols = {}

    latest = data.get("latest", [])
    scan_time = data.get("latest_scan", "?")

    # Find what's new or changed
    new_stocks = []
    status_changes = []
    for r in latest:
        sym = r.get("symbol", "")
        status = r.get("status", "")
        if status == "heating":
            continue  # don't alert on heating — too early, too noisy

        prev_status = _alerted_symbols.get(sym)
        if prev_status is None:
            new_stocks.append(r)
        elif prev_status != status:
            status_changes.append((r, prev_status))

    if not new_stocks and not status_changes:
        return None  # nothing changed — stay silent

    lines = [f"⚡ <b>Circuit Flash — {scan_time} IST</b>\n"]

    if new_stocks:
        for r in new_stocks:
            sym = r["symbol"]
            status = r.get("status", "")
            band = float(r.get("band", 0))
            band_pct = f"{int(band * 100)}%" if band else "?"
            ltp = r.get("ltp", "?")
            med = r.get("med_turn20")

            # Status emoji
            if status == "locked":
                emoji = "🔒"
                fill_note = f"Queue only — {_qty(r.get('total_buy_qty'))} buyers, no sellers"
            elif status == "at_circuit":
                emoji = "🟢"
                fill_note = "Sellers present on the book"
            else:
                emoji = "🟡"
                dist = r.get("distance_to_circuit")
                fill_note = f"{dist * 100:.1f}% away" if dist is not None else "approaching"

            lines.append(f"{emoji} <b>{sym}</b> — {band_pct} band · ₹{ltp}")
            lines.append(f"   {fill_note}")
            lines.append(f"   Turnover {_turn(med)}")

            # Context line
            ctx = _band_context(band)
            if ctx:
                lines.append(f"   <i>{ctx}</i>")
            tctx = _turnover_context(med)
            if tctx:
                lines.append(f"   <i>{tctx}</i>")
            lines.append("")

    if status_changes:
        lines.append("<b>Status changes:</b>")
        for r, prev in status_changes:
            sym = r["symbol"]
            new_st = r.get("status", "")
            arrow = {"locked": "🔒", "at_circuit": "🟢", "approaching": "🟡"}.get(new_st, "•")
            prev_label = {"locked": "locked", "at_circuit": "at circuit", "approaching": "approaching", "heating": "heating"}.get(prev, prev)
            new_label = {"locked": "locked", "at_circuit": "at circuit (sellers appeared!)", "approaching": "approaching"}.get(new_st, new_st)

            note = ""
            if prev == "locked" and new_st == "at_circuit":
                note = " — sellers just appeared on the book"
            elif prev == "approaching" and new_st in ("at_circuit", "locked"):
                note = " — just reached the circuit"

            lines.append(f"  {arrow} <b>{sym}</b>: {prev_label} → {new_label}{note}")
        lines.append("")

    # Update state
    for r in latest:
        status = r.get("status", "")
        if status != "heating":
            _alerted_symbols[r.get("symbol", "")] = status

    total_at = sum(1 for r in latest if r.get("status") in ("at_circuit", "locked"))
    lines.append(f"📊 {total_at} stocks at circuit right now")
    lines.append("\n<i>What the scanner observes, not a recommendation.</i>")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Morning scorecard — accountability
# --------------------------------------------------------------------------

def _track_bucket(summary: list) -> tuple[dict | None, str]:
    """
    The track-record bucket to headline, preferring the only one that reflects
    a trade that could have filled (live, sellers present). Returns (row, basis).
    """
    by = {s.get("bucket"): s for s in (summary or [])}
    for name, basis in (
        ("live: sellers present", "sellers present"),
        ("all live", "live snapshots, fill mixed"),
        ("all", "all observations, fill unknown"),
    ):
        if by.get(name):
            return by[name], basis
    return None, ""


def format_morning_scorecard(carry_data: dict) -> str | None:
    """Per-stock results from the most recent scored session."""
    daily = carry_data.get("daily", [])
    history = carry_data.get("history", [])
    if not daily:
        return None

    scored = [d for d in daily if d.get("mean_btst") is not None]
    if not scored:
        return None

    d = scored[0]
    as_of = d["as_of"]

    lines = [f"☀️ <b>Morning Brief</b>"]
    lines.append(f"Results for circuit list of {as_of}\n")

    # Per-stock results
    day_stocks = [r for r in history if str(r.get("as_of")) == str(as_of)]
    if day_stocks:
        day_stocks.sort(key=lambda r: -(r.get("btst") or -999))
        for r in day_stocks:
            sym = r.get("symbol", "?")
            btst = r.get("btst")
            hit = r.get("hit4")
            fill = r.get("fillable")
            band = f"{int(float(r.get('band', 0)) * 100)}%" if r.get("band") else "?"

            if fill is False:
                emoji = "🔒"
                result = "locked — no fill possible"
            elif btst is None:
                emoji = "⏳"
                result = "pending"
            elif hit:
                emoji = "✅"
                result = f"reached +4% · move {btst * 100:+.1f}%"
            else:
                emoji = "❌"
                result = f"did not reach · move {btst * 100:+.1f}%"

            lines.append(f"{emoji} <b>{sym}</b> ({band}) — {result}")

        lines.append("")

    # Summary
    n = d.get("n", 0)
    hits = d.get("hits", 0)
    mean = d.get("mean_btst", 0)
    lines.append(f"<b>Score: {hits}/{n}</b> reached +4%")
    lines.append(f"Mean move: {mean * 100:+.1f}%")

    # Running track record — the tradeable bucket (sellers present) when we
    # have it, not the blended number that includes names you couldn't buy.
    summary = carry_data.get("summary", [])
    tb, basis = _track_bucket(summary)
    if tb:
        lines.append(f"\n📋 <b>Running track record</b> ({tb.get('sessions', '?')} sessions · {basis})")
        lines.append(f"Reached +4%: {tb.get('hit4', 0) * 100:.0f}% of the time")
        lines.append(f"Mean move: {tb.get('mean_btst', 0) * 100:+.1f}%")
        lines.append(f"Share closing positive: {tb.get('win_rate', 0) * 100:.0f}%")

    lines.append("\n<i>Historical observations of past price data, not a performance claim.</i>")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Midday pulse — ONE consolidated scanner message at 12:30
# --------------------------------------------------------------------------

def format_midday_pulse(scanner_data: dict, circuit_data: dict) -> str | None:
    """Single midday message combining circuit status + market movers."""
    movers = scanner_data.get("big_movers", [])
    vol = scanner_data.get("unusual_volume", [])
    sectors = scanner_data.get("sector_pulse", [])
    latest = circuit_data.get("latest", [])
    summary = circuit_data.get("summary", {})

    lines = [f"📡 <b>Midday Pulse — 12:30 IST</b>\n"]

    # Circuit status (compact)
    at_c = sum(1 for r in latest if r.get("status") in ("at_circuit", "locked"))
    approaching = sum(1 for r in latest if r.get("status") == "approaching")
    if at_c or approaching:
        lines.append(f"🔔 <b>Circuit:</b> {at_c} at circuit, {approaching} approaching")
        for r in latest[:5]:
            if r.get("status") in ("at_circuit", "locked", "approaching"):
                st = {"locked": "🔒", "at_circuit": "🟢", "approaching": "🟡"}.get(r["status"], "•")
                lines.append(f"  {st} {r['symbol']} ({int(float(r.get('band', 0)) * 100)}%)")
        lines.append("")

    # Top movers (compact, max 5)
    if movers:
        lines.append(f"🚀 <b>Big movers</b> ({len(movers)} stocks up 5%+)")
        for r in movers[:5]:
            lines.append(f"  <b>{r['symbol']}</b> +{r['pchange']:.1f}% · {r['sector']}")
        if len(movers) > 5:
            lines.append(f"  +{len(movers) - 5} more on the dashboard")
        lines.append("")

    # Unusual volume (compact, max 5)
    if vol:
        top_vol = [v for v in vol if v["vol_ratio"] >= 3][:5]
        if top_vol:
            lines.append(f"📊 <b>Volume spikes</b> (3x+ normal)")
            for r in top_vol:
                lines.append(f"  <b>{r['symbol']}</b> {r['vol_ratio']}x · +{r['pchange']:.1f}%")
            lines.append("")

    # Hot sectors (top 3 only)
    if sectors:
        hot = [s for s in sectors if s["avg_change"] > 0.5][:3]
        if hot:
            lines.append("🌡️ <b>Hot sectors</b>")
            for s in hot:
                lines.append(f"  {s['sector']} — {s['adv_pct']:.0f}% advancing, "
                             f"top: {s['best_stock']} +{s['best_change']:.1f}%")
            lines.append("")

    if len(lines) <= 2:
        return None  # nothing interesting today

    lines.append("<i>What's moving — informational, not a recommendation.</i>")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# EOD wrap — final circuit list + day summary
# --------------------------------------------------------------------------

def format_eod_wrap(carry_data: dict, scanner_data: dict | None = None) -> str:
    """End-of-day combined summary for both channels."""
    lines = [f"📋 <b>EOD Wrap — {carry_data.get('as_of', '?')}</b>\n"]

    latest = carry_data.get("latest", [])
    source = carry_data.get("source", "eod")

    if latest:
        lines.append(f"🔔 <b>{len(latest)} stocks</b> closed at circuit ({source} snapshot)\n")
        for r in latest[:15]:
            band = f"{int(float(r.get('band', 0)) * 100)}%" if r.get("band") else "?"
            fill = "✅" if r.get("fillable") else "🔒"
            med = r.get("med_turn20")
            lines.append(f"{fill} <b>{r.get('symbol', '?')}</b> · {band} · "
                         f"₹{r.get('ltp', '?')} · {_turn(med)}")
        if len(latest) > 15:
            lines.append(f"   +{len(latest) - 15} more")
        lines.append("")

        # Context
        fillable = sum(1 for r in latest if r.get("fillable"))
        locked = len(latest) - fillable
        if source == "live":
            lines.append(f"📊 {fillable} with sellers present, {locked} queue only")
    else:
        lines.append("No stocks at circuit today.")

    # EOD scanner highlights (if available)
    if scanner_data:
        breakouts = scanner_data.get("breakouts_52w", [])
        streaks = scanner_data.get("momentum_streaks", [])
        if breakouts:
            lines.append(f"\n📈 <b>52-week breakouts:</b> {len(breakouts)} stocks")
            for r in breakouts[:3]:
                lines.append(f"  {r['symbol']} ₹{r['close']:.0f} · "
                             f"{r['vol_ratio']}x vol · {r['sector']}")
        if streaks:
            lines.append(f"\n🔥 <b>Momentum streaks:</b> {len(streaks)} stocks (3+ up days)")
            for r in streaks[:3]:
                lines.append(f"  {r['symbol']} · {r['streak']} days · +{r['cum_return']:.1f}%")

    pending = carry_data.get("pending", 0)
    if pending:
        lines.append(f"\n⏳ {pending} stocks awaiting next-session results")

    lines.append("\n<i>Tomorrow's morning brief will show how today's list performed.</i>")
    lines.append("<i>Observations, not recommendations.</i>")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Weekly digest — the conversion tool
# --------------------------------------------------------------------------

def format_weekly_digest(carry_data: dict) -> str | None:
    """Weekly summary for both channels."""
    daily = carry_data.get("daily", [])
    summary = carry_data.get("summary", [])
    if not daily:
        return None

    # Last 5 trading days
    recent = [d for d in daily if d.get("mean_btst") is not None][:5]
    if not recent:
        return None

    tb, basis = _track_bucket(summary)

    total_n = sum(d.get("n", 0) for d in recent)
    total_hits = sum(d.get("hits", 0) for d in recent)
    mean_btst = sum(d.get("mean_btst", 0) * d.get("n", 0) for d in recent) / max(total_n, 1)
    best = max((d.get("best", -1) for d in recent), default=0)
    worst = min((d.get("worst", 1) for d in recent), default=0)

    lines = [f"📊 <b>Weekly Digest</b>\n"]
    lines.append(f"This week: <b>{total_n} circuit observations</b> → "
                 f"<b>{total_hits} reached +4%</b> ({total_hits / max(total_n, 1) * 100:.0f}%)")
    lines.append(f"Mean next-session move: {mean_btst * 100:+.1f}%")
    lines.append(f"Best: {best * 100:+.1f}% · Worst: {worst * 100:+.1f}%\n")

    # Per-day breakdown
    lines.append("<b>Day by day:</b>")
    for d in recent:
        dt = d.get("as_of", "?")
        n = d.get("n", 0)
        h = d.get("hits", 0)
        m = d.get("mean_btst", 0)
        lines.append(f"  {dt}: {h}/{n} hit · mean {m * 100:+.1f}%")

    # Cumulative track record — tradeable bucket (sellers present) where we have it
    if tb:
        lines.append(f"\n📋 <b>All-time track record</b> ({basis})")
        lines.append(f"  {tb.get('n', '?')} observations over {tb.get('sessions', '?')} sessions")
        lines.append(f"  Reached +4%: {tb.get('hit4', 0) * 100:.0f}%")
        lines.append(f"  Mean move: {tb.get('mean_btst', 0) * 100:+.1f}%")
        lines.append(f"  Share positive: {tb.get('win_rate', 0) * 100:.0f}%")

    lines.append(f"\n💡 Pro subscribers see circuit flashes in real-time during market hours.")
    lines.append("<i>Historical observations, not a performance claim or projection.</i>")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python telegram_bot.py [test|circuit|midday|eod|morning|weekly]")
        return

    cmd = sys.argv[1]
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if cmd == "test":
        ok1 = _send(FREE_CHAT, "✅ <b>Free channel connected.</b> EOD summaries + morning briefs appear here.")
        ok2 = _send(PAID_CHAT, "✅ <b>Pro channel connected.</b> Real-time circuit flashes + midday pulse appear here.")
        print(f"Free: {'OK' if ok1 else 'FAIL'}, Paid: {'OK' if ok2 else 'FAIL'}")

    elif cmd == "circuit":
        import carry
        data = carry.intraday_payload()
        msg = smart_circuit_alert(data)
        if msg:
            send_paid(msg)
            print(f"[telegram] circuit flash sent")
        else:
            print("[telegram] no changes to alert")

    elif cmd == "midday":
        import carry, scanners
        sd = scanners.intraday_all()
        cd = carry.intraday_payload()
        msg = format_midday_pulse(sd, cd)
        if msg:
            send_paid(msg)
            print(f"[telegram] midday pulse sent")

    elif cmd == "eod":
        import carry
        cd = carry.payload()
        try:
            import scanners
            sd = scanners.eod_all()
        except Exception:
            sd = None
        msg = format_eod_wrap(cd, sd)
        send_both(msg)
        print(f"[telegram] EOD wrap sent to both")

    elif cmd == "morning":
        import carry
        data = carry.payload()
        msg = format_morning_scorecard(data)
        if msg:
            send_both(msg)
            print(f"[telegram] morning brief sent")

    elif cmd == "weekly":
        import carry
        data = carry.payload()
        msg = format_weekly_digest(data)
        if msg:
            send_both(msg)
            print(f"[telegram] weekly digest sent")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
