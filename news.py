"""
Exchange filings for the stock page, from NSE corporate announcements.

These are the company's own disclosures to the exchange -- results, board
outcomes, orders, credit ratings, management changes, clarifications on
media reports -- so they are the primary source rather than commentary.
Each filing is bucketed into a plain category and an importance level so the
page can lead with what matters and fold away routine notices (trading
window closures, newspaper copies, depository certificates).

Media headlines are deliberately not here: the free feeds that carry Indian
market news (Google News RSS and similar) are licensed for personal,
non-commercial use only.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import db
from fetch import BASE, NSESession

ANNOUNCEMENTS = BASE + "/api/corporate-announcements?index=equities&symbol={sym}&from_date={frm}&to_date={to}"
REFERER = BASE + "/companies-listing/corporate-filings-announcements"
LOOKBACK_DAYS = 120
CACHE_TTL = 30 * 60
MAX_ITEMS = 60

# (substring of NSE's category, our label, importance). First match wins, so
# the specific rules sit above the broad ones.
RULES: list[tuple[str, str, str]] = [
    ("financial result", "Results", "high"),
    ("outcome of board meeting", "Board outcome", "high"),
    ("bagging/receiving of orders", "Order win", "high"),
    ("rumour verification", "Clarification on media report", "high"),
    ("news verification", "Clarification on media report", "high"),
    ("clarification", "Clarification on media report", "high"),
    ("spurt in volume", "Exchange query on price/volume", "normal"),
    ("price movement", "Exchange query on price/volume", "normal"),
    ("credit rating", "Credit rating", "high"),
    ("memorandum of understanding", "Agreement / MoU", "high"),
    ("agreement", "Agreement / MoU", "high"),
    ("material issue", "Material disclosure", "high"),
    ("material event", "Material disclosure", "high"),
    ("auditor", "Auditor change", "high"),
    ("resignation", "Management change", "high"),
    ("cessation", "Management change", "high"),
    ("change in director", "Management change", "high"),
    ("change in management", "Management change", "high"),
    ("appointment", "Management change", "normal"),
    ("acquisition", "Deal / restructuring", "high"),
    ("amalgamation", "Deal / restructuring", "high"),
    ("merger", "Deal / restructuring", "high"),
    ("scheme of arrangement", "Deal / restructuring", "high"),
    ("restructuring", "Deal / restructuring", "high"),
    ("sale or disposal", "Deal / restructuring", "high"),
    ("action(s) taken or orders passed", "Legal / regulatory", "high"),
    ("litigation", "Legal / regulatory", "high"),
    ("dispute", "Legal / regulatory", "high"),
    ("default", "Legal / regulatory", "high"),
    ("fraud", "Legal / regulatory", "high"),
    ("pledge", "Promoter / insider", "high"),
    ("encumbrance", "Promoter / insider", "high"),
    ("insider trading", "Promoter / insider", "high"),
    ("sast", "Promoter / insider", "high"),
    ("buyback", "Corporate action", "high"),
    ("bonus", "Corporate action", "high"),
    ("split", "Corporate action", "high"),
    ("dividend", "Corporate action", "normal"),
    ("record date", "Corporate action", "normal"),
    ("rights", "Corporate action", "normal"),
    ("qualified institutional", "Fund raising", "high"),
    ("preferential issue", "Fund raising", "high"),
    ("fund raising", "Fund raising", "high"),
    ("esop", "Employee stock options", "low"),
    ("esos", "Employee stock options", "low"),
    ("allotment of securities", "Allotment", "normal"),
    ("press release", "Press release", "normal"),
    ("investor presentation", "Investor presentation", "normal"),
    ("analysts/institutional investor meet", "Investor meet", "low"),
    ("trading window", "Trading window", "low"),
    ("newspaper", "Newspaper notice", "low"),
    ("shareholders meeting", "Shareholder meeting", "low"),
    ("certificate under sebi", "Compliance certificate", "low"),
    ("loss of share certificate", "Share certificate", "low"),
    ("duplicate", "Share certificate", "low"),
    ("amendment to aoa", "Charter amendment", "low"),
]

_session: NSESession | None = None
_session_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}


def _get_session(fresh: bool = False) -> NSESession:
    global _session
    with _session_lock:
        if _session is None or fresh:
            _session = NSESession(timeout=15, retries=2)
        return _session


def classify(desc: str) -> tuple[str, str]:
    d = (desc or "").lower()
    for key, label, level in RULES:
        if key in d:
            return label, level
    return (desc or "Update").strip() or "Update", "normal"


def _parse_dt(raw: str | None) -> datetime | None:
    for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw or "", fmt)
        except ValueError:
            continue
    return None


def _shape(row: dict) -> dict | None:
    when = _parse_dt(row.get("an_dt")) or _parse_dt(row.get("sort_date"))
    if when is None:
        return None
    desc = (row.get("desc") or "").strip()
    category, importance = classify(desc)
    text = " ".join((row.get("attchmntText") or "").split())
    # Results often arrive under a generic heading ("Updates", "Board outcome").
    if importance != "low" and "financial result" in text.lower():
        category, importance = "Results", "high"
    if len(text) > 420:
        text = text[:417].rstrip() + "…"
    return {
        "date": when.strftime("%Y-%m-%d"),
        "time": when.strftime("%H:%M"),
        "category": category,
        "importance": importance,
        "title": desc or category,
        "text": text,
        "url": row.get("attchmntFile") or None,
    }


def filings(symbol: str) -> dict:
    """Recent exchange filings for one symbol, newest first. Cached 30 min."""
    sym = symbol.strip().upper()
    hit = _cache.get(sym)
    now = time.monotonic()
    if hit and hit[0] > now:
        return hit[1]

    today = db.now_ist().date()
    url = ANNOUNCEMENTS.format(
        sym=sym,
        frm=(today - timedelta(days=LOOKBACK_DAYS)).strftime("%d-%m-%Y"),
        to=today.strftime("%d-%m-%Y"),
    )
    rows = None
    last_err = None
    for fresh in (False, True):
        try:
            rows = _get_session(fresh).get_json(url, referer=REFERER)
            break
        except Exception as exc:
            last_err = exc
    if rows is None:
        # Don't cache a failure for long: NSE blips are usually brief.
        out = {"symbol": sym, "filings": [], "error": f"NSE filings unavailable: {last_err}"}
        _cache[sym] = (now + 60, out)
        return out

    if isinstance(rows, dict):
        rows = rows.get("data") or []
    items = [s for s in (_shape(r) for r in rows if isinstance(r, dict)) if s]
    items.sort(key=lambda x: (x["date"], x["time"]), reverse=True)
    out = {
        "symbol": sym,
        "lookback_days": LOOKBACK_DAYS,
        "source": "NSE corporate announcements",
        "filings": items[:MAX_ITEMS],
    }
    _cache[sym] = (now + CACHE_TTL, out)
    if len(_cache) > 500:
        for k in [k for k, (exp, _) in _cache.items() if exp <= now]:
            _cache.pop(k, None)
    return out
