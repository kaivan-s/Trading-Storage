"""
Facts pulled out of exchange filings, so a move can be tied to what was
actually disclosed rather than to a filing's category name.

  results_numbers()  Quarterly revenue / profit / EPS from the company's
                     Integrated Filing XBRL (SEBI's format since 2025), with a
                     year-on-year comparison against the same quarter's filing,
                     or quarter-on-quarter when last year's isn't on NSE.
  pdf_facts()        From the filing PDF: order value for order wins, the
                     action and agency for credit ratings, and the media
                     report a clarification responds to.

Everything is best-effort and cached: a filing NSE won't serve, or a PDF that
is a scanned image, just yields no facts.
"""

from __future__ import annotations

import io
import re
import threading
import time
from datetime import datetime

import pandas as pd

from fetch import BASE, NSESession

INTEGRATED = BASE + "/api/integrated-filing-results?index=equities&symbol={sym}"
REFERER = BASE + "/companies-listing/corporate-integrated-filing"
PAUSE = 0.3
MAX_PDF_BYTES = 4_000_000

REV_TAGS = ("RevenueFromOperations", "TotalRevenueFromOperations", "InterestEarned",
            "TotalIncome", "Income")
PAT_TAGS = ("ProfitLossForPeriod", "ProfitLossForThePeriod", "NetProfitLossForThePeriod",
            "ProfitLossFromContinuingOperations")
EPS_TAGS = ("BasicEarningsLossPerShareFromContinuingOperations",
            "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
            "BasicEarningsPerShareAfterExtraordinaryItems")

_session: NSESession | None = None
_lock = threading.Lock()
_cache: dict[str, tuple[float, object]] = {}
TTL = 6 * 3600


def _sess(fresh: bool = False) -> NSESession:
    global _session
    with _lock:
        if _session is None or fresh:
            _session = NSESession(timeout=15, retries=2)
        return _session


def _cached(key: str, fn):
    hit = _cache.get(key)
    now = time.monotonic()
    if hit and hit[0] > now:
        return hit[1]
    val = fn()
    _cache[key] = (now + TTL, val)
    return val


def _get(url: str, referer: str = REFERER, as_json: bool = False):
    for fresh in (False, True):
        try:
            s = _sess(fresh)
            time.sleep(PAUSE)
            return s.get_json(url, referer=referer) if as_json else s.get(url, referer=referer)
        except Exception:
            continue
    return None


def fno_symbols() -> set[str] | None:
    """Stocks with exchange-traded futures & options. None if NSE is unreachable."""
    def load():
        d = _get(BASE + "/api/master-quote", referer=BASE + "/market-data/equity-derivatives-watch",
                 as_json=True)
        return {str(s).upper() for s in d} if isinstance(d, list) and d else None
    hit = _cache.get("fno")
    if hit and hit[0] > time.monotonic():
        return hit[1]
    val = load()
    # Retry soon after a failure rather than sitting on "unknown" for hours.
    _cache["fno"] = (time.monotonic() + (TTL if val else 300), val)
    return val


# --------------------------------------------------------------------------
# Results numbers from XBRL
# --------------------------------------------------------------------------

def _quarter_context(xml: str) -> str | None:
    """The plain (no dimension) context spanning ~3 months: the quarter."""
    for cid, body in re.findall(r'<xbrli:context id="([^"]+)">(.*?)</xbrli:context>', xml, re.S):
        if "explicitMember" in body or "typedMember" in body:
            continue
        sd = re.search(r"<xbrli:startDate>([^<]+)", body)
        ed = re.search(r"<xbrli:endDate>([^<]+)", body)
        if not (sd and ed):
            continue
        days = (pd.Timestamp(ed.group(1)) - pd.Timestamp(sd.group(1))).days
        if 80 <= days <= 100:
            return cid
    return None


def _tag_value(xml: str, tags, ctx: str) -> float | None:
    for tag in tags:
        m = re.search(rf'<[\w-]+:{tag}\s+contextRef="{re.escape(ctx)}"[^>]*>([^<]+)<', xml)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _xbrl_numbers(url: str) -> dict | None:
    def load():
        r = _get(url)
        if r is None:
            return None
        xml = r.text
        ctx = _quarter_context(xml)
        if not ctx:
            return None
        out = {
            "revenue": _tag_value(xml, REV_TAGS, ctx),
            "profit": _tag_value(xml, PAT_TAGS, ctx),
            "eps": _tag_value(xml, EPS_TAGS, ctx),
        }
        return out if out["revenue"] is not None or out["profit"] is not None else None
    return _cached(f"xbrl:{url}", load)


def _filing_rows(symbol: str) -> list[dict]:
    def load():
        d = _get(INTEGRATED.format(sym=symbol), as_json=True)
        rows = (d or {}).get("data") if isinstance(d, dict) else d
        out = []
        for r in rows or []:
            if "financial" not in (r.get("type") or "").lower() or not r.get("xbrl"):
                continue
            qe = pd.to_datetime(r.get("qe_Date"), format="%d-%b-%Y", errors="coerce")
            bc = pd.to_datetime(r.get("broadcast_Date"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
            if pd.isna(qe):
                continue
            out.append({"qe": qe, "broadcast": bc, "basis": r.get("consolidated") or "",
                        "xbrl": r["xbrl"]})
        return out
    return _cached(f"rows:{symbol}", load)


def _pick(rows: list[dict], qe: pd.Timestamp, basis: str | None = None) -> dict | None:
    cand = [r for r in rows if abs((r["qe"] - qe).days) <= 5]
    if basis:
        same = [r for r in cand if r["basis"] == basis]
        cand = same or cand
    cand.sort(key=lambda r: 0 if r["basis"] == "Consolidated" else 1)
    return cand[0] if cand else None


def _chg(a, b):
    if a is None or b is None or b == 0:
        return None
    # A swing from loss to profit (or back) has no meaningful percentage.
    if (a > 0) != (b > 0):
        return None
    return round(a / b - 1.0, 4)


def _cr(v):
    return None if v is None else round(v / 1e7, 2)


def results_numbers(symbol: str, around: str | None = None) -> dict | None:
    """Numbers for the results filed on/just before `around` (YYYY-MM-DD)."""
    rows = _filing_rows(symbol.upper())
    if not rows:
        return None
    day = pd.Timestamp(around) if around else pd.Timestamp.today()
    filed = [r for r in rows if pd.notna(r["broadcast"])
             and -1 <= (day.normalize() - r["broadcast"].normalize()).days <= 3]
    if filed:
        qe = max(r["qe"] for r in filed)
    else:
        past = [r["qe"] for r in rows if r["qe"] <= day]
        if not past:
            return None
        qe = max(past)
    cur_row = _pick(rows, qe)
    cur = _xbrl_numbers(cur_row["xbrl"]) if cur_row else None
    if not cur:
        return None

    basis = cur_row["basis"]
    prev, compare = None, None
    yoy_row = _pick(rows, qe - pd.DateOffset(years=1), basis)
    if yoy_row:
        prev, compare = _xbrl_numbers(yoy_row["xbrl"]), "YoY"
    if not prev:
        qoq_row = _pick(rows, (qe - pd.DateOffset(months=3)) + pd.offsets.MonthEnd(0), basis)
        if qoq_row:
            prev, compare = _xbrl_numbers(qoq_row["xbrl"]), "QoQ"

    q = (qe.month - 4) % 12 // 3 + 1
    fy = qe.year + (1 if qe.month >= 4 else 0)
    out = {
        "quarter": f"Q{q} FY{str(fy)[-2:]}",
        "quarter_end": qe.strftime("%Y-%m-%d"),
        "basis": basis or "Consolidated",
        "revenue_cr": _cr(cur.get("revenue")),
        "profit_cr": _cr(cur.get("profit")),
        "eps": cur.get("eps"),
        "compare": compare,
        "revenue_chg": _chg(cur.get("revenue"), (prev or {}).get("revenue")),
        "profit_chg": _chg(cur.get("profit"), (prev or {}).get("profit")),
        "prev_profit_cr": _cr((prev or {}).get("profit")),
    }
    if prev and cur.get("profit") is not None and prev.get("profit") is not None:
        if prev["profit"] <= 0 < cur["profit"]:
            out["profit_swing"] = "turned to profit"
        elif cur["profit"] <= 0 < prev["profit"]:
            out["profit_swing"] = "swung to a loss"
    return out


def describe_results(r: dict | None) -> str | None:
    if not r:
        return None
    bits = []
    if r.get("revenue_cr") is not None:
        s = f"revenue ₹{r['revenue_cr']:,.0f} cr"
        if r.get("revenue_chg") is not None:
            s += f" ({r['revenue_chg'] * 100:+.0f}% {r['compare']})"
        bits.append(s)
    if r.get("profit_cr") is not None:
        s = f"profit ₹{r['profit_cr']:,.0f} cr"
        if r.get("profit_chg") is not None:
            s += f" ({r['profit_chg'] * 100:+.0f}% {r['compare']})"
        elif r.get("profit_swing"):
            s += f" ({r['profit_swing']} {r['compare']})"
        bits.append(s)
    if not bits:
        return None
    return f"{r['quarter']} {r['basis'].lower()}: " + ", ".join(bits)


# --------------------------------------------------------------------------
# Facts from filing PDFs
# --------------------------------------------------------------------------

AMOUNT = re.compile(
    r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)\s*(crores?|cr\b\.?|lakhs?|lacs?|million|mn\b|billion|bn\b)",
    re.I,
)
UNIT_TO_CR = {"crore": 1, "crores": 1, "cr": 1, "cr.": 1, "lakh": 0.01, "lakhs": 0.01,
              "lac": 0.01, "lacs": 0.01, "million": 0.1, "mn": 0.1, "billion": 100, "bn": 100}
AGENCIES = ("CRISIL", "ICRA", "CARE", "India Ratings", "Acuite", "Brickwork", "Infomerics",
            "Fitch", "Moody", "S&P")
TITLE = re.compile(r"(?:titled|captioned|headlined|heading|news item|article)\s*[:\-]?\s*[\"“‘']([^\"”’']{12,220})[\"”’']", re.I)
# Exchange-initiated: "...news item captioned <headline>. The response from the Company is awaited."
TITLE_BARE = re.compile(r"news item captioned\s+(.{12,240}?)\.\s+The response", re.I)
RATING = re.compile(r"\b(AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|A1\+?|A2\+?)\s*/\s*(Stable|Positive|Negative|Watch[\w ]{0,30})", re.I)


def _pdf_text(url: str, max_pages: int = 4) -> str:
    def load():
        r = _get(url, referer=BASE + "/companies-listing/corporate-filings-announcements")
        if r is None or not r.content or len(r.content) > MAX_PDF_BYTES:
            return ""
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(r.content))
            return " ".join((p.extract_text() or "") for p in reader.pages[:max_pages])
        except Exception:
            return ""
    return _cached(f"pdf:{url}", load)


def _largest_amount_cr(text: str) -> float | None:
    best = None
    for num, unit in AMOUNT.findall(text):
        try:
            v = float(num.replace(",", "")) * UNIT_TO_CR.get(unit.lower().rstrip("."), 0)
        except ValueError:
            continue
        if v and (best is None or v > best):
            best = v
    return round(best, 2) if best else None


def pdf_facts(filing: dict) -> dict:
    """Category-specific facts for one filing (as shaped by news._shape)."""
    cat = filing.get("category") or ""
    if cat not in ("Order win", "Credit rating", "Clarification on media report", "Deal / restructuring"):
        return {}
    text = " ".join(filter(None, [filing.get("text"), _pdf_text(filing["url"]) if filing.get("url") else ""]))
    text = " ".join(text.split())
    if not text:
        return {}
    low = text.lower()
    out: dict = {}
    if cat in ("Order win", "Deal / restructuring"):
        amt = _largest_amount_cr(text)
        if amt:
            out["value_cr"] = amt
    elif cat == "Credit rating":
        # "CARE" only in capitals -- lowercase "care" is ordinary prose.
        agency = next((a for a in AGENCIES if (a in text if a == "CARE" else a.lower() in low)), None)
        if "downgrad" in low:
            action = "downgraded"
        elif "upgrad" in low:
            action = "upgraded"
        elif "outlook" in low and ("positive" in low or "negative" in low):
            action = "revised outlook to " + ("positive" if "positive" in low else "negative")
        elif "reaffirm" in low:
            action = "reaffirmed"
        elif "assigned" in low:
            action = "assigned"
        else:
            action = None
        if agency:
            out["agency"] = agency
        if action:
            out["rating_action"] = action
        m = RATING.search(text)
        if m:
            out["rating"] = f"{m.group(1).upper()}/{m.group(2).strip().title()}"
    elif cat == "Clarification on media report":
        m = TITLE.search(text) or TITLE_BARE.search(text)
        if m:
            out["news_title"] = m.group(1).strip()
        if "response from the company is awaited" in low:
            out["company_stance"] = "had not yet responded to the exchange's query"
        elif any(p in low for p in ("not aware", "no such", "denies", "baseless", "speculative", "no information")):
            out["company_stance"] = "said it had no such information / the report was speculative"
        elif "confirm" in low:
            out["company_stance"] = "confirmed the report"
    return out


def describe_pdf_facts(cat: str, f: dict, ttm_revenue_cr: float | None = None) -> str | None:
    if not f:
        return None
    if f.get("value_cr"):
        s = f"{cat}: value ₹{f['value_cr']:,.0f} cr"
        if ttm_revenue_cr:
            s += f" (~{f['value_cr'] / ttm_revenue_cr * 100:.0f}% of trailing-year revenue)"
        return s
    if f.get("rating_action") or f.get("rating"):
        s = f"Credit rating {f.get('rating_action') or 'communicated'}"
        if f.get("rating"):
            s += f" at {f['rating']}"
        return s + (f" by {f['agency']}" if f.get("agency") else "")
    if f.get("news_title") or f.get("company_stance"):
        s = "Clarification"
        if f.get("news_title"):
            s += f" on the report \u201c{f['news_title']}\u201d"
        if f.get("company_stance"):
            s += f"; the company {f['company_stance']}"
        return s
    return None
