"""
Fundamental data for Indian equities.

Three NSE endpoints supply what the product needs:

    quote-equity               → PE, sector PE, issued shares, face value
    results_comparision        → quarterly P&L (revenue, PAT, EPS) ~5 quarters
    corporate-share-holdings   → promoter holding %

Market cap = issued shares × last close ÷ 1e7 (crores).
PE TTM is taken directly from NSE (pdSymbolPe) or computed as
    price / sum(last 4 quarters' EPS).
Growth = YoY change in quarterly revenue / PAT.
Earnings trend = trajectory of YoY PAT growth across recent quarters.

Cached in Supabase `company_fundamentals`. Refreshed quarterly (after
results season) or on demand for individual stocks. The screen reads
the cache; the StockDrawer fetches on demand if the cache is stale.

ROCE and debt/equity are not available from these endpoints — they
come from the annual report XBRL, which is a later addition.

Amounts in the results_comparision response are in Rupees Lakhs.
Divide by 100 for Crores.
"""

from __future__ import annotations

import json
import time
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import db
import fetch

_HERE = Path(__file__).resolve().parent
CACHE_DIR = _HERE / "data" / "cache"

# How old a cached row can be before we re-fetch (days).
# Quarterly results come ~45 days after quarter end, so 100 days means
# a refresh covers each results season with margin.
STALE_DAYS = 100

# NSE rate-limit pause between calls.
PAUSE = 0.5

# -------------------------------------------------------------------------
# NSE data fetchers
# -------------------------------------------------------------------------

def _get_sess() -> fetch.NSESession:
    return fetch.NSESession(timeout=20, retries=2, pause=0.4)


def fetch_quote_info(symbol: str, sess=None) -> dict | None:
    """
    From NSE quote-equity: PE, sector PE, face value, issued shares.

    Returns None on failure rather than raising — a missing quote should
    not break a bulk refresh.
    """
    sess = sess or _get_sess()
    url = f"{fetch.BASE}/api/quote-equity?symbol={symbol}"
    try:
        payload = sess.get_json(url)
    except Exception as exc:
        print(f"  quote-equity {symbol}: {exc}")
        return None

    if not isinstance(payload, dict):
        return None

    meta = payload.get("metadata") or {}
    sec = payload.get("securityInfo") or {}
    price = payload.get("priceInfo") or {}
    ind = payload.get("industryInfo") or {}

    ltp = price.get("lastPrice") or price.get("close")

    return {
        "symbol": symbol,
        "name": (payload.get("info") or {}).get("companyName"),
        "industry": ind.get("basicIndustry"),
        "face_value": sec.get("faceValue"),
        "issued_shares": sec.get("issuedSize"),
        "pe_nse": _num(meta.get("pdSymbolPe")),
        "sector_pe": _num(meta.get("pdSectorPe")),
        "ltp": _num(ltp),
    }


def fetch_quarterly_results(symbol: str, sess=None) -> list[dict]:
    """
    From NSE results_comparision: ~5 quarters of revenue, PAT, EPS.

    NSE spells it 'comparision' (their typo, not ours). Amounts are in
    Rupees Lakhs — we convert to Crores (÷ 100) on return.

    Returns an empty list on failure.
    """
    sess = sess or _get_sess()
    url = (f"{fetch.BASE}/api/corporates-financial-results-comparision"
           f"?symbol={symbol}")
    try:
        payload = sess.get_json(url)
    except Exception as exc:
        print(f"  results_comparision {symbol}: {exc}")
        return []

    if not isinstance(payload, dict):
        return []

    rows = payload.get("resCmpData")
    if not isinstance(rows, list) or not rows:
        return []

    quarters = []
    for r in rows:
        if not isinstance(r, dict):
            continue

        # Period identification
        from_dt = r.get("re_FromDate") or r.get("from_date")
        to_dt = r.get("re_ToDate") or r.get("to_date")
        relating = r.get("relatingTo") or ""

        # Financial figures (in Lakhs → Crores)
        revenue = _lakhs_to_cr(r.get("re_TotalIncome") or r.get("re_IncomeFromOperations"))
        operating_profit = _lakhs_to_cr(r.get("re_OperatingProfit"))
        pat = _lakhs_to_cr(r.get("re_PATNetProfit") or r.get("re_NetProfitLoss"))
        eps_basic = _num(r.get("re_BasicEPS"))
        eps_diluted = _num(r.get("re_DilEPS"))

        opm = None
        if revenue and operating_profit and revenue > 0:
            opm = round(operating_profit / revenue * 100, 1)

        quarter_end = _parse_nse_date(to_dt)
        quarter_label = _quarter_label(quarter_end) if quarter_end else relating

        quarters.append({
            "quarter": quarter_label,
            "quarter_end": quarter_end.isoformat() if quarter_end else None,
            "from_date": _parse_nse_date(from_dt).isoformat() if _parse_nse_date(from_dt) else None,
            "relating_to": relating,
            "revenue_cr": revenue,
            "operating_profit_cr": operating_profit,
            "pat_cr": pat,
            "opm_pct": opm,
            "eps_basic": eps_basic,
            "eps_diluted": eps_diluted,
            "consolidated": bool(r.get("consolidated") or r.get("re_Consolidated")),
        })

    # Sort newest first
    quarters.sort(key=lambda q: q.get("quarter_end") or "", reverse=True)
    return quarters


def fetch_shareholding(symbol: str, sess=None) -> dict | None:
    """
    From NSE corporate-share-holdings-master: promoter holding %.

    Returns None on failure.
    """
    sess = sess or _get_sess()
    url = (f"{fetch.BASE}/api/corporate-share-holdings-master"
           f"?symbol={symbol}")
    try:
        payload = sess.get_json(url)
    except Exception as exc:
        print(f"  shareholding {symbol}: {exc}")
        return None

    if not isinstance(payload, dict):
        return None

    # The response structure varies. Look for promoter data in
    # common locations.
    promoter_pct = None
    pledge_pct = None

    # Try the direct fields first
    data = payload.get("data") or payload
    if isinstance(data, list):
        # Some responses return a list of holding categories
        for item in data:
            cat = str(item.get("category") or item.get("desc") or "").lower()
            if "promoter" in cat and "group" not in cat:
                promoter_pct = _num(item.get("shareholdingPercent")
                                    or item.get("holdingPercent")
                                    or item.get("percent"))
                pledge_pct = _num(item.get("pledgePercent")
                                  or item.get("pledgedPercent"))
    elif isinstance(data, dict):
        # Nested structure
        for key in ("promoterHolding", "promoterAndPromoterGroup",
                     "promoterAndGroup"):
            sub = data.get(key)
            if isinstance(sub, dict):
                promoter_pct = _num(sub.get("totalPercent")
                                    or sub.get("percent")
                                    or sub.get("val"))
                pledge_pct = _num(sub.get("pledgedPercent")
                                  or sub.get("pledgePercent"))
                break

    if promoter_pct is None:
        # Fallback: walk the whole payload looking for "promoter" keys
        promoter_pct = _find_promoter_pct(payload)

    return {
        "promoter_pct": promoter_pct,
        "promoter_pledge_pct": pledge_pct,
    }


def _find_promoter_pct(obj, depth=0) -> float | None:
    """Walk a nested dict/list looking for a promoter percentage."""
    if depth > 5:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if "promoter" in kl and "percent" in kl:
                return _num(v)
            if "promoter" in kl and isinstance(v, (dict, list)):
                found = _find_promoter_pct(v, depth + 1)
                if found is not None:
                    return found
        for v in obj.values():
            if isinstance(v, (dict, list)):
                found = _find_promoter_pct(v, depth + 1)
                if found is not None:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_promoter_pct(item, depth + 1)
            if found is not None:
                return found
    return None


# -------------------------------------------------------------------------
# Computation
# -------------------------------------------------------------------------

def compute_fundamentals(symbol: str, sess=None, price: float | None = None) -> dict:
    """
    Full fundamental snapshot for one symbol. Three NSE calls.

    `price` overrides the LTP from quote-equity — pass the panel's last
    close to avoid a timing mismatch during post-market runs.
    """
    sess = sess or _get_sess()

    quote = fetch_quote_info(symbol, sess)
    time.sleep(PAUSE)

    quarters = fetch_quarterly_results(symbol, sess)
    time.sleep(PAUSE)

    holding = fetch_shareholding(symbol, sess)

    # Merge everything
    q = quote or {}
    h = holding or {}

    issued = q.get("issued_shares")
    ltp = price or q.get("ltp")

    market_cap_cr = None
    if issued and ltp:
        market_cap_cr = round(issued * ltp / 1e7, 1)

    # TTM from last 4 quarters
    ttm = _ttm(quarters)

    # PE: prefer NSE's own, fall back to computed
    pe_ttm = q.get("pe_nse")
    if pe_ttm is None and ltp and ttm.get("eps_basic"):
        eps = ttm["eps_basic"]
        if eps > 0:
            pe_ttm = round(ltp / eps, 1)

    # Latest quarter YoY growth
    latest_growth = _latest_yoy_growth(quarters)

    # Earnings trend
    trend = earnings_trend(quarters)

    return {
        "symbol": symbol,
        "name": q.get("name"),
        "industry": q.get("industry"),
        "face_value": q.get("face_value"),
        "issued_shares": issued,
        "market_cap_cr": market_cap_cr,
        "pe_ttm": pe_ttm,
        "sector_pe": q.get("sector_pe"),
        "promoter_pct": h.get("promoter_pct"),
        "promoter_pledge_pct": h.get("promoter_pledge_pct"),
        "ttm_revenue_cr": ttm.get("revenue_cr"),
        "ttm_pat_cr": ttm.get("pat_cr"),
        "latest_revenue_growth_yoy": latest_growth.get("revenue"),
        "latest_pat_growth_yoy": latest_growth.get("pat"),
        "latest_opm_pct": (quarters[0].get("opm_pct")
                           if quarters else None),
        "earnings_trend": trend,
        "quarters": quarters,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def earnings_trend(quarters: list[dict]) -> str:
    """
    Trajectory of YoY PAT growth across recent quarters.

        ↑↑  accelerating — latest 2+ quarters of positive, increasing growth
        ↑   growing      — positive growth, not accelerating
        →   flat         — near zero or mixed
        ↓   declining    — negative growth

    Needs at least 5 quarters (current + 4 for YoY comparison).
    """
    if len(quarters) < 5:
        return "?"

    # Compute YoY growth for each quarter that has a year-ago match.
    # Quarters are sorted newest-first.
    growths = []
    for i, q in enumerate(quarters):
        pat = q.get("pat_cr")
        if pat is None:
            continue
        # Find the same quarter one year ago (4 quarters back)
        yago = _find_year_ago(q, quarters)
        if yago is None or yago.get("pat_cr") is None:
            continue
        base = yago["pat_cr"]
        if abs(base) < 0.01:
            continue
        growths.append(pat / base - 1.0)

    if len(growths) < 2:
        return "?"

    latest = growths[0]

    if latest < -0.05:
        return "↓"

    if abs(latest) <= 0.05:
        return "→"

    # Positive growth. Check acceleration: each quarter better than the next.
    if (len(growths) >= 3
            and all(growths[i] > growths[i + 1] for i in range(min(2, len(growths) - 1)))
            and latest > 0.10):
        return "↑↑"

    return "↑"


def earnings_trend_label(trend: str) -> str:
    return {
        "↑↑": "Accelerating",
        "↑": "Growing",
        "→": "Flat",
        "↓": "Declining",
        "?": "Insufficient data",
    }.get(trend, trend)


# -------------------------------------------------------------------------
# Bulk refresh
# -------------------------------------------------------------------------

_refresh_lock = threading.Lock()


def refresh_symbol(symbol: str, sess=None, price: float | None = None) -> dict:
    """Fetch, compute, and cache one symbol. Returns the computed dict."""
    data = compute_fundamentals(symbol, sess, price)
    try:
        db.save_fundamentals(data)
    except Exception as exc:
        print(f"  save fundamentals {symbol}: {exc}")
    return data


def refresh_bulk(
    symbols: list[str],
    on_progress=None,
    prices: dict[str, float] | None = None,
) -> int:
    """
    Refresh fundamentals for many symbols. Resumable: skips symbols
    already refreshed within STALE_DAYS.

    Returns the number of symbols updated.
    """
    if not _refresh_lock.acquire(blocking=False):
        print("[fundamentals] refresh already running")
        return 0

    try:
        return _refresh_bulk_inner(symbols, on_progress, prices)
    finally:
        _refresh_lock.release()


def _refresh_bulk_inner(symbols, on_progress, prices):
    prices = prices or {}
    sess = _get_sess()
    cutoff = (datetime.now() - timedelta(days=STALE_DAYS)).isoformat()

    # Find which symbols are already fresh
    try:
        fresh = set(db.get_fresh_fundamental_symbols(cutoff))
    except Exception:
        fresh = set()

    todo = [s for s in symbols if s not in fresh]
    if not todo:
        print(f"[fundamentals] all {len(symbols)} symbols are fresh")
        return 0

    print(f"[fundamentals] refreshing {len(todo)} of {len(symbols)} symbols "
          f"({len(fresh)} already fresh)")

    done = 0
    errors = 0
    for i, sym in enumerate(todo):
        try:
            refresh_symbol(sym, sess, prices.get(sym))
            done += 1
        except Exception as exc:
            errors += 1
            print(f"  [{i+1}/{len(todo)}] {sym}: {exc}")

        if on_progress and (i + 1) % 25 == 0:
            on_progress(i + 1, len(todo))

        # Re-warm the session every 200 calls to avoid cookie expiry
        if (i + 1) % 200 == 0:
            try:
                sess = _get_sess()
            except Exception:
                pass

        time.sleep(PAUSE)

    print(f"[fundamentals] done: {done} updated, {errors} errors")
    return done


# -------------------------------------------------------------------------
# Read (from cache or on-demand)
# -------------------------------------------------------------------------

def get_fundamentals(symbol: str, max_age_days: int = STALE_DAYS) -> dict | None:
    """
    Cached fundamentals for one symbol. Returns None if not cached or
    too stale. The caller (analyze.py) decides whether to fetch on demand.
    """
    try:
        row = db.get_fundamentals_row(symbol)
    except Exception:
        return None

    if not row:
        return None

    # Check freshness
    updated = row.get("updated_at")
    if updated and max_age_days:
        try:
            ts = pd.Timestamp(updated)
            if (pd.Timestamp.now() - ts).days > max_age_days:
                return None
        except Exception:
            pass

    return row


def get_fundamentals_or_fetch(symbol: str, sess=None, price=None) -> dict | None:
    """Cache-first, fetch on miss. For the StockDrawer."""
    cached = get_fundamentals(symbol)
    if cached:
        return cached

    try:
        return refresh_symbol(symbol, sess, price)
    except Exception as exc:
        print(f"  fundamentals fetch failed for {symbol}: {exc}")
        return None


def get_screen_data(
    min_mcap_cr: float | None = None,
    max_pe: float | None = None,
    min_growth: float | None = None,
    sector: str | None = None,
    sort_by: str = "market_cap_cr",
    sort_desc: bool = True,
    limit: int = 500,
) -> list[dict]:
    """
    All cached fundamentals, optionally filtered. For the Fundamentals screen.
    """
    try:
        rows = db.get_all_fundamentals()
    except Exception:
        return []

    if not rows:
        return []

    df = pd.DataFrame(rows)

    if min_mcap_cr is not None and "market_cap_cr" in df.columns:
        df = df[df["market_cap_cr"].fillna(0) >= min_mcap_cr]

    if max_pe is not None and "pe_ttm" in df.columns:
        df = df[(df["pe_ttm"].fillna(999) <= max_pe) & (df["pe_ttm"] > 0)]

    if min_growth is not None and "latest_pat_growth_yoy" in df.columns:
        df = df[df["latest_pat_growth_yoy"].fillna(-999) >= min_growth]

    if sector and "industry" in df.columns:
        df = df[df["industry"].str.contains(sector, case=False, na=False)]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not sort_desc, na_position="last")

    df = df.head(limit)

    # Drop the quarters JSONB to keep the response light
    drop = [c for c in ["quarters", "id"] if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    return df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict("records")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _num(v) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _lakhs_to_cr(v) -> float | None:
    x = _num(v)
    if x is None:
        return None
    return round(x / 100, 2)


def _parse_nse_date(s) -> date | None:
    if not s:
        return None
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except ValueError:
            continue
    return None


def _quarter_label(d: date) -> str:
    """FY quarter label from the quarter-end date. Indian FY = Apr–Mar."""
    m = d.month
    y = d.year
    if m <= 3:
        return f"Q4FY{y % 100:02d}"
    elif m <= 6:
        fy = (y + 1) % 100
        return f"Q1FY{fy:02d}"
    elif m <= 9:
        fy = (y + 1) % 100
        return f"Q2FY{fy:02d}"
    else:
        fy = (y + 1) % 100
        return f"Q3FY{fy:02d}"


def _ttm(quarters: list[dict]) -> dict:
    """Trailing twelve months: sum of the 4 most recent quarters."""
    if len(quarters) < 4:
        recent = quarters
    else:
        recent = quarters[:4]

    rev = [q["revenue_cr"] for q in recent if q.get("revenue_cr") is not None]
    pat = [q["pat_cr"] for q in recent if q.get("pat_cr") is not None]
    eps = [q["eps_basic"] for q in recent if q.get("eps_basic") is not None]

    return {
        "revenue_cr": round(sum(rev), 2) if len(rev) == 4 else None,
        "pat_cr": round(sum(pat), 2) if len(pat) == 4 else None,
        "eps_basic": round(sum(eps), 2) if len(eps) == 4 else None,
    }


def _find_year_ago(q: dict, quarters: list[dict]) -> dict | None:
    """Find the quarter from one year ago (same fiscal quarter)."""
    end = q.get("quarter_end")
    if not end:
        return None
    try:
        d = date.fromisoformat(end)
    except (TypeError, ValueError):
        return None
    target = d - timedelta(days=365)
    best = None
    best_dist = 999
    for other in quarters:
        oend = other.get("quarter_end")
        if not oend:
            continue
        try:
            od = date.fromisoformat(oend)
        except (TypeError, ValueError):
            continue
        dist = abs((od - target).days)
        if dist < best_dist and dist < 45:
            best = other
            best_dist = dist
    return best


def _latest_yoy_growth(quarters: list[dict]) -> dict:
    """YoY growth for the most recent quarter."""
    if len(quarters) < 5:
        return {}
    latest = quarters[0]
    yago = _find_year_ago(latest, quarters)
    if not yago:
        return {}

    out = {}
    for key, field in (("revenue", "revenue_cr"), ("pat", "pat_cr")):
        cur = latest.get(field)
        prev = yago.get(field)
        if cur is not None and prev is not None and abs(prev) > 0.01:
            out[key] = round(cur / prev - 1.0, 4)
    return out
