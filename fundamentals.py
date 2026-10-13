"""
Fundamental data for Indian equities — powered by Yahoo Finance.

Yahoo Finance provides:
    info               → PE, market cap, shares outstanding, sector, industry
    quarterly_income_stmt → quarterly P&L (revenue, net income, EPS)
    info.heldPercentInsiders → promoter/insider holding %

Market cap is taken directly from Yahoo (in INR, converted to Crores).
Growth = YoY change in quarterly revenue / PAT.
Earnings trend = trajectory of YoY PAT growth across recent quarters.

Cached in Supabase `company_fundamentals`. Refreshed quarterly (after
results season) or on demand for individual stocks. The screen reads
the cache; the StockDrawer fetches on demand if the cache is stale.
"""

from __future__ import annotations

import json
import time
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import db

_HERE = Path(__file__).resolve().parent
CACHE_DIR = _HERE / "data" / "cache"

# How old a cached row can be before we re-fetch (days).
STALE_DAYS = 100

# Pause between Yahoo Finance calls to avoid rate limits.
PAUSE = 0.3

# -------------------------------------------------------------------------
# Yahoo Finance data fetchers
# -------------------------------------------------------------------------

def fetch_yf_fundamentals(symbol: str, price: float | None = None) -> dict:
    """
    Fetch all fundamental data for one NSE symbol via Yahoo Finance.

    Yahoo uses the `.NS` suffix for NSE stocks (e.g. RELIANCE.NS).
    Returns a flat dict ready for compute/save.
    """
    yf_symbol = f"{symbol}.NS"

    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info or {}
    except Exception as exc:
        print(f"  yf info {symbol}: {exc}")
        info = {}

    # --- Quote info ---
    name = info.get("longName") or info.get("shortName")
    industry = info.get("industry")
    sector = info.get("sector")
    ltp = price or info.get("currentPrice") or info.get("previousClose")
    pe_ttm = _num(info.get("trailingPE"))
    forward_pe = _num(info.get("forwardPE"))
    shares_outstanding = info.get("sharesOutstanding")

    # Market cap: Yahoo gives it in INR, convert to Crores (÷ 1e7)
    market_cap_raw = info.get("marketCap")
    market_cap_cr = round(market_cap_raw / 1e7, 1) if market_cap_raw else None

    # If Yahoo didn't give market cap but we have shares + price, compute it
    if market_cap_cr is None and shares_outstanding and ltp:
        market_cap_cr = round(shares_outstanding * ltp / 1e7, 1)

    # --- Promoter / insider holding ---
    # Yahoo's heldPercentInsiders is the closest proxy for Indian promoter holding
    promoter_pct = _num(info.get("heldPercentInsiders"))
    if promoter_pct is not None:
        promoter_pct = round(promoter_pct * 100, 2)

    # --- Quarterly income statement ---
    quarters = []
    try:
        qs = ticker.quarterly_income_stmt
        if qs is not None and not qs.empty:
            for col in qs.columns:
                q_end = col
                if hasattr(q_end, "date"):
                    q_end_date = q_end.date()
                else:
                    q_end_date = pd.Timestamp(q_end).date()

                quarter_label = _quarter_label(q_end_date)

                revenue = _val_cr(qs, "Total Revenue", col)
                if revenue is None:
                    revenue = _val_cr(qs, "Operating Revenue", col)
                net_income = _val_cr(qs, "Net Income", col)
                operating_income = _val_cr(qs, "Operating Income", col)
                gross_profit = _val_cr(qs, "Gross Profit", col)
                ebitda = _val_cr(qs, "EBITDA", col)
                basic_eps = _val(qs, "Basic EPS", col)
                diluted_eps = _val(qs, "Diluted EPS", col)

                opm = None
                if revenue and operating_income and revenue > 0:
                    opm = round(operating_income / revenue * 100, 1)

                quarters.append({
                    "quarter": quarter_label,
                    "quarter_end": q_end_date.isoformat(),
                    "revenue_cr": revenue,
                    "operating_profit_cr": operating_income,
                    "pat_cr": net_income,
                    "opm_pct": opm,
                    "eps_basic": basic_eps,
                    "eps_diluted": diluted_eps,
                    "ebitda_cr": ebitda,
                    "gross_profit_cr": gross_profit,
                })
    except Exception as exc:
        print(f"  yf quarterly {symbol}: {exc}")

    # Sort newest first
    quarters.sort(key=lambda q: q.get("quarter_end") or "", reverse=True)

    # --- Compute derived fields ---

    # TTM from last 4 quarters
    ttm = _ttm(quarters)

    # PE: prefer Yahoo's, fall back to computed
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
        "name": name,
        "industry": industry or sector,
        "face_value": None,  # not available from Yahoo
        "issued_shares": shares_outstanding,
        "market_cap_cr": market_cap_cr,
        "pe_ttm": pe_ttm,
        "sector_pe": None,  # not available from Yahoo
        "promoter_pct": promoter_pct,
        "promoter_pledge_pct": None,  # not available from Yahoo
        "ttm_revenue_cr": ttm.get("revenue_cr"),
        "ttm_pat_cr": ttm.get("pat_cr"),
        "latest_revenue_growth_yoy": latest_growth.get("revenue"),
        "latest_pat_growth_yoy": latest_growth.get("pat"),
        "latest_opm_pct": (quarters[0].get("opm_pct") if quarters else None),
        "earnings_trend": trend,
        "quarters": quarters,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _val(df, row_label: str, col) -> float | None:
    """Extract a scalar value from a DataFrame cell."""
    if row_label not in df.index:
        return None
    v = df.loc[row_label, col]
    if pd.isna(v):
        return None
    return round(float(v), 2)


def _val_cr(df, row_label: str, col) -> float | None:
    """Extract a value and convert from INR to Crores (÷ 1e7)."""
    v = _val(df, row_label, col)
    if v is None:
        return None
    return round(v / 1e7, 2)


# -------------------------------------------------------------------------
# Earnings trend
# -------------------------------------------------------------------------

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

    growths = []
    for i, q in enumerate(quarters):
        pat = q.get("pat_cr")
        if pat is None:
            continue
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


def refresh_symbol(symbol: str, price: float | None = None, **_kw) -> dict:
    """Fetch, compute, and cache one symbol. Returns the computed dict."""
    data = fetch_yf_fundamentals(symbol, price)
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
            refresh_symbol(sym, prices.get(sym))
            done += 1
        except Exception as exc:
            errors += 1
            print(f"  [{i+1}/{len(todo)}] {sym}: {exc}")

        if on_progress and (i + 1) % 25 == 0:
            on_progress(i + 1, len(todo))

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(todo)}] progress: {done} done, {errors} errors")

        time.sleep(PAUSE)

    print(f"[fundamentals] done: {done} updated, {errors} errors")
    return done


# -------------------------------------------------------------------------
# Read (from cache or on-demand)
# -------------------------------------------------------------------------

def get_fundamentals(symbol: str, max_age_days: int = STALE_DAYS) -> dict | None:
    """
    Cached fundamentals for one symbol. Returns None if not cached or
    too stale.
    """
    try:
        row = db.get_fundamentals_row(symbol)
    except Exception:
        return None

    if not row:
        return None

    updated = row.get("updated_at")
    if updated and max_age_days:
        try:
            ts = pd.Timestamp(updated)
            if (pd.Timestamp.now() - ts).days > max_age_days:
                return None
        except Exception:
            pass

    return row


def get_fundamentals_or_fetch(symbol: str, price=None, **_kw) -> dict | None:
    """Cache-first, fetch on miss. For the StockDrawer."""
    cached = get_fundamentals(symbol)
    if cached:
        return cached

    try:
        return refresh_symbol(symbol, price)
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


# -------------------------------------------------------------------------
# Results calendar
# -------------------------------------------------------------------------

def fetch_earnings_date(symbol: str) -> dict | None:
    """
    Fetch next earnings date + estimates from Yahoo Finance.
    Returns a dict ready for db.save_results_calendar, or None.
    """
    yf_symbol = f"{symbol}.NS"
    try:
        ticker = yf.Ticker(yf_symbol)
        cal = ticker.calendar
    except Exception as exc:
        print(f"  yf calendar {symbol}: {exc}")
        return None

    if not cal or not isinstance(cal, dict):
        return None

    earnings_raw = cal.get("Earnings Date")
    if not earnings_raw:
        return None

    if isinstance(earnings_raw, list):
        earnings_date = earnings_raw[0]
    else:
        earnings_date = earnings_raw

    if hasattr(earnings_date, "isoformat"):
        earnings_date_str = earnings_date.isoformat()
    else:
        earnings_date_str = str(earnings_date)

    eps_est = _num(cal.get("Earnings Average"))
    eps_high = _num(cal.get("Earnings High"))
    eps_low = _num(cal.get("Earnings Low"))
    rev_est = _num(cal.get("Revenue Average"))
    rev_est_cr = round(rev_est / 1e7, 0) if rev_est else None

    ex_div = cal.get("Ex-Dividend Date")
    ex_div_str = None
    if ex_div and hasattr(ex_div, "isoformat"):
        ex_div_str = ex_div.isoformat()

    return {
        "symbol": symbol,
        "earnings_date": earnings_date_str,
        "eps_estimate": eps_est,
        "eps_high": eps_high,
        "eps_low": eps_low,
        "revenue_estimate_cr": rev_est_cr,
        "ex_dividend_date": ex_div_str,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def refresh_results_calendar(
    symbols: list[str],
    on_progress=None,
) -> int:
    """
    Fetch upcoming earnings dates for all symbols and save to DB.
    Returns the number of entries saved.
    """
    from datetime import date as date_cls
    today = date_cls.today().isoformat()

    # Clear future entries before refreshing
    try:
        db.clear_results_calendar_future(today)
    except Exception:
        pass

    saved = 0
    batch = []
    for i, sym in enumerate(symbols):
        try:
            entry = fetch_earnings_date(sym)
            if entry and entry.get("earnings_date") and entry["earnings_date"] >= today:
                batch.append(entry)
        except Exception as exc:
            print(f"  [{i+1}/{len(symbols)}] calendar {sym}: {exc}")

        # Save in batches of 50
        if len(batch) >= 50:
            try:
                saved += db.save_results_calendar(batch)
            except Exception as exc:
                print(f"  save calendar batch: {exc}")
            batch = []

        if on_progress and (i + 1) % 50 == 0:
            on_progress(i + 1, len(symbols))

        if (i + 1) % 100 == 0:
            print(f"  [calendar {i+1}/{len(symbols)}] {saved} saved so far")

        time.sleep(0.2)

    # Save remaining
    if batch:
        try:
            saved += db.save_results_calendar(batch)
        except Exception as exc:
            print(f"  save calendar batch: {exc}")

    print(f"[calendar] done: {saved} entries saved from {len(symbols)} symbols")
    return saved


def get_results_calendar(days_ahead: int = 45) -> list[dict]:
    """
    Read results calendar from DB, enriched with fundamentals data.
    Returns entries from today to days_ahead days from now.
    """
    from datetime import date as date_cls
    today = date_cls.today()
    end = (today + timedelta(days=days_ahead)).isoformat()

    rows = db.get_results_calendar(from_date=today.isoformat(), to_date=end)
    if not rows:
        return []

    # Enrich with fundamentals (earnings trend, market cap, PE)
    symbols = [r["symbol"] for r in rows]
    fund_lookup = {}
    try:
        from supabase import Client
        client = db.get_client()
        resp = (client.table("company_fundamentals")
                .select("symbol,name,industry,earnings_trend,market_cap_cr,pe_ttm")
                .in_("symbol", symbols)
                .execute())
        fund_lookup = {r["symbol"]: r for r in (resp.data or [])}
    except Exception:
        pass

    for row in rows:
        f = fund_lookup.get(row["symbol"], {})
        row["name"] = f.get("name")
        row["industry"] = f.get("industry")
        row["earnings_trend"] = f.get("earnings_trend")
        row["market_cap_cr"] = f.get("market_cap_cr")
        row["pe_ttm"] = f.get("pe_ttm")

    rows.sort(key=lambda r: (r.get("earnings_date") or "", r.get("symbol") or ""))
    return rows


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
