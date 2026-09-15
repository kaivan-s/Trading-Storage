"""
Supabase integration for storing predictions and caching data.

Tables (create these in Supabase SQL editor):

-- For Tom predictions logged daily
CREATE TABLE tom_predictions (
    id BIGSERIAL PRIMARY KEY,
    scan_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,
    sector TEXT,
    trigger NUMERIC,
    price_at_scan NUMERIC,
    to_trigger NUMERIC,
    vol_expand NUMERIC,
    pchange NUMERIC,
    pos_hi NUMERIC,
    rsi NUMERIC,
    cmf NUMERIC,
    vol_ratio NUMERIC,
    why TEXT,
    score NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(scan_date, symbol)
);

-- If the table already exists without score:
-- ALTER TABLE tom_predictions ADD COLUMN IF NOT EXISTS score NUMERIC;

-- Next-day outcomes for verification
CREATE TABLE tom_outcomes (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES tom_predictions(id),
    scan_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    next_date DATE,
    next_open NUMERIC,
    next_high NUMERIC,
    next_low NUMERIC,
    next_close NUMERIC,
    next_volume BIGINT,
    broke_out BOOLEAN,
    gain_from_trigger NUMERIC,
    gain_from_close NUMERIC,
    verified_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(scan_date, symbol)
);

-- Cached daily scan results
CREATE TABLE daily_cache (
    id BIGSERIAL PRIMARY KEY,
    cache_date DATE NOT NULL UNIQUE,
    scan_rows JSONB,
    coil_rows JSONB,
    buys JSONB,
    breakouts JSONB,
    tom JSONB,
    n_stocks INT,
    n_sectors INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Leaders at rest is the coil pool ordered by 12-1 momentum, so it lives as
-- one column on coiled_bases rather than a second table holding the same
-- rows. Run this once on an existing install:
-- ALTER TABLE coiled_bases ADD COLUMN IF NOT EXISTS mom12_1 NUMERIC;

-- The UI reads these tables directly and renders them as-is, so anything it
-- displays has to be stored rather than recomputed on the way out. Run once:
-- ALTER TABLE coiled_bases ADD COLUMN IF NOT EXISTS rest_rank INT;
-- ALTER TABLE coiled_bases ADD COLUMN IF NOT EXISTS coil_days INT;
-- ALTER TABLE coiled_bases ADD COLUMN IF NOT EXISTS recommended BOOLEAN;
-- ALTER TABLE coiled_bases ADD COLUMN IF NOT EXISTS episode_days INT;
-- ALTER TABLE coiled_bases ADD COLUMN IF NOT EXISTS episode_new BOOLEAN;
-- ALTER TABLE setups ADD COLUMN IF NOT EXISTS episode_days INT;
-- ALTER TABLE setups ADD COLUMN IF NOT EXISTS episode_new BOOLEAN;

-- One row per basing episode: a base keeps its identity from the session it
-- first qualifies to the session it resolves, which is what lets the app say
-- "this broke out" instead of silently dropping the row. Keyed on the start
-- date rather than the symbol alone, because a stock bases more than once.
CREATE TABLE base_episodes (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    sector TEXT,
    started_on DATE NOT NULL,
    entry_price NUMERIC,
    entry_trigger NUMERIC,
    state TEXT NOT NULL,
    state_since DATE,
    reason TEXT,
    lost_gates TEXT,
    last_seen_on DATE,
    last_close NUMERIC,
    peak_close NUMERIC,
    coil_sessions INT,
    age INT,
    gap INT,
    below INT,
    mom12_1 NUMERIC,
    triggered_on DATE,
    trigger_age INT,
    trigger_vol BOOLEAN,
    resolved_on DATE,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, started_on)
);

CREATE INDEX idx_base_episodes_state ON base_episodes(state, resolved_on);
CREATE INDEX idx_base_episodes_since ON base_episodes(state_since);

CREATE INDEX idx_tom_predictions_date ON tom_predictions(scan_date);
CREATE INDEX idx_tom_outcomes_date ON tom_outcomes(scan_date);
CREATE INDEX idx_daily_cache_date ON daily_cache(cache_date);
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

# Load .env file
_env_path = Path(__file__).resolve().parent / ".env"
if not _env_path.exists():
    _env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

_client = None


def get_client():
    """Lazy-load Supabase client."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def _int(v):
    """
    Coerce to a Python int, for the columns declared INT.

    A pandas column of counts holding a single NaN is float64, so a count of
    17 arrives here as 17.0 and Postgres rejects the whole insert with
    "invalid input syntax for type integer: 17.0". `_clean` cannot prevent
    it, having no way to know which columns are integral -- so the integer
    columns say so explicitly at the call site.
    """
    v = _clean(v)
    if v is None:
        return None
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _clean(v):
    """Convert pandas/numpy types to JSON-safe Python types."""
    import numpy as np
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if pd.isna(x) or np.isinf(x):
            return None
        return round(x, 6)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _row_dict(row: dict | pd.Series) -> dict:
    """Convert a row to a clean dict."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    return {k: _clean(v) for k, v in row.items()}


_SCORE_RE = re.compile(r"(?:momentum\s+)?score\s+(\d+(?:\.\d+)?)", re.I)


def prediction_score(row: dict) -> float | None:
    """Score from the saved column, or parsed from the why line."""
    v = row.get("score")
    if v is not None and v != "":
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    why = row.get("why") or ""
    m = _SCORE_RE.search(str(why))
    return float(m.group(1)) if m else None


def _with_score(rows: list[dict]) -> list[dict]:
    for row in rows:
        if row.get("score") is None:
            s = prediction_score(row)
            if s is not None:
                row["score"] = s
    return rows


# Trading session is 09:00–16:00 IST. A run after midnight IST (before 09:00)
# still belongs to yesterday's For Tom list.
IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = time(9, 0)
SESSION_CLOSE = time(16, 0)


def now_ist(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(IST)
    if now.tzinfo is None:
        return now.replace(tzinfo=IST)
    return now.astimezone(IST)


def session_date(now: datetime | None = None) -> date:
    now = now_ist(now)
    d = now.date()
    if now.time() < SESSION_OPEN:
        d -= timedelta(days=1)
    return d


def verify_scan_date(saved_dates: list[str], now: datetime | None = None) -> str | None:
    """Which saved scan to show for manual / live verify (IST session clock)."""
    if not saved_dates:
        return None
    now = now_ist(now)
    sess = session_date(now).isoformat()
    if now.time() < SESSION_OPEN:
        return next((d for d in saved_dates if d <= sess), saved_dates[0])
    return next((d for d in saved_dates if d < sess), saved_dates[0])


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def _as_date(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def is_past_holiday(d: date, sessions: set[date] | None, today: date) -> bool:
    """Weekend, or a past weekday that never produced a session file."""
    if is_weekend(d):
        return True
    if not sessions or d >= today:
        return False
    return d not in sessions


def last_trading_date(d: date, sessions: set[date] | None, limit: int = 21) -> date:
    """Walk backward from `d` to the last real session (skip weekends/holidays)."""
    cur = d
    for _ in range(limit):
        if is_weekend(cur):
            cur -= timedelta(days=1)
            continue
        if sessions and cur not in sessions:
            cur -= timedelta(days=1)
            continue
        return cur
    return cur


def next_trading_date(d: date, sessions: set[date] | None, today: date,
                      limit: int = 21) -> date | None:
    """First session strictly after `d`. Future weekdays are assumed to trade."""
    cur = d + timedelta(days=1)
    for _ in range(limit):
        if is_past_holiday(cur, sessions, today):
            cur += timedelta(days=1)
            continue
        return cur
    return None


def session_calendar(now: datetime | None, sessions: set[date] | None,
                     as_of: str | date | None = None) -> dict:
    """
    Holiday / last-session context for Track Record.

    `sessions` should be known trading days from the loaded panel. Without
    that we only skip weekends.
    """
    now = now_ist(now)
    today = now.date()
    known = {x for x in (sessions or set()) if isinstance(x, date)}
    as_of_d = _as_date(as_of)
    if as_of_d:
        known.add(as_of_d)

    yesterday = today - timedelta(days=1)
    yesterday_holiday = is_past_holiday(yesterday, known or None, today)

    today_weekend = is_weekend(today)
    today_nse_holiday = (
        not today_weekend
        and today not in known
        and bool(known)
        and max(known) < today
        and now.time() >= SESSION_CLOSE
    )
    today_holiday = today_weekend or today_nse_holiday
    today_kind = (
        "weekend" if today_weekend
        else "nse_holiday" if today_nse_holiday
        else "session"
    )

    last_session = last_trading_date(
        yesterday if today_holiday else session_date(now),
        known or None,
    )
    nxt = next_trading_date(last_session, known or None, today)

    return {
        "today": today.isoformat(),
        "today_holiday": today_holiday,
        "today_kind": today_kind,
        "yesterday_holiday": yesterday_holiday,
        "last_session": last_session.isoformat(),
        "next_session": nxt.isoformat() if nxt else None,
    }


# ---------------------------------------------------------------------------
# Tom Predictions
# ---------------------------------------------------------------------------

def save_tom_predictions(scan_date: date | str, tom_rows: pd.DataFrame) -> int:
    """
    Save For Tom predictions for a given date.
    Full replacement: deletes existing rows for that date, then inserts new ones.
    Returns number of rows inserted.
    """
    if tom_rows is None or tom_rows.empty:
        return 0
    
    scan_date = str(scan_date)[:10]
    client = get_client()
    
    records = []
    for _, r in tom_rows.iterrows():
        rec = {
            "scan_date": scan_date,
            "symbol": str(r.get("symbol", "")),
            "kind": str(r.get("kind", "")),
            "sector": r.get("sector"),
            "trigger": _clean(r.get("trigger")),
            "price_at_scan": _clean(r.get("ltp") or r.get("adj")),
            "to_trigger": _clean(r.get("to_trigger")),
            "vol_expand": _clean(r.get("vol_expand")),
            "pchange": _clean(r.get("pchange")),
            "pos_hi": _clean(r.get("pos_hi")),
            "rsi": _clean(r.get("rsi")),
            "cmf": _clean(r.get("cmf")),
            "vol_ratio": _clean(r.get("vol_ratio")),
            "why": r.get("why"),
            "score": _clean(r.get("score")) if r.get("score") is not None
                     else prediction_score({"why": r.get("why")}),
        }
        records.append(rec)
    
    if not records:
        return 0
    
    # Full replacement: delete existing rows for this date, then insert
    try:
        client.table("tom_predictions").delete().eq("scan_date", scan_date).execute()
    except Exception as e:
        print(f"delete old predictions failed: {e}")
    
    try:
        result = client.table("tom_predictions").insert(records).execute()
    except Exception as e:
        if "score" not in str(e).lower():
            raise
        for rec in records:
            rec.pop("score", None)
        result = client.table("tom_predictions").insert(records).execute()
    
    return len(result.data) if result.data else 0


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _outcome_record(pred: dict, next_date, *,
                    next_open=None, next_high=None, next_low=None,
                    next_close=None, next_volume=None) -> dict | None:
    """
    Score one pick against the next session.

    The number that matters is next-session high vs scan price — that is
    how far the name ran, not whether it closed through the 20-day trigger.
    """
    scan = _f(pred.get("price_at_scan"))
    high = _f(next_high)
    close = _f(next_close)
    if high is None:
        high = close
    if high is None and scan is None:
        return None

    trigger = _f(pred.get("trigger"))
    broke_out = bool(high is not None and trigger and high >= trigger)
    gain_from_trigger = (
        (close / trigger - 1.0) if close and trigger and trigger > 0 else None
    )
    # Stored as gain_from_close for the existing column; computed from HIGH.
    gain_from_high = (high / scan - 1.0) if high and scan and scan > 0 else None

    vol = next_volume
    try:
        vol = int(vol) if vol is not None and pd.notna(vol) else None
    except (TypeError, ValueError):
        vol = None

    return {
        "prediction_id": pred["id"],
        "scan_date": str(pred["scan_date"])[:10],
        "symbol": pred["symbol"],
        "next_date": str(next_date)[:10],
        "next_open": _clean(next_open),
        "next_high": _clean(high),
        "next_low": _clean(next_low),
        "next_close": _clean(close),
        "next_volume": vol,
        "broke_out": broke_out,
        "gain_from_trigger": _clean(gain_from_trigger),
        "gain_from_close": _clean(gain_from_high),
    }


def _upsert_outcomes(records: list[dict]) -> int:
    if not records:
        return 0
    result = get_client().table("tom_outcomes").upsert(
        records, on_conflict="scan_date,symbol"
    ).execute()
    return len(result.data) if result.data else 0


def verify_tom_from_quotes(
    scan_date: date | str,
    quotes: pd.DataFrame,
    next_date: date | str | None = None,
) -> int:
    """
    Verify yesterday's picks against today's live/closing quotes.

    Uses the session high vs price_at_scan. `next_date` defaults to today.
    """
    scan_date = str(scan_date)[:10]
    next_date = str(next_date or session_date())[:10]
    preds = get_predictions_on(scan_date)
    if not preds or quotes is None or quotes.empty:
        return 0

    q = quotes.copy()
    q["symbol"] = q["symbol"].astype(str).str.strip().str.upper()
    q = q.drop_duplicates("symbol", keep="last").set_index("symbol")

    records = []
    for p in preds:
        sym = str(p.get("symbol") or "").strip().upper()
        if not sym or sym not in q.index:
            continue
        r = q.loc[sym]
        high = r.get("high")
        ltp = r.get("ltp")
        records.append(_outcome_record(
            p, next_date,
            next_open=r.get("open"),
            next_high=high if pd.notna(high) else ltp,
            next_low=r.get("low"),
            next_close=ltp,
            next_volume=r.get("volume"),
        ))
    return _upsert_outcomes([r for r in records if r])


def verify_tom_outcomes(scan_date: date | str, stocks_df: pd.DataFrame) -> int:
    """
    Verify previous day's predictions against the next session in `stocks_df`.
    Returns number of outcomes recorded.
    """
    scan_date = str(scan_date)[:10]
    preds = get_predictions_on(scan_date)
    if not preds or stocks_df is None or stocks_df.empty:
        return 0

    pred_date = pd.Timestamp(scan_date)
    later = stocks_df[stocks_df["date"] > pred_date]
    if later.empty:
        return 0

    next_date = later["date"].min()
    next_day = stocks_df[stocks_df["date"] == next_date]

    records = []
    for p in preds:
        row = next_day[next_day["symbol"] == p["symbol"]]
        if row.empty:
            continue
        r = row.iloc[0]
        close = r["adj"] if "adj" in r.index and pd.notna(r["adj"]) else r.get("close")
        high = (
            r["adj_high"] if "adj_high" in r.index and pd.notna(r["adj_high"])
            else r.get("high")
        )
        low = (
            r["adj_low"] if "adj_low" in r.index and pd.notna(r["adj_low"])
            else r.get("low")
        )
        records.append(_outcome_record(
            p, next_date,
            next_open=r.get("open"),
            next_high=high,
            next_low=low,
            next_close=close,
            next_volume=r.get("volume"),
        ))
    return _upsert_outcomes([r for r in records if r])


def get_prediction_dates() -> list[str]:
    """Distinct scan dates, newest first."""
    client = get_client()
    result = (client.table("tom_predictions")
              .select("scan_date")
              .order("scan_date", desc=True)
              .limit(500)
              .execute())
    seen, out = set(), []
    for row in (result.data or []):
        d = str(row.get("scan_date") or "")[:10]
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def get_predictions_on(scan_date: date | str) -> list[dict]:
    scan_date = str(scan_date)[:10]
    client = get_client()
    result = (client.table("tom_predictions")
              .select("*")
              .eq("scan_date", scan_date)
              .execute())
    return _with_score(result.data or [])


def latest_tom_scan_date() -> str | None:
    """Today's saved list if it exists, else the most recent scan date."""
    dates = get_prediction_dates()
    if not dates:
        return None
    today = session_date().isoformat()
    if today in dates:
        return today
    return dates[0]


def tom_frame_from_predictions(preds: list[dict]) -> pd.DataFrame:
    """Shape saved rows into the For Tom table the UI already knows."""
    if not preds:
        return pd.DataFrame()
    rows = []
    for p in preds:
        rows.append({
            "symbol": p.get("symbol"),
            "sector": p.get("sector"),
            "kind": p.get("kind"),
            "ltp": p.get("price_at_scan"),
            "trigger": p.get("trigger"),
            "to_trigger": p.get("to_trigger"),
            "vol_expand": p.get("vol_expand"),
            "vol_ratio": p.get("vol_ratio"),
            "pchange": p.get("pchange"),
            "pos_hi": p.get("pos_hi"),
            "rsi": p.get("rsi"),
            "cmf": p.get("cmf"),
            "why": p.get("why"),
            "score": prediction_score(p),
            "scan_date": str(p.get("scan_date") or "")[:10],
            "created_at": p.get("created_at"),
        })
    return pd.DataFrame(rows)


def get_predictions_with_outcomes(
    days: int = 30,
    kind: str | None = None,
    scan_date: date | str | None = None,
) -> list[dict]:
    """
    Predictions with outcomes. Pass `scan_date` to get one day's list.
    """
    client = get_client()
    
    query = client.table("tom_predictions").select(
        "*, tom_outcomes(*)"
    ).order("scan_date", desc=True)
    
    if scan_date:
        query = query.eq("scan_date", str(scan_date)[:10])
    else:
        query = query.limit(500)
    
    if kind:
        query = query.eq("kind", kind)
    
    result = query.execute()
    return _with_score(result.data if result.data else [])


def get_track_record_stats(
    days: int = 30,
    kind: str | None = None,
    scan_date: date | str | None = None,
    preds: list[dict] | None = None,
) -> dict:
    """
    Aggregate stats. Same filters as the table when `scan_date` / `preds` given.
    """
    if preds is None:
        preds = get_predictions_with_outcomes(
            days=days, kind=kind, scan_date=scan_date,
        )
    
    empty = {
        "total": 0, "verified": 0, "broke_out": 0,
        "hit_rate": None, "avg_gain": None, "avg_close": None, "by_kind": {},
    }
    if not preds:
        return empty
    
    total = len(preds)
    verified = sum(1 for p in preds if p.get("tom_outcomes"))
    broke_out = sum(
        1 for p in preds
        if p.get("tom_outcomes") and any(o.get("broke_out") for o in p["tom_outcomes"])
    )
    
    def _reached(pred, outcome) -> float | None:
        high = _f((outcome or {}).get("next_high"))
        scan = _f(pred.get("price_at_scan"))
        if high is not None and scan and scan > 0:
            return high / scan - 1.0
        return _f((outcome or {}).get("gain_from_close"))

    def _close_pct(pred, outcome) -> float | None:
        close = _f((outcome or {}).get("next_close"))
        scan = _f(pred.get("price_at_scan"))
        if close is not None and scan and scan > 0:
            return close / scan - 1.0
        return None

    # By kind
    by_kind = {}
    for p in preds:
        k = p.get("kind", "unknown")
        if k not in by_kind:
            by_kind[k] = {
                "total": 0, "verified": 0, "broke_out": 0,
                "gains": [], "closes": [],
            }
        by_kind[k]["total"] += 1
        
        outcomes = p.get("tom_outcomes") or []
        if outcomes:
            by_kind[k]["verified"] += 1
            for o in outcomes:
                if o.get("broke_out"):
                    by_kind[k]["broke_out"] += 1
                g = _reached(p, o)
                if g is not None:
                    by_kind[k]["gains"].append(g)
                c = _close_pct(p, o)
                if c is not None:
                    by_kind[k]["closes"].append(c)
    
    # Calculate avg gains
    all_gains = []
    all_closes = []
    for k, v in by_kind.items():
        gains = v.pop("gains", [])
        closes = v.pop("closes", [])
        all_gains.extend(gains)
        all_closes.extend(closes)
        v["avg_gain"] = sum(gains) / len(gains) if gains else None
        v["avg_close"] = sum(closes) / len(closes) if closes else None
        v["hit_rate"] = v["broke_out"] / v["verified"] if v["verified"] > 0 else None
    
    return {
        "total": total,
        "verified": verified,
        "broke_out": broke_out,
        "hit_rate": broke_out / verified if verified > 0 else None,
        "avg_gain": sum(all_gains) / len(all_gains) if all_gains else None,
        "avg_close": sum(all_closes) / len(all_closes) if all_closes else None,
        "by_kind": by_kind,
    }


# ---------------------------------------------------------------------------
# Daily Cache
# ---------------------------------------------------------------------------

def save_daily_cache(
    cache_date: date | str,
    scan_rows: pd.DataFrame | None = None,
    coil_rows: pd.DataFrame | None = None,
    buys: pd.DataFrame | None = None,
    breakouts: pd.DataFrame | None = None,
    tom: pd.DataFrame | None = None,
    n_stocks: int = 0,
    n_sectors: int = 0,
) -> bool:
    """Save daily scan results to cache."""
    cache_date = str(cache_date)[:10]
    client = get_client()
    
    def df_to_json(df):
        if df is None or df.empty:
            return None
        return [_row_dict(r) for _, r in df.iterrows()]
    
    record = {
        "cache_date": cache_date,
        "scan_rows": df_to_json(scan_rows),
        "coil_rows": df_to_json(coil_rows),
        "buys": df_to_json(buys),
        "breakouts": df_to_json(breakouts),
        "tom": df_to_json(tom),
        "n_stocks": n_stocks,
        "n_sectors": n_sectors,
    }
    
    result = client.table("daily_cache").upsert(
        record,
        on_conflict="cache_date"
    ).execute()
    
    return bool(result.data)


def load_daily_cache(cache_date: date | str) -> dict | None:
    """Load cached scan results for a date."""
    cache_date = str(cache_date)[:10]
    client = get_client()
    
    result = client.table("daily_cache").select("*").eq(
        "cache_date", cache_date
    ).single().execute()
    
    return result.data if result.data else None


def get_cached_dates(limit: int = 30) -> list[str]:
    """Get list of dates with cached data."""
    client = get_client()
    
    result = client.table("daily_cache").select("cache_date").order(
        "cache_date", desc=True
    ).limit(limit).execute()
    
    return [r["cache_date"] for r in result.data] if result.data else []


# ---------------------------------------------------------------------------
# Sector Scans (Section 1: Post-Market Sector Lookouts)
# ---------------------------------------------------------------------------

def save_sector_scans(
    scan_date: date | str,
    scan_rows: pd.DataFrame,
    panel: pd.DataFrame | None = None,
) -> int:
    """
    Save sector classifications with shape reports to Supabase.
    
    `scan_rows` = output of scan.classify()
    `panel` = sector panel for computing shape_report per sector
    
    Returns number of rows saved.
    """
    if scan_rows is None or scan_rows.empty:
        return 0
    
    scan_date = str(scan_date)[:10]
    client = get_client()
    
    # Import here to avoid circular import
    import scan as sc
    
    records = []
    for _, r in scan_rows.iterrows():
        sector = r.get("sector")
        
        # Compute shape report if panel is provided
        shape_report = None
        if panel is not None and sector:
            try:
                hist = panel[panel["sector"] == sector]
                if not hist.empty:
                    _, shape = sc.shape_report(hist)
                    shape_report = shape
            except Exception as e:
                print(f"shape_report error for {sector}: {e}")
        
        rec = {
            "scan_date": scan_date,
            "sector": sector,
            "klass": r.get("klass"),
            "t_rel": _clean(r.get("T_rel")),
            "b": _clean(r.get("B")),
            "cmf_rel": _clean(r.get("cmf_rel")),
            "rs": _clean(r.get("rs")),
            "rs_chg_5": _clean(r.get("rs_chg_5")),
            "deliv_quality_rel": _clean(r.get("deliv_quality_rel")),
            "n_stocks": _int(r.get("n_stocks")),
            "n_adv": _clean(r.get("n_adv")),
            "top_share": _clean(r.get("top_share")),
            "cmf": _clean(r.get("cmf")),
            "buy_ready": bool(r.get("buy_ready", False)),
            "note": r.get("note"),
            "shape_report": shape_report,
        }
        records.append(rec)
    
    if not records:
        return 0
    
    # Full replacement for the date
    try:
        client.table("sector_scans").delete().eq("scan_date", scan_date).execute()
    except Exception as e:
        print(f"delete old sector scans failed: {e}")
    
    try:
        result = client.table("sector_scans").insert(records).execute()
        return len(result.data) if result.data else 0
    except Exception as e:
        print(f"save sector scans failed: {e}")
        return 0


def get_sector_scan_dates(limit: int = 60) -> list[str]:
    """Get distinct sector scan dates, newest first."""
    client = get_client()
    result = (client.table("sector_scans")
              .select("scan_date")
              .order("scan_date", desc=True)
              .limit(500)
              .execute())
    seen, out = set(), []
    for row in (result.data or []):
        d = str(row.get("scan_date") or "")[:10]
        if d and d not in seen:
            seen.add(d)
            out.append(d)
            if len(out) >= limit:
                break
    return out


def get_sector_scans_on(scan_date: date | str) -> list[dict]:
    """Get all sector classifications for a specific date."""
    scan_date = str(scan_date)[:10]
    client = get_client()
    result = (client.table("sector_scans")
              .select("*")
              .eq("scan_date", scan_date)
              .execute())
    return result.data or []


def get_sector_history(
    sector: str,
    days: int = 30,
    end_date: date | str | None = None,
) -> list[dict]:
    """
    Get historical scan data for a single sector (for time-series heatmap).
    Returns rows sorted by date ascending.
    """
    client = get_client()
    query = (client.table("sector_scans")
             .select("*")
             .eq("sector", sector)
             .order("scan_date", desc=True)
             .limit(days))
    
    if end_date:
        query = query.lte("scan_date", str(end_date)[:10])
    
    result = query.execute()
    rows = result.data or []
    # Return sorted ascending by date for time-series display
    return sorted(rows, key=lambda x: x.get("scan_date", ""))


def get_all_sectors_latest(scan_date: date | str | None = None) -> list[dict]:
    """
    Get the latest scan for all sectors (for cross-sectional heatmap).
    If scan_date is None, uses the most recent scan date.
    """
    if scan_date is None:
        dates = get_sector_scan_dates(limit=1)
        if not dates:
            return []
        scan_date = dates[0]
    
    return get_sector_scans_on(scan_date)


def get_sector_constituents(
    sector: str,
    scan_date: date | str | None = None,
    stocks_df: pd.DataFrame | None = None,
    top: int = 15,
) -> list[dict]:
    """
    Get top stocks by turnover for a sector on a given date.
    Requires stocks_df (the stock-level data with sector assignments).
    """
    if stocks_df is None or stocks_df.empty:
        return []
    
    if scan_date is None:
        scan_date = stocks_df["date"].max()
    
    scan_date = pd.Timestamp(str(scan_date)[:10])
    day = stocks_df[
        (stocks_df["date"] == scan_date) & 
        (stocks_df["sector"] == sector)
    ]
    
    if day.empty:
        return []
    
    # Sort by turnover descending
    day = day.sort_values("turnover", ascending=False).head(top)
    
    cols = ["symbol", "close", "ret", "turnover", "deliv_pct", 
            "deliv_quality", "cmf", "volume"]
    cols = [c for c in cols if c in day.columns]
    
    return [_row_dict(r) for _, r in day[cols].iterrows()]


# ---------------------------------------------------------------------------
# Coiled Bases (Stock-level pre-breakout candidates)
# ---------------------------------------------------------------------------

def save_coiled_bases(
    scan_date: date | str,
    coil_rows: pd.DataFrame,
) -> int:
    """
    Save coiled base candidates to Supabase.
    
    `coil_rows` = output of stocks.scan() — stocks passing all coil filters,
    ranked by coil score. Called after market close using closing prices.
    
    Returns number of rows saved.
    """
    if coil_rows is None or coil_rows.empty:
        return 0
    
    scan_date = str(scan_date)[:10]
    client = get_client()
    
    records = []
    for _, r in coil_rows.iterrows():
        rec = {
            "scan_date": scan_date,
            "symbol": r.get("symbol"),
            "sector": r.get("sector"),
            "adj": _clean(r.get("adj")),
            "coil": _clean(r.get("coil")),
            "pos_hi": _clean(r.get("pos_hi")),
            "to_trigger": _clean(r.get("to_trigger")),
            "trigger": _clean(r.get("trigger")),
            "rsi": _clean(r.get("rsi")),
            "vol_ratio": _clean(r.get("vol_ratio")),
            "range20": _clean(r.get("range20")),
            "contraction": _clean(r.get("contraction")),
            "cmf": _clean(r.get("cmf")),
            "deliv_quality_rel": _clean(r.get("deliv_quality_rel")),
            "deliv_pct": _clean(r.get("deliv_pct")),
            "base_days": _clean(r.get("base_days")),
            "atr_pct": _clean(r.get("atr_pct")),
            "ext_ema20": _clean(r.get("ext_ema20")),
            # Leaders at rest is the top 20 of this table by mom12_1, so the
            # ranked list needs no table of its own.
            "mom12_1": _clean(r.get("mom12_1")),
            # Everything below is stored so a read is a plain SELECT. The UI
            # renders these and must not re-derive them: rest_rank in
            # particular is assigned by position.leaders_at_rest, and
            # re-sorting by mom12_1 on read would silently disagree with it
            # whenever a name has no 12-1 reading (those rank last, not out).
            "rest_rank": _int(r.get("rest_rank")),
            "coil_days": _int(r.get("coil_days")),
            "recommended": bool(r.get("recommended", False)),
            "episode_days": _int(r.get("episode_days")),
            "episode_new": bool(r.get("episode_new", False)),
        }
        records.append(rec)
    
    if not records:
        return 0
    
    # Full replacement for the date (idempotent)
    try:
        client.table("coiled_bases").delete().eq("scan_date", scan_date).execute()
    except Exception as e:
        print(f"delete old coiled bases failed: {e}")
    
    try:
        result = client.table("coiled_bases").insert(records).execute()
        return len(result.data) if result.data else 0
    except Exception as e:
        print(f"save coiled bases failed: {e}")
        return 0


def get_coiled_bases_dates(limit: int = 60) -> list[str]:
    """Get distinct coiled bases scan dates, newest first."""
    client = get_client()
    result = (client.table("coiled_bases")
              .select("scan_date")
              .order("scan_date", desc=True)
              .limit(500)
              .execute())
    seen, out = set(), []
    for row in (result.data or []):
        d = str(row.get("scan_date") or "")[:10]
        if d and d not in seen:
            seen.add(d)
            out.append(d)
            if len(out) >= limit:
                break
    return out


def get_coiled_bases(scan_date: date | str) -> list[dict]:
    """Get coiled bases for a specific date."""
    scan_date = str(scan_date)[:10]
    client = get_client()
    result = (client.table("coiled_bases")
              .select("*")
              .eq("scan_date", scan_date)
              .order("coil", desc=True)
              .execute())
    return result.data or []


# ---------------------------------------------------------------------------
# Setups (Coiled stocks in actionable sectors)
# ---------------------------------------------------------------------------

def save_setups(
    scan_date: date | str,
    buys: pd.DataFrame,
    verdict_by_sector: dict[str, str] | None = None,
) -> int:
    """
    Save buy setups to Supabase.
    
    `buys` = coiled stocks filtered to sectors in setup patterns (CROSSING,
    PULLBACK, CROSSING_UNVERIFIED). This is the intersection of stock-level
    coil filters and sector-level classification.
    
    Called together with save_sector_scans since setups depend on sector state.
    
    Returns number of rows saved.
    """
    if buys is None or buys.empty:
        return 0
    
    verdict_by_sector = verdict_by_sector or {}
    scan_date = str(scan_date)[:10]
    client = get_client()
    
    records = []
    for _, r in buys.iterrows():
        sector = r.get("sector")
        rec = {
            "scan_date": scan_date,
            "symbol": r.get("symbol"),
            "sector": sector,
            "sector_klass": r.get("sector_klass"),
            "verdict": verdict_by_sector.get(sector, ""),
            "why": r.get("why"),
            "adj": _clean(r.get("adj")),
            "trigger": _clean(r.get("trigger")),
            "coil": _clean(r.get("coil")),
            "to_trigger": _clean(r.get("to_trigger")),
            "pos_hi": _clean(r.get("pos_hi")),
            "rsi": _clean(r.get("rsi")),
            "vol_ratio": _clean(r.get("vol_ratio")),
            "range20": _clean(r.get("range20")),
            "cmf": _clean(r.get("cmf")),
            "deliv_quality_rel": _clean(r.get("deliv_quality_rel")),
            "base_days": _clean(r.get("base_days")),
            "recommended": bool(r.get("recommended", False)),
            # Bridged episode age, so the UI does not need the panel to tell
            # a genuinely new setup from one that wobbled for a session.
            "episode_days": _int(r.get("episode_days")),
            "episode_new": bool(r.get("episode_new", False)),
        }
        records.append(rec)
    
    if not records:
        return 0
    
    # Full replacement for the date (idempotent)
    try:
        client.table("setups").delete().eq("scan_date", scan_date).execute()
    except Exception as e:
        print(f"delete old setups failed: {e}")
    
    try:
        result = client.table("setups").insert(records).execute()
        return len(result.data) if result.data else 0
    except Exception as e:
        print(f"save setups failed: {e}")
        return 0


def get_setups_dates(limit: int = 60) -> list[str]:
    """Get distinct setups scan dates, newest first."""
    client = get_client()
    result = (client.table("setups")
              .select("scan_date")
              .order("scan_date", desc=True)
              .limit(500)
              .execute())
    seen, out = set(), []
    for row in (result.data or []):
        d = str(row.get("scan_date") or "")[:10]
        if d and d not in seen:
            seen.add(d)
            out.append(d)
            if len(out) >= limit:
                break
    return out


def get_setups(scan_date: date | str) -> list[dict]:
    """Get setups for a specific date."""
    scan_date = str(scan_date)[:10]
    client = get_client()
    result = (client.table("setups")
              .select("*")
              .eq("scan_date", scan_date)
              .order("coil", desc=True)
              .execute())
    return result.data or []


# ---------------------------------------------------------------- episodes

def _date_or_none(v):
    """Supabase wants an ISO date string or a real NULL, never 'NaT'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        t = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(t) else str(t.date())


def save_episodes(eps: pd.DataFrame) -> int:
    """
    Upsert base episodes on (symbol, started_on).

    Upsert rather than replace-by-date: an episode is a span, not a snapshot,
    and the nightly recompute only sees the sessions still inside the panel
    window. Deleting first would drop every episode that has aged out of the
    window but is still worth showing.

    Returns number of rows written.
    """
    if eps is None or eps.empty:
        return 0

    client = get_client()
    records = []
    for _, r in eps.iterrows():
        started = _date_or_none(r.get("started_on"))
        if not r.get("symbol") or not started:
            continue
        records.append({
            "symbol": r.get("symbol"),
            "sector": r.get("sector"),
            "started_on": started,
            "entry_price": _clean(r.get("entry_price")),
            "entry_trigger": _clean(r.get("entry_trigger")),
            "state": r.get("state"),
            "state_since": _date_or_none(r.get("state_since")),
            "reason": r.get("reason"),
            "lost_gates": r.get("lost_gates"),
            "last_seen_on": _date_or_none(r.get("last_seen_on")),
            "last_close": _clean(r.get("last_close")),
            "peak_close": _clean(r.get("peak_close")),
            "coil_sessions": _int(r.get("coil_sessions")),
            "age": _int(r.get("age")),
            "gap": _int(r.get("gap")),
            "below": _int(r.get("below")),
            "mom12_1": _clean(r.get("mom12_1")),
            "triggered_on": _date_or_none(r.get("triggered_on")),
            "trigger_age": _int(r.get("trigger_age")),
            "trigger_vol": (None if r.get("trigger_vol") is None
                            else bool(r.get("trigger_vol"))),
            "resolved_on": _date_or_none(r.get("resolved_on")),
        })

    if not records:
        return 0

    saved = 0
    for i in range(0, len(records), 500):
        chunk = records[i:i + 500]
        try:
            result = (client.table("base_episodes")
                      .upsert(chunk, on_conflict="symbol,started_on")
                      .execute())
            saved += len(result.data) if result.data else 0
        except Exception as e:
            print(f"save episodes failed: {e}")
    return saved


# ------------------------------------------------------- the whole read path

# Leaders at rest is the top of the coil pool by 12-1 momentum. Kept here so
# the compatibility path below cannot drift from position.REST_TOP_N.
REST_TOP_N = 20


def _latest_date(table: str, col: str = "scan_date") -> str | None:
    """Most recent date present in `table`, or None if it is empty."""
    client = get_client()
    try:
        r = (client.table(table).select(col)
             .order(col, desc=True).limit(1).execute())
        rows = r.data or []
        return str(rows[0][col])[:10] if rows else None
    except Exception as e:
        print(f"latest date for {table} failed: {e}")
        return None


def get_lists() -> dict:
    """
    Everything the dashboard shows, read straight out of Supabase.

    This is the whole read path. The scans are computed once by the
    post-market cron and written here, so serving the UI is a SELECT and
    nothing in the request path depends on the in-memory panel — which means
    a restarted process serves the same data immediately instead of showing
    an empty app until it has re-fetched a year of bhavcopies.

    Each table is read at its own latest date rather than one shared date.
    They are written together so those normally agree, but if a sector save
    fails the coil list should still render rather than the whole page
    blanking out on a date that one table happens to be missing.

    Which means a list can legitimately lag the session being shown: a night
    with no qualifying setups saves no setup rows, so the newest ones are
    from an earlier session. Their date is returned alongside them so that
    reads as "none tonight" rather than as tonight's list -- acting on a
    four-day-old setup believing it is current is the one failure here that
    costs money.
    """
    scan_date = _latest_date("sector_scans")
    setup_date = _latest_date("setups")
    coil_date = _latest_date("coiled_bases")

    scan = get_all_sectors_latest(scan_date) if scan_date else []
    setups = get_setups(setup_date) if setup_date else []
    coils = get_coiled_bases(coil_date) if coil_date else []

    ranked = [r for r in coils if r.get("rest_rank") is not None]
    if ranked:
        ranked.sort(key=lambda r: r["rest_rank"])
    else:
        # Compatibility path for rows written before rest_rank existed. Same
        # rule position.leaders_at_rest applies: order by 12-1 momentum and
        # keep the top N, with no reading sorting last rather than dropping
        # out. Rows saved by any current run carry rest_rank and skip this.
        ranked = sorted(
            coils,
            key=lambda r: (r.get("mom12_1") is None, -(r.get("mom12_1") or 0.0)),
        )[:REST_TOP_N]

    session = max([d for d in (coil_date, scan_date, setup_date) if d],
                  default=None)
    return {
        "as_of": coil_date or scan_date or setup_date,
        "scan_date": scan_date,
        "scan": scan,
        "buys": setups,
        "coil": coils,
        "rest": ranked,
        # Per-list dates, so a list that lags the session can say so.
        "dates": {"scan": scan_date, "setups": setup_date, "coil": coil_date},
        "session": session,
        "stale": [name for name, d in (("scan", scan_date),
                                       ("setups", setup_date),
                                       ("coil", coil_date))
                  if d and session and d < session],
    }


def get_open_episodes(retain_sessions: int = 5, limit: int = 400) -> list[dict]:
    """
    Episodes worth putting in front of someone tonight.

    Open bases plus anything that resolved in the last few sessions. The full
    table keeps every episode ever tracked, which is the right thing for a
    record but the wrong thing for a screen: reading twenty sessions of
    resolved history back meant several hundred rows, almost all of them
    closed and none of them news.
    """
    client = get_client()
    cutoff = (session_date() - timedelta(days=int(retain_sessions * 1.6))).isoformat()
    try:
        r = (client.table("base_episodes")
             .select("*")
             .or_(f"resolved_on.is.null,resolved_on.gte.{cutoff}")
             .order("state_since", desc=True)
             .limit(limit)
             .execute())
        return r.data or []
    except Exception as e:
        print(f"get open episodes failed: {e}")
        return []


def latest_saved_session() -> str | None:
    """
    The most recent session both core tables already hold.

    Used to decide whether a post-market run has anything to write. `setups`
    is deliberately excluded: a session with no qualifying setups saves zero
    rows, so its latest date legitimately lags and would make every evening
    look outstanding forever.
    """
    dates = [_latest_date("sector_scans"), _latest_date("coiled_bases")]
    if any(d is None for d in dates):
        return None
    return min(dates)


def delete_session(scan_date: date | str) -> dict[str, int]:
    """
    Remove every saved row for one date.

    For clearing a date the market never traded. NSE serves the previous
    session's bhavcopy on a holiday rather than returning nothing, so before
    fetch.py learned to check the file's own trade date it was possible to
    publish a full scan under a non-session.
    """
    scan_date = str(scan_date)[:10]
    client = get_client()
    out = {}
    for t in ("sector_scans", "setups", "coiled_bases"):
        try:
            before = (client.table(t).select("*", count="exact")
                      .eq("scan_date", scan_date).execute()).count or 0
            client.table(t).delete().eq("scan_date", scan_date).execute()
            out[t] = before
        except Exception as e:
            print(f"delete {t} for {scan_date} failed: {e}")
            out[t] = -1
    for t, col in (("base_episodes", "started_on"),):
        try:
            before = (client.table(t).select("*", count="exact")
                      .eq(col, scan_date).execute()).count or 0
            client.table(t).delete().eq(col, scan_date).execute()
            out[t] = before
        except Exception as e:
            print(f"delete {t} for {scan_date} failed: {e}")
            out[t] = -1
    return out


def get_published_coils(limit: int = 20000) -> dict[str, list[str]]:
    """
    Every coil list ever published, as {scan_date: [symbols]}.

    This is what seeds episode tracking. Using the saved lists rather than
    re-running the gates over the price panel keeps the tracked set to names
    that were actually put in front of someone: on the first run that is one
    day and ~80 names, and it grows by the new entrants each evening instead
    of arriving as several hundred rows of history nobody ever saw.
    """
    client = get_client()
    try:
        r = (client.table("coiled_bases")
             .select("scan_date,symbol")
             .order("scan_date", desc=True)
             .limit(limit)
             .execute())
    except Exception as e:
        print(f"get published coils failed: {e}")
        return {}

    out: dict[str, list[str]] = {}
    for row in (r.data or []):
        d = str(row.get("scan_date") or "")[:10]
        s = row.get("symbol")
        if d and s:
            out.setdefault(d, []).append(s)
    return out
