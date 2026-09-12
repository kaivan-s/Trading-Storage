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
