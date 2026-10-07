"""
Upper-circuit carry: a paper-trade tracker for the one short-term pattern
eval_shortterm.py found worth building on.

THE PATTERN  A stock that closes on its upper price band (5/10/20%) reached
             +4% above that close the next session 74% of the time over 261
             sessions, and a buy-at-close / sell-at-gap-or-+4%-or-close trade
             averaged +2.4% to +3.6% gross. Off the band, closing at the high
             does nothing.

THE UNKNOWN  Whether the buy fills. A stock pinned at its upper circuit has a
             queue of buyers and often no sellers; the bhavcopy cannot say
             whether an order at that price would have been hit. That is what
             this tracker exists to measure before any of it reaches the app.

Flow, one row per candidate per session in the Supabase `carry_log` table
(schema in db.py):

    snap   ~15:20-15:28 IST. Groww live prices for the liquid universe, then a
           per-symbol quote for names up >= 4.5% at their high: circuit limit,
           total buy / sell quantity. `total_sell_quantity == 0` at the
           circuit means nothing is on offer -- the order would sit in the
           queue. Circuit limits roll to the next session after the close, so
           this only works while the market is open.
    eod    Same candidates rebuilt from a bhavcopy, with no order book. Used to
           backfill the log so scoring can start immediately; rows are tagged
           source=eod and never count toward the fill question.
    score  After ~18:30 IST once the next session's bhavcopy is out. Fills in
           the official entry close and the next session's outcome.

Re-running `snap` or `eod` for the same date replaces that date's rows of the
same source.
"""

from __future__ import annotations

from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import db
import fetch

EVENT = 0.04
BANDS = (0.05, 0.10, 0.20)
MIN_PRICE = 20.0
MIN_TURN_LACS = 100.0
MIN_HIST = 60
PRE_PCHANGE = 4.5        # % -- below this no 5/10/20% band is reachable
AT_HIGH = 0.999
COST = 0.0025            # delivery round trip, for the net column in report

# Intraday thresholds: how close to circuit to start tracking
# For each band: (heating_min, approaching_min, at_circuit_min)
# e.g., 5% band: heating at 2.5%+, approaching at 4%+, at_circuit at 4.9%+
INTRADAY_THRESHOLDS = {
    0.05: {"heating": 0.025, "approaching": 0.040, "at_circuit": 0.049},
    0.10: {"heating": 0.060, "approaching": 0.085, "at_circuit": 0.098},
    0.20: {"heating": 0.140, "approaching": 0.175, "at_circuit": 0.195},
}

# Expected volume fraction by hour (cumulative). Used to normalize vol_ratio.
# Market: 9:15-15:30 = 6.25 hours. Volume is front-loaded.
VOLUME_BY_HOUR = {
    10: 0.25,   # 10:00 - ~25% of day's volume done
    11: 0.40,   # 11:00 - ~40%
    12: 0.52,   # 12:00 - ~52%
    13: 0.62,   # 13:00 - ~62%
    14: 0.75,   # 14:00 - ~75%
    15: 0.92,   # 15:00 - ~92%
}

COLUMNS = [
    "as_of", "source", "logged_at", "symbol", "sector", "prev_close", "ltp",
    "high", "pchange", "upper_circuit", "band", "at_circuit",
    "total_buy_qty", "total_sell_qty", "fillable", "volume", "med_turn20",
    "vol_ratio",
    # filled by score()
    "entry_close", "closed_at_circuit", "nx_date", "nx_open", "nx_high",
    "nx_low", "nx_close", "gap", "reach", "hit4", "btst",
]

INTRADAY_COLUMNS = [
    "as_of", "scan_time", "logged_at", "symbol", "sector", "prev_close", "ltp",
    "high", "low", "pchange", "upper_circuit", "lower_circuit", "band",
    "distance_to_circuit", "status", "total_buy_qty", "total_sell_qty",
    "fillable", "volume", "vol_ratio", "med_turn20",
]


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

def _hist_files(end: date, n: int) -> pd.DataFrame:
    """The last `n` sessions up to and including `end`, from cache or NSE."""
    return fetch.load_history(end, n, verbose=False)


def universe(end: date) -> pd.DataFrame:
    """Liquid EQ companies as of the last session on or before `end`."""
    raw = _hist_files(end, MIN_HIST + 5)
    # History counts BE sessions too: names move between EQ and trade-for-trade
    # often, and an EQ-only count drops them for weeks after they move back --
    # which is exactly when they tend to run into the upper band.
    df = raw[raw["series"].str.upper().isin(["EQ", "BE"])]
    on_eq = set(df.loc[(df["date"] == df["date"].max())
                       & (df["series"].str.upper() == "EQ"), "symbol"])
    df = df[df["symbol"].isin(on_eq)]
    eq_list = fetch.CACHE_DIR / "equity_list.csv"
    if eq_list.exists():
        names = set(pd.read_csv(eq_list)["symbol"].astype(str).str.strip())
        df = df[df["symbol"].isin(names)]
    df = df.sort_values(["symbol", "date"])
    g = df.groupby("symbol")
    out = pd.DataFrame({
        "n_hist": g.size(),
        "med_turn20": g["turnover"].apply(lambda s: s.tail(20).median()),
        "last_close": g["close"].last(),
    })
    # load_history stops at MIN_HIST + 5 sessions, so n_hist is capped there;
    # the check is "has a full window", not "has 60 sessions in total".
    out = out[(out["n_hist"] >= MIN_HIST)
              & (out["med_turn20"] >= MIN_TURN_LACS)
              & (out["last_close"] >= MIN_PRICE)]
    out = out.reset_index()
    out["sector"] = out["symbol"].map(_sector_map()).fillna("Unknown")
    return out


def _sector_map() -> dict:
    path = fetch.CACHE_DIR / "sector_map.json"
    if not path.exists():
        return {}
    smap = pd.read_json(path, orient="index")
    return smap["basic_industry"].dropna().to_dict()


def _band_of(pct: float) -> float:
    """Nearest price band to a % change; NaN if the move is not on one."""
    for b in BANDS:
        if abs(pct - b) < 0.003:
            return b
    return np.nan


# --------------------------------------------------------------------------
# Live snapshot
# --------------------------------------------------------------------------

def _depth_one(groww, symbol: str) -> dict | None:
    try:
        q = groww.get_quote(exchange=groww.EXCHANGE_NSE,
                            segment=groww.SEGMENT_CASH, trading_symbol=symbol)
    except Exception as exc:
        print(f"  quote {symbol}: {exc}")
        return None
    if not isinstance(q, dict):
        return None
    return {
        "symbol": symbol,
        "upper_circuit": q.get("upper_circuit_limit"),
        "total_buy_qty": q.get("total_buy_quantity"),
        "total_sell_qty": q.get("total_sell_quantity"),
        "volume": q.get("volume"),
        "q_ltp": q.get("last_price"),
    }


def snapshot(on_progress=None) -> pd.DataFrame:
    """Today's at-circuit candidates with the order book behind them."""
    today = db.now_ist().date()
    uni = universe(today - timedelta(days=1))
    print(f"  universe: {len(uni):,} liquid names")
    live = fetch.live_quotes_groww(uni["symbol"].tolist(), on_progress=on_progress)
    if live.empty:
        raise RuntimeError("Groww returned no prices; is the market open?")
    m = live.merge(uni[["symbol", "sector", "med_turn20"]], on="symbol")
    pre = m[(m["pchange"] >= PRE_PCHANGE) & (m["ltp"] >= m["high"] * AT_HIGH)].copy()
    print(f"  {len(pre)} names up >= {PRE_PCHANGE}% and at their high; reading order books…")
    if pre.empty:
        return pd.DataFrame(columns=COLUMNS)

    groww = fetch._get_groww_client()
    with ThreadPoolExecutor(max_workers=fetch.GROWW_QUOTE_WORKERS) as pool:
        depth = [d for d in pool.map(lambda s: _depth_one(groww, s), pre["symbol"]) if d]
    pre = pre.drop(columns=["volume", "turnover", "time"], errors="ignore")
    pre = pre.merge(pd.DataFrame(depth), on="symbol", how="left")
    for c in ("upper_circuit", "total_buy_qty", "total_sell_qty", "volume", "q_ltp"):
        pre[c] = pd.to_numeric(pre[c], errors="coerce")
    # The per-symbol quote is a few seconds fresher than the batch LTP.
    pre["ltp"] = pre["q_ltp"].fillna(pre["ltp"])
    pre["at_circuit"] = pre["ltp"] >= pre["upper_circuit"] * AT_HIGH
    pre["band"] = (pre["upper_circuit"] / pre["prev_close"] - 1).round(2)
    pre["fillable"] = pre["total_sell_qty"].fillna(0) > 0

    # Turnover ratio: approximate today's turnover (vol × ltp) vs 20d median
    pre["vol_ratio"] = np.where(
        pre["med_turn20"] > 0,
        (pre["volume"] * pre["ltp"] / 1e5) / pre["med_turn20"],
        np.nan)
    pre["as_of"] = today.isoformat()
    pre["source"] = "live"
    pre["logged_at"] = db.now_ist().isoformat(timespec="seconds")
    return pre.reindex(columns=COLUMNS)


# --------------------------------------------------------------------------
# Intraday scanner - runs every 30 minutes during market hours
# --------------------------------------------------------------------------

def _determine_band(upper_circuit: float, prev_close: float) -> float | None:
    """Determine which price band (5%, 10%, 20%) applies based on circuit limit."""
    if not prev_close or prev_close <= 0:
        return None
    ratio = upper_circuit / prev_close - 1
    for band in BANDS:
        if abs(ratio - band) < 0.005:  # within 0.5% of the band
            return band
    return None


def _classify_status(pchange_pct: float, band: float, at_circuit: bool, has_sellers: bool) -> str:
    """
    Classify a stock's status based on how close it is to circuit.
    
    Returns: 'heating', 'approaching', 'at_circuit', or 'locked'
    """
    if band not in INTRADAY_THRESHOLDS:
        return "heating"
    
    thresholds = INTRADAY_THRESHOLDS[band]
    pchange = pchange_pct / 100.0  # convert from % to fraction
    
    if at_circuit:
        return "locked" if not has_sellers else "at_circuit"
    elif pchange >= thresholds["at_circuit"]:
        return "at_circuit"
    elif pchange >= thresholds["approaching"]:
        return "approaching"
    elif pchange >= thresholds["heating"]:
        return "heating"
    return "heating"


def _expected_volume_fraction(hour: int) -> float:
    """What fraction of daily volume should have traded by this hour."""
    if hour <= 10:
        return VOLUME_BY_HOUR.get(10, 0.25)
    if hour >= 15:
        return VOLUME_BY_HOUR.get(15, 0.92)
    return VOLUME_BY_HOUR.get(hour, 0.5)


def intraday_scan(scan_time: str | None = None, on_progress=None) -> pd.DataFrame:
    """
    Scan for stocks approaching or at their upper circuit.
    
    This is the main intraday scanner. Call it every 30 minutes during market hours.
    It identifies stocks that are:
    - heating: up significantly, showing momentum toward circuit
    - approaching: very close to circuit, may still be buyable
    - at_circuit: at the limit, check if sellers present
    - locked: at circuit with no sellers (queue only)
    
    Args:
        scan_time: Override scan time label (default: current time rounded to 30 min)
        on_progress: Callback for progress updates
    
    Returns:
        DataFrame of candidates with their status and order book info
    """
    now = db.now_ist()
    today = now.date()
    
    # Determine scan time label (round to nearest 30 min)
    if scan_time is None:
        minute = 30 if now.minute >= 15 else 0
        if now.minute >= 45:
            hour = now.hour + 1
            minute = 0
        else:
            hour = now.hour
        scan_time = f"{hour:02d}:{minute:02d}"
    
    print(f"[intraday] scan at {scan_time} IST")
    
    # Get universe
    uni = universe(today - timedelta(days=1))
    print(f"  universe: {len(uni):,} liquid names")
    
    # Fetch live prices
    live = fetch.live_quotes_groww(uni["symbol"].tolist(), on_progress=on_progress)
    if live.empty:
        print("  no prices returned")
        return pd.DataFrame(columns=INTRADAY_COLUMNS)
    
    # Merge with universe data
    m = live.merge(uni[["symbol", "sector", "med_turn20"]], on="symbol")
    
    # Filter to stocks showing momentum: up at least 2% and near day's high
    MIN_GAIN = 0.02  # 2% minimum to consider
    m = m[(m["pchange"] >= MIN_GAIN * 100) & (m["ltp"] >= m["high"] * 0.995)].copy()
    print(f"  {len(m)} names up >= 2% near day's high")
    
    if m.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS)
    
    # Get detailed quotes with circuit limits and order book
    groww = fetch._get_groww_client()
    
    def get_full_quote(sym):
        try:
            q = groww.get_quote(exchange=groww.EXCHANGE_NSE,
                                segment=groww.SEGMENT_CASH, trading_symbol=sym)
            if not isinstance(q, dict):
                return None
            return {
                "symbol": sym,
                "upper_circuit": q.get("upper_circuit_limit"),
                "lower_circuit": q.get("lower_circuit_limit"),
                "total_buy_qty": q.get("total_buy_quantity"),
                "total_sell_qty": q.get("total_sell_quantity"),
                "volume": q.get("volume"),
                "q_ltp": q.get("last_price"),
                "q_high": q.get("ohlc", {}).get("high"),
                "q_low": q.get("ohlc", {}).get("low"),
            }
        except Exception as exc:
            print(f"  quote {sym}: {exc}")
            return None
    
    with ThreadPoolExecutor(max_workers=fetch.GROWW_QUOTE_WORKERS) as pool:
        quotes = [q for q in pool.map(get_full_quote, m["symbol"]) if q]
    
    if not quotes:
        print("  no detailed quotes returned")
        return pd.DataFrame(columns=INTRADAY_COLUMNS)
    
    # Merge detailed quotes
    m = m.drop(columns=["volume", "turnover", "time"], errors="ignore")
    m = m.merge(pd.DataFrame(quotes), on="symbol", how="inner")
    
    # Convert numeric columns
    for c in ("upper_circuit", "lower_circuit", "total_buy_qty", "total_sell_qty", 
              "volume", "q_ltp", "q_high", "q_low"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    
    # Use fresher quote data where available
    m["ltp"] = m["q_ltp"].fillna(m["ltp"])
    m["high"] = m["q_high"].fillna(m["high"])
    m["low"] = m["q_low"].fillna(m["low"])
    
    # Determine band and calculate distance to circuit
    m["band"] = m.apply(
        lambda r: _determine_band(r["upper_circuit"], r["prev_close"]), axis=1)
    m = m[m["band"].notna()].copy()
    
    if m.empty:
        print("  no stocks with valid circuit bands")
        return pd.DataFrame(columns=INTRADAY_COLUMNS)
    
    # Calculate distance to circuit (0 = at circuit, positive = below circuit)
    m["distance_to_circuit"] = m["upper_circuit"] / m["ltp"] - 1
    
    # Filter to stocks reasonably close to circuit (within 3% of upper limit)
    MAX_DISTANCE = 0.03
    m = m[m["distance_to_circuit"] <= MAX_DISTANCE].copy()
    print(f"  {len(m)} stocks within {MAX_DISTANCE:.0%} of their circuit")
    
    if m.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS)
    
    # Determine status
    m["at_circuit"] = m["ltp"] >= m["upper_circuit"] * AT_HIGH
    m["fillable"] = m["total_sell_qty"].fillna(0) > 0
    m["status"] = m.apply(
        lambda r: _classify_status(r["pchange"], r["band"], r["at_circuit"], r["fillable"]),
        axis=1)
    
    # Calculate volume ratio (current volume vs expected at this time of day)
    # This helps identify unusual activity
    hour = now.hour
    expected_frac = _expected_volume_fraction(hour)
    # Estimate full-day volume based on what we've seen so far
    m["vol_ratio"] = (m["volume"] / expected_frac) / (m["med_turn20"] * 100000)  # convert lacs to shares approx
    
    # Build output
    m["as_of"] = today.isoformat()
    m["scan_time"] = scan_time
    m["logged_at"] = now.isoformat(timespec="seconds")
    
    out = m.reindex(columns=INTRADAY_COLUMNS)
    
    # Sort by most actionable first
    status_order = {"approaching": 0, "at_circuit": 1, "heating": 2, "locked": 3}
    out["_sort"] = out["status"].map(status_order).fillna(4)
    out = out.sort_values(["_sort", "distance_to_circuit"]).drop(columns="_sort")
    
    print(f"  results: {(out['status'] == 'heating').sum()} heating, "
          f"{(out['status'] == 'approaching').sum()} approaching, "
          f"{(out['status'] == 'at_circuit').sum()} at circuit, "
          f"{(out['status'] == 'locked').sum()} locked")
    
    return out.reset_index(drop=True)


def save_intraday_scan(rows: pd.DataFrame) -> int:
    """Save intraday scan results to Supabase."""
    return db.upsert_carry_intraday(rows)


def get_intraday_candidates(as_of: str | None = None) -> pd.DataFrame:
    """Get all intraday scan results for a date."""
    return db.get_carry_intraday(as_of)


def intraday_payload(as_of: str | None = None) -> dict:
    """
    Package intraday data for the API response.
    
    Returns latest scan, progression of each stock through the day,
    and summary stats.
    """
    now = db.now_ist()
    if as_of is None:
        as_of = now.date().isoformat()
    
    df = get_intraday_candidates(as_of)
    if df.empty:
        return {
            "as_of": as_of,
            "scans": [],
            "latest": [],
            "progression": [],
            "summary": {"heating": 0, "approaching": 0, "at_circuit": 0, "locked": 0},
        }
    
    # List of scan times we have
    scans = sorted(df["scan_time"].unique().tolist())
    latest_time = scans[-1] if scans else None
    
    # Latest scan results
    latest = df[df["scan_time"] == latest_time].copy() if latest_time else pd.DataFrame()
    
    # Track progression: how each stock moved through statuses during the day
    def _py(v):
        """Convert numpy types to Python natives for JSON."""
        if pd.isna(v):
            return None
        if isinstance(v, (np.bool_, np.integer)):
            return bool(v) if isinstance(v, np.bool_) else int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    progression = []
    for symbol in df["symbol"].unique():
        sym_df = df[df["symbol"] == symbol].sort_values("scan_time")
        history = []
        for _, row in sym_df.iterrows():
            history.append({
                "time": row["scan_time"],
                "status": row["status"],
                "pchange": _py(row["pchange"]),
                "distance": _py(row["distance_to_circuit"]),
                "fillable": _py(row.get("fillable")),
            })
        
        last = sym_df.iloc[-1]
        progression.append({
            "symbol": symbol,
            "sector": last["sector"],
            "current_status": last["status"],
            "current_price": _py(last["ltp"]),
            "pchange": _py(last["pchange"]),
            "band": _py(last["band"]),
            "distance_to_circuit": _py(last["distance_to_circuit"]),
            "fillable": _py(last.get("fillable")),
            "vol_ratio": _py(last.get("vol_ratio")),
            "med_turn20": _py(last.get("med_turn20")),
            "first_seen": sym_df["scan_time"].min(),
            "times_seen": len(sym_df),
            "history": history,
        })
    
    # Sort by actionability
    status_order = {"approaching": 0, "at_circuit": 1, "heating": 2, "locked": 3}
    progression.sort(key=lambda x: (status_order.get(x["current_status"], 4), 
                                     x.get("distance_to_circuit") or 1))
    
    # Summary counts from latest scan
    summary = {"heating": 0, "approaching": 0, "at_circuit": 0, "locked": 0}
    if not latest.empty:
        for status in summary:
            summary[status] = int((latest["status"] == status).sum())
    
    import json
    return {
        "as_of": as_of,
        "latest_scan": latest_time,
        "scans": scans,
        "latest": json.loads(latest.to_json(orient="records")) if not latest.empty else [],
        "progression": progression,
        "summary": summary,
    }


# --------------------------------------------------------------------------
# EOD backfill
# --------------------------------------------------------------------------

def from_bhavcopy(d: date) -> pd.DataFrame:
    """Candidates for session `d` rebuilt from its bhavcopy. No order book."""
    day = fetch.fetch_bhavcopy(d)
    if day is None or day.empty:
        return pd.DataFrame(columns=COLUMNS)
    uni = universe(d - timedelta(days=1))
    m = day.merge(uni[["symbol", "sector", "med_turn20"]], on="symbol")
    m = m[m["series"].str.upper() == "EQ"]
    ret = m["close"] / m["prev_close"] - 1
    m["band"] = ret.map(_band_of)
    m = m[(m["close"] >= m["high"] * AT_HIGH) & m["band"].notna()].copy()
    m["pchange"] = (m["close"] / m["prev_close"] - 1) * 100
    m["ltp"] = m["close"]
    m["upper_circuit"] = m["close"]
    m["at_circuit"] = True
    # Turnover ratio: today's turnover vs 20-day median (both in lacs)
    m["vol_ratio"] = np.where(m["med_turn20"] > 0,
                              m["turnover"] / m["med_turn20"], np.nan)
    m["as_of"] = d.isoformat()
    m["source"] = "eod"
    m["logged_at"] = db.now_ist().isoformat(timespec="seconds")
    return m.reindex(columns=COLUMNS)


# --------------------------------------------------------------------------
# Log
# --------------------------------------------------------------------------

def load_log() -> pd.DataFrame:
    log = db.get_carry_log()
    if log.empty:
        return pd.DataFrame(columns=COLUMNS)
    log = log.reindex(columns=COLUMNS)
    for c in ("prev_close", "ltp", "high", "pchange", "upper_circuit", "band",
              "total_buy_qty", "total_sell_qty", "volume", "med_turn20",
              "vol_ratio", "entry_close", "nx_open", "nx_high", "nx_low",
              "nx_close", "gap", "reach", "btst"):
        log[c] = pd.to_numeric(log[c], errors="coerce")
    return log


def append(rows: pd.DataFrame) -> int:
    """Replace this (as_of, source) slice of the log with `rows`."""
    return db.replace_carry_slice(rows)


def _next_session(d: date, limit: int = 7) -> tuple[date, pd.DataFrame] | None:
    for k in range(1, limit + 1):
        nd = d + timedelta(days=k)
        if nd >= date.today() + timedelta(days=1):
            return None
        if nd.weekday() >= 5:
            continue
        df = fetch.fetch_bhavcopy(nd)
        if df is not None and not df.empty:
            return nd, df
    return None


def score() -> tuple[int, int]:
    """Fill outcomes for every row whose next session is now published."""
    log = load_log()
    if log.empty:
        return 0, 0
    for c in ("closed_at_circuit", "nx_date", "hit4"):
        log[c] = log[c].astype(object)
    todo = log["btst"].isna()
    scored: list = []
    for as_of in sorted(log.loc[todo, "as_of"].astype(str).unique()):
        d = date.fromisoformat(as_of)
        today_bhav = fetch.fetch_bhavcopy(d)
        nxt = _next_session(d)
        if today_bhav is None or nxt is None:
            continue
        nd, nx = nxt
        cur = today_bhav.set_index("symbol")
        nx = nx.set_index("symbol")
        idx = log.index[todo & (log["as_of"].astype(str) == as_of)]
        for i in idx:
            sym = log.at[i, "symbol"]
            if sym not in cur.index or sym not in nx.index:
                continue
            c = cur.loc[sym]
            n = nx.loc[sym]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[0]
            if isinstance(n, pd.DataFrame):
                n = n.iloc[0]
            # The next session's PREV_CLOSE is the official close, adjusted if
            # the name went ex overnight -- the right base for every return.
            base = float(n["prev_close"])
            gap = float(n["open"]) / base - 1
            reach = float(n["high"]) / base - 1
            ret = float(n["close"]) / base - 1
            uc = log.at[i, "upper_circuit"]
            log.at[i, "entry_close"] = float(c["close"])
            log.at[i, "closed_at_circuit"] = (
                bool(float(c["close"]) >= float(uc) * AT_HIGH) if pd.notna(uc) else np.nan)
            log.at[i, "nx_date"] = nd.isoformat()
            log.at[i, "nx_open"] = float(n["open"])
            log.at[i, "nx_high"] = float(n["high"])
            log.at[i, "nx_low"] = float(n["low"])
            log.at[i, "nx_close"] = float(n["close"])
            log.at[i, "gap"] = gap
            log.at[i, "reach"] = reach
            log.at[i, "hit4"] = bool(reach >= EVENT)
            log.at[i, "btst"] = gap if gap >= EVENT else (EVENT if reach >= EVENT else ret)
            scored.append(i)
    db.upsert_carry(log.loc[scored])
    return len(scored), int(log["btst"].isna().sum())


def _dedupe(log: pd.DataFrame) -> pd.DataFrame:
    """One row per (as_of, symbol): the live snapshot wins over the eod rebuild."""
    if log.empty:
        return log
    order = log["source"].map({"live": 0, "eod": 1}).fillna(2)
    return (log.assign(_o=order).sort_values(["as_of", "symbol", "_o"])
               .drop_duplicates(["as_of", "symbol"]).drop(columns="_o"))


def ensure_session(d: date) -> int:
    """
    Rebuild session `d` from its bhavcopy when no live snapshot was taken.

    Called by the post-market cron, so the list for the day exists even if the
    15:22 snapshot was missed. A day that has live rows is left alone.
    """
    log = load_log()
    if not log.empty and ((log["as_of"].astype(str) == d.isoformat())
                          & (log["source"] == "live")).any():
        return 0
    return append(from_bhavcopy(d))


def _records(df: pd.DataFrame) -> list[dict]:
    import json
    return json.loads(df.to_json(orient="records", date_format="iso"))


def payload(history_sessions: int = 60) -> dict:
    """Everything the Circuit carry screen shows, in one response."""
    log = load_log()
    if log.empty:
        return {"as_of": None, "latest": [], "summary": [], "daily": [], "history": []}
    for c in ("at_circuit", "fillable", "hit4"):
        log[c] = log[c].astype(str).str.lower().map(
            {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False})
    log = _dedupe(log[log["at_circuit"] != False])  # noqa: E712  (None = unknown, keep)
    log["as_of"] = log["as_of"].astype(str)

    as_of = log["as_of"].max()
    latest = log[log["as_of"] == as_of].sort_values(
        ["band", "pchange"], ascending=[False, False])

    scored = log[log["btst"].notna()]
    daily = (scored.groupby("as_of")
             .agg(n=("symbol", "size"), hits=("hit4", "sum"),
                  mean_btst=("btst", "mean"), best=("btst", "max"), worst=("btst", "min"),
                  source=("source", lambda s: "live" if (s == "live").any() else "eod"))
             .reset_index().sort_values("as_of", ascending=False)
             .head(history_sessions))
    daily["hit4"] = daily["hits"] / daily["n"]
    keep = set(daily["as_of"])
    history = scored[scored["as_of"].isin(keep)].sort_values(
        ["as_of", "btst"], ascending=[False, False])

    rep = report()
    return {
        "as_of": as_of,
        "source": "live" if (latest["source"] == "live").any() else "eod",
        "logged_at": latest["logged_at"].dropna().max() if len(latest) else None,
        "latest": _records(latest),
        "summary": _records(rep) if not rep.empty else [],
        "daily": _records(daily),
        "history": _records(history),
        "pending": int(log["btst"].isna().sum()),
    }


def report(log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-bucket results. Only at-circuit rows count; live rows split by fill."""
    log = load_log() if log is None else log
    s = log[log["btst"].notna()].copy()
    if s.empty:
        return pd.DataFrame()
    s["hit4"] = s["hit4"].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
    s["at_circuit"] = s["at_circuit"].astype(str).str.lower().isin(["true", "1", "1.0"])
    s["fillable"] = s["fillable"].astype(str).str.lower().isin(["true", "1", "1.0"])
    s = s[s["at_circuit"]]

    def bucket(r):
        if r["source"] == "eod":
            return "eod backfill (fill unknown)"
        return "live: sellers present" if r["fillable"] else "live: no sellers (queue)"

    s["bucket"] = s.apply(bucket, axis=1)
    # Buying at the next open sidesteps the circuit queue entirely. Over the
    # year it only paid on the 20% band (+1.1%), so it is reported per band.
    touched = s["nx_high"] / s["nx_open"] - 1 >= EVENT
    s["open_tr"] = np.where(touched, EVENT, s["nx_close"] / s["nx_open"] - 1)
    band = pd.to_numeric(s["band"], errors="coerce").round(2)
    s["band_bucket"] = "band " + (band * 100).round().astype("Int64").astype(str) + "%"
    rows = []
    groups = ([("all", _dedupe(s))] + list(s.groupby("bucket"))
              + [("all live", s[s["source"] == "live"])]
              + list(s.groupby("band_bucket")))
    for name, g in groups:
        if g.empty:
            continue
        rows.append({
            "bucket": name, "n": len(g), "sessions": g["as_of"].nunique(),
            "hit4": g["hit4"].mean(),
            "gap4": (g["gap"] >= EVENT).mean(),
            "mean_btst": g["btst"].mean(),
            "median_btst": g["btst"].median(),
            "net_mean": g["btst"].mean() - COST,
            "win_rate": (g["btst"] > 0).mean(),
            "open_trade": g["open_tr"].mean(),
            "worst": g["btst"].min(),
        })
    return pd.DataFrame(rows)
