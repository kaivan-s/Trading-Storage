"""
Broader market scanners — informational watchlists that run alongside
the circuit carry scanner.

These are NOT predictive signals. The momentum study across 343,000
name-days showed none of these patterns have a tradeable edge after
costs. They are context: "what's moving in the market right now."

Scanners
--------
1. unusual_volume   — stocks with 2x+ normal turnover and positive momentum
2. big_movers       — stocks up 5%+ from prev close
3. momentum_streaks — 3+ consecutive up days, cumulative gain > 5%
4. breakouts_52w    — stocks at/near 52-week high with volume
5. sector_pulse     — which sectors are hot today
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import db
import fetch

MIN_PRICE = 10.0
MIN_TURN_LACS = 100.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _sector_map() -> dict:
    path = fetch.CACHE_DIR / "sector_map.json"
    if not path.exists():
        return {}
    smap = pd.read_json(path, orient="index")
    return smap["basic_industry"].dropna().to_dict()


_liquid_cache: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame]] = {}


def _liquid_eq(end: date, n_sessions: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick liquid-universe builder for the scanners.

    Memoised per (end, n_sessions): derived from settled bhavcopy, so the
    30-min crons reuse one build instead of reloading history each time.
    """
    key = (end.isoformat(), n_sessions)
    cached = _liquid_cache.get(key)
    if cached is not None:
        return cached
    out = _build_liquid_eq(end, n_sessions)
    if len(_liquid_cache) > 8:
        _liquid_cache.clear()
    _liquid_cache[key] = out
    return out


def _build_liquid_eq(end: date, n_sessions: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = fetch.load_history(end, n_sessions, verbose=False)
    raw = raw[raw["series"].str.upper().isin(["EQ", "BE"])]
    on_eq = set(raw.loc[(raw["date"] == raw["date"].max())
                        & (raw["series"].str.upper() == "EQ"), "symbol"])
    raw = raw[raw["symbol"].isin(on_eq)]

    eq_list = fetch.CACHE_DIR / "equity_list.csv"
    if eq_list.exists():
        names = set(pd.read_csv(eq_list)["symbol"].astype(str).str.strip())
        raw = raw[raw["symbol"].isin(names)]

    raw = raw.sort_values(["symbol", "date"])
    g = raw.groupby("symbol")
    uni = pd.DataFrame({
        "med_turn20": g["turnover"].apply(lambda s: s.tail(20).median()),
        "vol_avg20": g["volume"].apply(lambda s: s.tail(20).mean()),
        "last_close": g["close"].last(),
        "last_high": g["high"].last(),
    })
    uni = uni[(uni["med_turn20"] >= MIN_TURN_LACS) & (uni["last_close"] >= MIN_PRICE)]
    uni = uni.reset_index()
    sectors = _sector_map()
    uni["sector"] = uni["symbol"].map(sectors).fillna("Unknown")
    return uni, raw


# --------------------------------------------------------------------------
# 1. Unusual volume
# --------------------------------------------------------------------------

# The batch price endpoint does not carry traded volume (see fetch.py), so the
# live `volume` column is empty. We fetch it per-symbol for a bounded set of
# advancers only — enough to surface genuine volume spikes without quoting the
# whole universe or tripping the Groww rate limiter.
UNUSUAL_VOL_MAX_QUOTES = 60


def augment_with_volume(live: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """
    Fill `volume` for the top advancers by fetching per-symbol quotes.

    Without this the batch snapshot has `volume = None`, which made
    `unusual_volume` divide by NaN and silently return nothing. We only quote
    advancers above the turnover floor, capped at UNUSUAL_VOL_MAX_QUOTES.
    """
    if live.empty or "pchange" not in live.columns:
        return live
    cand = live.merge(uni[["symbol", "med_turn20"]], on="symbol", how="left")
    cand = cand[(cand["pchange"] > 0) & (cand["med_turn20"] >= MIN_TURN_LACS)]
    if cand.empty:
        return live
    cand = cand.nlargest(UNUSUAL_VOL_MAX_QUOTES, "pchange")
    try:
        groww = fetch._get_groww_client()
        vols = fetch.quote_batch_sequential(
            groww, cand["symbol"].tolist(), {"volume": "volume"},
            label="unusual-vol")
    except Exception as exc:
        print(f"  unusual-vol quote failed: {exc}")
        return live
    if not vols:
        return live
    vdf = pd.DataFrame(vols)
    vdf["volume"] = pd.to_numeric(vdf["volume"], errors="coerce")
    return live.drop(columns=["volume"], errors="ignore").merge(
        vdf[["symbol", "volume"]], on="symbol", how="left")


def unusual_volume(live: pd.DataFrame, uni: pd.DataFrame) -> list[dict]:
    """
    Stocks with 2x+ normal volume and positive price action.

    Args:
        live: snapshot with ltp, pchange and (via augment_with_volume) volume.
        uni: Universe DataFrame with vol_avg20, med_turn20, sector.
    """
    if "volume" not in live.columns:
        return []
    m = live.merge(uni[["symbol", "vol_avg20", "med_turn20", "sector"]], on="symbol")
    # Live volume is cumulative-so-far today vs the full-day 20-session average,
    # so this is a conservative floor: a name already at 2x by now is a real spike.
    m["vol_ratio"] = np.where(
        m["vol_avg20"] > 0, m["volume"] / m["vol_avg20"], np.nan)
    m = m[(m["vol_ratio"] >= 2.0) & (m["pchange"] > 0)].copy()
    m = m.sort_values("vol_ratio", ascending=False)

    return [{
        "symbol": r["symbol"],
        "ltp": float(r["ltp"]),
        "pchange": float(r["pchange"]),
        "vol_ratio": round(float(r["vol_ratio"]), 1),
        "turnover_cr": round(float(r["med_turn20"]) / 100, 1),
        "sector": r["sector"],
    } for _, r in m.head(50).iterrows()]


# --------------------------------------------------------------------------
# 2. Big movers
# --------------------------------------------------------------------------

def big_movers(live: pd.DataFrame, uni: pd.DataFrame,
               threshold: float = 5.0) -> list[dict]:
    """Stocks up `threshold`% or more from previous close."""
    m = live.merge(uni[["symbol", "med_turn20", "sector"]], on="symbol")
    m = m[m["pchange"] >= threshold].copy()
    m = m.sort_values("pchange", ascending=False)

    return [{
        "symbol": r["symbol"],
        "ltp": float(r["ltp"]),
        "pchange": float(r["pchange"]),
        "prev_close": float(r["prev_close"]),
        "high": float(r["high"]),
        "turnover_cr": round(float(r["med_turn20"]) / 100, 1),
        "sector": r["sector"],
    } for _, r in m.head(50).iterrows()]


# --------------------------------------------------------------------------
# 3. Momentum streaks
# --------------------------------------------------------------------------

def momentum_streaks(raw: pd.DataFrame, uni: pd.DataFrame,
                     min_streak: int = 3,
                     min_cum: float = 0.05) -> list[dict]:
    """
    Stocks with `min_streak`+ consecutive up days and cumulative gain >= `min_cum`.
    Uses bhavcopy history (EOD data).
    """
    hist = raw[raw["symbol"].isin(set(uni["symbol"]))].copy()
    hist = hist.sort_values(["symbol", "date"])
    g = hist.groupby("symbol")

    hist["ret"] = hist["close"] / hist["prev_close"] - 1
    hist["up"] = (hist["ret"] > 0).astype(int)
    hist["streak"] = g["up"].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumsum())
    hist[f"cum_ret{min_streak}"] = g["ret"].transform(
        lambda s: (1 + s).rolling(min_streak, min_periods=min_streak)
        .apply(np.prod, raw=True) - 1)

    # Only look at the latest row per symbol
    latest = hist.groupby("symbol").last().reset_index()
    latest = latest[(latest["streak"] >= min_streak)
                    & (latest[f"cum_ret{min_streak}"] >= min_cum)]
    latest = latest.merge(uni[["symbol", "sector", "med_turn20"]], on="symbol")
    latest = latest.sort_values(f"cum_ret{min_streak}", ascending=False)

    return [{
        "symbol": r["symbol"],
        "close": float(r["close"]),
        "streak": int(r["streak"]),
        "cum_return": round(float(r[f"cum_ret{min_streak}"]) * 100, 1),
        "turnover_cr": round(float(r["med_turn20"]) / 100, 1),
        "sector": r["sector"],
    } for _, r in latest.head(50).iterrows()]


# --------------------------------------------------------------------------
# 4. 52-week breakouts
# --------------------------------------------------------------------------

def breakouts_52w(raw: pd.DataFrame, uni: pd.DataFrame) -> list[dict]:
    """
    Stocks at or within 2% of their 52-week high, with above-average volume.
    """
    hist = raw[raw["symbol"].isin(set(uni["symbol"]))].copy()
    hist = hist.sort_values(["symbol", "date"])
    g = hist.groupby("symbol")

    # 52-week high (250 trading sessions, use whatever we have)
    hist["hi_250"] = g["high"].transform(
        lambda s: s.shift(1).rolling(250, min_periods=100).max())
    hist["vol_avg20"] = g["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).mean())

    latest = hist.groupby("symbol").last().reset_index()
    latest["from_hi"] = latest["close"] / latest["hi_250"] - 1
    latest["vol_ratio"] = np.where(
        latest["vol_avg20"] > 0,
        latest["volume"] / latest["vol_avg20"], np.nan)
    latest["ret"] = latest["close"] / latest["prev_close"] - 1

    # Within 2% of 52w high AND positive day AND above-average volume
    hits = latest[
        (latest["from_hi"] >= -0.02)
        & (latest["ret"] > 0)
        & (latest["vol_ratio"] >= 1.0)
    ].copy()
    hits = hits.merge(uni[["symbol", "sector", "med_turn20"]], on="symbol")
    hits = hits.sort_values("from_hi", ascending=False)

    return [{
        "symbol": r["symbol"],
        "close": float(r["close"]),
        "high_52w": float(r["hi_250"]),
        "from_high": round(float(r["from_hi"]) * 100, 1),
        "vol_ratio": round(float(r["vol_ratio"]), 1),
        "change": round(float(r["ret"]) * 100, 1),
        "turnover_cr": round(float(r["med_turn20"]) / 100, 1),
        "sector": r["sector"],
    } for _, r in hits.head(50).iterrows()]


# --------------------------------------------------------------------------
# 5. Sector pulse
# --------------------------------------------------------------------------

def sector_pulse(live: pd.DataFrame, uni: pd.DataFrame) -> list[dict]:
    """
    Sector-level snapshot: how many names advancing, average gain, standouts.
    """
    m = live.merge(uni[["symbol", "sector", "med_turn20"]], on="symbol")
    m = m[m["sector"] != "Unknown"]

    out = []
    for sector, grp in m.groupby("sector"):
        if len(grp) < 3:
            continue
        advancing = (grp["pchange"] > 0).sum()
        top = grp.nlargest(1, "pchange").iloc[0]
        out.append({
            "sector": sector,
            "stocks": len(grp),
            "advancing": int(advancing),
            "adv_pct": round(advancing / len(grp) * 100, 0),
            "avg_change": round(float(grp["pchange"].mean()), 2),
            "best_stock": top["symbol"],
            "best_change": round(float(top["pchange"]), 2),
        })

    out.sort(key=lambda x: x["avg_change"], reverse=True)
    return out


# --------------------------------------------------------------------------
# Combined intraday scan
# --------------------------------------------------------------------------

def intraday_all(on_progress=None) -> dict:
    """
    Run all intraday scanners in one pass (shares the universe + live fetch).
    Returns a dict with keys: unusual_volume, big_movers, sector_pulse, as_of.
    """
    now = db.now_ist()
    today = now.date()

    uni, _ = _liquid_eq(today - timedelta(days=1))
    print(f"[scanners] universe: {len(uni)} liquid names")

    # Shared snapshot — the circuit scan runs on the same schedule and reuses
    # this fetch instead of pulling the whole universe a second time.
    live = fetch.live_universe_snapshot(uni["symbol"].tolist(), on_progress=on_progress)
    if live.empty:
        return {
            "as_of": today.isoformat(),
            "scan_time": f"{now.hour:02d}:{now.minute:02d}",
            "unusual_volume": [],
            "big_movers": [],
            "sector_pulse": [],
        }

    # Volume for unusual_volume comes from a bounded per-symbol fetch; the
    # batch snapshot carries no volume. big_movers/sector_pulse use pchange.
    live_vol = augment_with_volume(live, uni)

    return {
        "as_of": today.isoformat(),
        "scan_time": f"{now.hour:02d}:{now.minute:02d}",
        "unusual_volume": unusual_volume(live_vol, uni),
        "big_movers": big_movers(live, uni),
        "sector_pulse": sector_pulse(live, uni),
    }


def eod_all() -> dict:
    """
    EOD scanners that use bhavcopy history. Run once after market close.
    Returns: momentum_streaks, breakouts_52w.
    """
    now = db.now_ist()
    today = now.date()

    uni, raw = _liquid_eq(today, n_sessions=260)

    return {
        "as_of": today.isoformat(),
        "momentum_streaks": momentum_streaks(raw, uni),
        "breakouts_52w": breakouts_52w(raw, uni),
    }
