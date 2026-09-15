"""
Live overlay at Refresh: uptrend names near their 20-day high.

Last complete bhavcopy is usually yesterday. This joins last prices from
the moment of Refresh onto those EOD indicators and asks which names are
close enough to (or already through) the 20-day high to be in play.

This is not the EOD Setups list, and it is not a return forecast — the
shipped ranking predicts how far a name travels, not which way. See the
block comment above `for_tomorrow_momentum` for the measurements.
Delivery is unknown until the bhavcopy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as stk

NEAR_TRIGGER = 0.02      # tightened: within 2% of trigger (was 3%)
POTENTIAL_TRIGGER = 0.06 # wider net for "potential" early signals (within 6%)
THROUGH_EPS = 0.0        # LTP at or above yesterday's 20-day high
MIN_PCHANGE_HEAT = 2.0   # intraday move % to flag as heating up
VOL_BUILDING = 0.6       # volume building but not yet loud
# Near filter thresholds (tightened for quality)
NEAR_VOL_MIN = 1.2       # volume must be 1.2x 20d avg (was 1.0x)
NEAR_POS_HI = 0.92       # must be at 92% of range (was 88%)
NEAR_CMF_MIN = 0.0       # positive money flow required
# Momentum / volume break detection (different from coil)
MOMENTUM_VOL_RATIO = 1.5   # EOD vol_ratio threshold for momentum names
MOMENTUM_CMF = 0.10        # CMF threshold for money flow confirmation
MOMENTUM_RSI_MAX = 85.0    # not completely spent


def _v20(hist: pd.DataFrame) -> pd.Series:
    last = hist.sort_values(["symbol", "date"])
    return last.groupby("symbol")["volume"].apply(
        lambda s: s.tail(20).mean() if len(s) else np.nan
    )


def _breakout_line(m: pd.DataFrame) -> pd.Series:
    """
    The 20-day high the live price has to clear.

    `trigger` on a row includes that bar's own high, so a close on the same
    session can never print through it. When live LTP is that same bar
    (official close, or scan_all feeding adj as ltp), use `prior_trigger`.
    When live is a later overlay, `trigger` is already yesterday's 20-day
    high — the right line.
    """
    trig = pd.to_numeric(m["trigger"], errors="coerce")
    ltp = pd.to_numeric(m["ltp"], errors="coerce")
    adj = pd.to_numeric(m["adj"], errors="coerce")
    prior = (pd.to_numeric(m["prior_trigger"], errors="coerce")
             if "prior_trigger" in m.columns
             else pd.Series(np.nan, index=m.index))
    same = (ltp - adj).abs() <= (adj.abs() * 1e-4 + 1e-6)
    return trig.mask(same & prior.notna(), prior)


def for_tomorrow(coil_stocks: pd.DataFrame, live: pd.DataFrame,
                 buys: pd.DataFrame | None = None,
                 scan_rows: pd.DataFrame | None = None,
                 near: float = NEAR_TRIGGER) -> pd.DataFrame:
    """
    `coil_stocks` = add_indicators output through last EOD.
    `live` = fetch.live_snapshot() (ltp, volume, high, …).
    `scan_rows` = sector classifications (for early detection in CROSSING sectors).
    """
    empty_cols = [
        "symbol", "sector", "ltp", "trigger", "to_trigger", "vol_expand",
        "pchange", "pos_hi", "rsi", "cmf", "kind", "was_setup", "sector_klass", "why",
    ]
    if coil_stocks is None or coil_stocks.empty or live is None or live.empty:
        return pd.DataFrame(columns=empty_cols)

    as_of = coil_stocks["date"].max()
    eod = coil_stocks[coil_stocks["date"] == as_of].copy()
    if eod.empty:
        return pd.DataFrame(columns=empty_cols)

    v20 = _v20(coil_stocks)
    setup_syms = set()
    if buys is not None and not buys.empty:
        setup_syms = set(buys["symbol"].astype(str))
    
    # Get sector classifications
    crossing_sectors = set()
    pullback_sectors = set()
    if scan_rows is not None and not scan_rows.empty:
        crossing_sectors = set(scan_rows[scan_rows["klass"] == "CROSSING"]["sector"])
        pullback_sectors = set(scan_rows[scan_rows["klass"] == "PULLBACK"]["sector"])

    snap = live.copy()
    snap["symbol"] = snap["symbol"].astype(str)
    keep_live = [c for c in ["symbol", "ltp", "volume", "pchange", "high", "time"]
                 if c in snap.columns]
    snap = snap[keep_live].rename(columns={
        "volume": "live_volume", "high": "live_high", "time": "live_time",
    })
    m = eod.merge(snap, on="symbol", how="inner")
    if m.empty:
        return pd.DataFrame(columns=empty_cols)

    ltp = pd.to_numeric(m["ltp"], errors="coerce")
    trig = _breakout_line(m)
    m["trigger"] = trig
    m["to_trigger"] = trig / ltp - 1.0
    live_vol = pd.to_numeric(m.get("live_volume"), errors="coerce")
    m["vol_expand"] = live_vol / m["symbol"].map(v20).replace(0, np.nan)

    through = ltp >= trig * (1.0 + THROUGH_EPS)
    near_line = m["to_trigger"].between(-0.005, near)
    trend = (
        (m["adj"] > m["ema50"]) & (m["ema50"] > m["ema200"])
        if {"ema50", "ema200", "adj"}.issubset(m.columns)
        else True
    )
    m["was_setup"] = m["symbol"].isin(setup_syms)
    # NEAR filter requirements (tightened)
    loud = m["vol_expand"].fillna(0) >= NEAR_VOL_MIN  # 1.2x volume
    near_high = m["pos_hi"].fillna(0) >= NEAR_POS_HI if "pos_hi" in m.columns else True  # 92% of high
    near_cmf = m["cmf"].fillna(0) >= NEAR_CMF_MIN if "cmf" in m.columns else True  # positive CMF

    # "potential" - wider net for early breakout signals (coil-like)
    pchange = pd.to_numeric(m.get("pchange"), errors="coerce").fillna(0)
    potential_zone = m["to_trigger"].between(-0.01, POTENTIAL_TRIGGER)  # within 6%
    vol_building = m["vol_expand"].fillna(0) >= VOL_BUILDING
    heating = pchange >= MIN_PCHANGE_HEAT  # strong intraday move
    rsi_ok = m["rsi"].fillna(50).between(40, 72) if "rsi" in m.columns else True
    cmf_ok = m["cmf"].fillna(0) >= -0.05 if "cmf" in m.columns else True

    # Potential: in the zone, with volume building OR heating up, good trend
    is_potential = (
        trend & potential_zone & ~through & ~near_line
        & (vol_building | heating)
        & rsi_ok & cmf_ok & near_high
    )

    # "momentum" - different signal: volume break candidates (not coiled)
    # These fail coil filters but show strong momentum toward trigger
    eod_vol_ratio = m["vol_ratio"].fillna(0) if "vol_ratio" in m.columns else 0
    eod_cmf = m["cmf"].fillna(0) if "cmf" in m.columns else 0
    eod_rsi = m["rsi"].fillna(50) if "rsi" in m.columns else 50
    is_momentum = (
        trend
        & potential_zone  # within 6% of trigger
        & ~through
        & (eod_vol_ratio >= MOMENTUM_VOL_RATIO)  # volume already expanding (not dry)
        & (eod_cmf >= MOMENTUM_CMF)  # strong money flow
        & (eod_rsi <= MOMENTUM_RSI_MAX)  # not completely exhausted
        & near_high  # at least near the high
        & ~is_potential  # not already caught as potential
    )

    keep = trend & (
        through
        | (near_line & (m["was_setup"] | (loud & near_high & near_cmf)))
        | is_potential
        | is_momentum
    )
    # Add sector classification
    def get_sector_klass(sector):
        if sector in pullback_sectors:
            return "PULLBACK"
        if sector in crossing_sectors:
            return "CROSSING"
        return ""
    m["sector_klass"] = m["sector"].apply(get_sector_klass)

    hit = m[keep].copy()
    if hit.empty:
        return pd.DataFrame(columns=empty_cols)

    def kind(r):
        if r["to_trigger"] <= 0:
            return "through"
        # Coil in CROSSING sector = "early" (ahead of pullback)
        if r["was_setup"] and r["to_trigger"] <= near:
            if r.get("sector_klass") == "CROSSING":
                return "early"
            return "setup"
        if r["to_trigger"] <= near:
            return "near"
        # Check if it's momentum (high vol_ratio, not coil-like)
        vol_r = r.get("vol_ratio", 0) or 0
        rsi = r.get("rsi", 50) or 50
        if vol_r >= MOMENTUM_VOL_RATIO and rsi > 68:
            return "momentum"
        return "potential"

    hit["kind"] = hit.apply(kind, axis=1)

    def why(r):
        ltp_s = float(r["ltp"])
        trig_s = float(r["trigger"])
        gap = float(r["to_trigger"]) * 100
        vol = r["vol_expand"]
        pch = r.get("pchange") or 0
        vol_s = f"{float(vol):.1f}× so far" if pd.notna(vol) else "volume unknown"
        if r["kind"] == "through":
            return (
                f"Live {ltp_s:.2f} is through yesterday's trigger {trig_s:.2f} "
                f"({vol_s}). Today's close can confirm; if it holds, this is "
                "the break going into tomorrow."
            )
        if r["kind"] == "setup":
            return (
                f"Logged setup, live {ltp_s:.2f} is {gap:.1f}% from the trigger "
                f"at {trig_s:.2f} ({vol_s}). A close through that line is the "
                "buy; tomorrow's open is the follow if it closes under."
            )
        if r["kind"] == "early":
            return (
                f"Early: logged setup in CROSSING sector (not yet in pullback). "
                f"Live {ltp_s:.2f} is {gap:.1f}% from trigger {trig_s:.2f} ({vol_s}). "
                "Getting coiled before sector confirms — higher risk but earlier entry."
            )
        if r["kind"] == "potential":
            heat = f", up {pch:.1f}% today" if pch >= MIN_PCHANGE_HEAT else ""
            return (
                f"Potential early: live {ltp_s:.2f} is {gap:.1f}% from trigger "
                f"{trig_s:.2f} ({vol_s}{heat}). Further out but showing interest — "
                "watch for volume surge into close or gap tomorrow."
            )
        if r["kind"] == "momentum":
            vol_r = r.get("vol_ratio") or 0
            rsi = r.get("rsi") or 0
            return (
                f"Momentum: live {ltp_s:.2f} is {gap:.1f}% from trigger {trig_s:.2f}. "
                f"Not a coil (RSI {rsi:.0f}, vol {vol_r:.1f}× 20d avg) — already running. "
                f"({vol_s}). Potential volume break if it closes through on expansion."
            )
        return (
            f"Live {ltp_s:.2f} is {gap:.1f}% from the 20-day high {trig_s:.2f} "
            f"({vol_s}). Not a coiled setup — watch the last 15 minutes and "
            "tomorrow's open, first unit only."
        )

    hit["why"] = hit.apply(why, axis=1)
    cols = [
        "symbol", "sector", "ltp", "trigger", "to_trigger", "vol_expand",
        "pchange", "pos_hi", "rsi", "cmf", "kind", "was_setup", "sector_klass", "why",
    ]
    cols = [c for c in cols if c in hit.columns]
    order = {"through": 0, "setup": 1, "early": 2, "near": 3, "momentum": 4, "potential": 5}
    hit["_o"] = hit["kind"].map(order)
    return (hit.sort_values(["_o", "to_trigger", "vol_expand"],
                            ascending=[True, True, False])[cols]
            .head(100).reset_index(drop=True))


# ==========================================================================
# Energy-scored variant — what it does and does not predict
# --------------------------------------------------------------------------
# Measured over 324 cached sessions (114 usable scan days), three independent
# ways, at 1 / 5 / 20-session horizons:
#
#   The score does NOT rank return. Median excess return of the top 40 came
#   in BELOW the median of its own gated pool at every horizon (+0.02% vs
#   +0.06% at 1 session, +0.26% vs +0.37% at 5, +1.09% vs +1.49% at 20), so
#   ranking by it is worse than taking every name that clears the gates.
#   Against the market the next-day edge is +0.02%, p=0.73.
#
#   The score DOES rank intraday range, very reliably. The top 40 reached
#   +3% above the close on the next session 9-11pp more often than the rest
#   of the gated pool, t=13, holding on 87-92% of sessions. Its next-day
#   return dispersion is 3.18% against 2.51% for the pool — it selects
#   movement in both directions, at the same return per unit of risk.
#
# Hence the UI presents this as expected range, not as picks. Two dead ends
# recorded so they are not retried: reweighting the score from the features
# with the strongest cross-sectional ICs scored WORSE than the current
# weights, and the single best 1-day feature (low ATR) was significantly
# negative inside this gate set — broad-universe ICs do not survive being
# conditioned on "uptrend, near the 20-day high". Filtering on sector state
# also hurts (-0.97pp at 20 sessions, t=-4.68), which is why scan_rows is
# used only to label rows and never to select them.
# ==========================================================================

# Momentum zone: how far below / above the trigger a name may sit.
MOM_ZONE_BELOW = 0.08    # up to 8% below the 20-day high
MOM_ZONE_ABOVE = 0.06    # up to 6% already past it (don't chase big gaps)
MOM_RSI_MAX = 85.0       # skip fully exhausted names
MOM_MIN_PRICE = 20.0

# Score weights (sum ~1.0).
MOM_W = {
    "vol": 0.30,   # volume energy vs 20-day
    "rsi": 0.20,   # momentum (favour 58-80)
    "atr": 0.18,   # volatility / range expansion
    "ext": 0.14,   # extension above the 20-EMA
    "cmf": 0.18,   # money flow
}


def _clip01(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((pd.to_numeric(s, errors="coerce") - lo) / (hi - lo)).clip(0, 1)


def for_tomorrow_momentum(coil_stocks: pd.DataFrame, live: pd.DataFrame,
                          scan_rows: pd.DataFrame | None = None,
                          top_n: int = 100) -> pd.DataFrame:
    """
    Energy-scored variant of `for_tomorrow`. Same inputs, but ranks uptrend
    names by expected intraday range rather than by coil quietness. Returns
    a frame with a `score` column.

    `score` is a validated range forecast and not a return forecast — see
    the block comment above before treating the order as a preference.
    """
    empty_cols = [
        "symbol", "sector", "ltp", "trigger", "to_trigger", "vol_expand",
        "pchange", "pos_hi", "rsi", "cmf", "vol_ratio", "atr_pct",
        "ext_ema20", "score", "kind", "sector_klass", "why",
    ]
    if coil_stocks is None or coil_stocks.empty or live is None or live.empty:
        return pd.DataFrame(columns=empty_cols)

    as_of = coil_stocks["date"].max()
    eod = coil_stocks[coil_stocks["date"] == as_of].copy()
    if eod.empty:
        return pd.DataFrame(columns=empty_cols)

    v20 = _v20(coil_stocks)

    crossing_sectors, pullback_sectors = set(), set()
    if scan_rows is not None and not scan_rows.empty:
        crossing_sectors = set(scan_rows[scan_rows["klass"] == "CROSSING"]["sector"])
        pullback_sectors = set(scan_rows[scan_rows["klass"] == "PULLBACK"]["sector"])

    snap = live.copy()
    snap["symbol"] = snap["symbol"].astype(str)
    keep_live = [c for c in ["symbol", "ltp", "volume", "pchange", "high", "time"]
                 if c in snap.columns]
    snap = snap[keep_live].rename(columns={
        "volume": "live_volume", "high": "live_high", "time": "live_time",
    })
    m = eod.merge(snap, on="symbol", how="inner")
    if m.empty:
        return pd.DataFrame(columns=empty_cols)

    ltp = pd.to_numeric(m["ltp"], errors="coerce")
    trig = _breakout_line(m)
    m["trigger"] = trig
    m["to_trigger"] = trig / ltp - 1.0
    live_vol = pd.to_numeric(m.get("live_volume"), errors="coerce")
    m["vol_expand"] = live_vol / m["symbol"].map(v20).replace(0, np.nan)

    # --- gates -----------------------------------------------------------
    trend = (
        (m["adj"] > m["ema50"]) & (m["ema50"] > m["ema200"])
        if {"ema50", "ema200", "adj"}.issubset(m.columns)
        else pd.Series(True, index=m.index)
    )
    in_zone = m["to_trigger"].between(-MOM_ZONE_ABOVE, MOM_ZONE_BELOW)
    priced = ltp >= MOM_MIN_PRICE
    rsi = m["rsi"].fillna(50) if "rsi" in m.columns else pd.Series(50.0, index=m.index)
    not_spent = rsi <= MOM_RSI_MAX

    keep = trend & in_zone & priced & not_spent
    m = m[keep].copy()
    if m.empty:
        return pd.DataFrame(columns=empty_cols)

    # --- momentum score (energy, not quietness) --------------------------
    vol_r = m["vol_ratio"] if "vol_ratio" in m.columns else m["vol_expand"]
    s_vol = _clip01(vol_r.fillna(0), 0.8, 2.5)
    # RSI: ramp 55->72 as good, taper 72->85 (avoid exhaustion)
    r = m["rsi"].fillna(50)
    s_rsi = np.where(
        r <= 72,
        _clip01(r, 55, 72),
        (1.0 - _clip01(r, 72, 85)),
    )
    s_rsi = pd.Series(s_rsi, index=m.index).clip(0, 1)
    s_atr = _clip01(m["atr_pct"], 0.015, 0.06) if "atr_pct" in m.columns else 0.0
    s_ext = _clip01(m["ext_ema20"], 0.0, 0.15) if "ext_ema20" in m.columns else 0.0
    s_cmf = _clip01(m["cmf"], 0.0, 0.25) if "cmf" in m.columns else 0.0

    m["score"] = (
        MOM_W["vol"] * s_vol
        + MOM_W["rsi"] * s_rsi
        + MOM_W["atr"] * s_atr
        + MOM_W["ext"] * s_ext
        + MOM_W["cmf"] * s_cmf
    )
    # small heat bonus for today's move
    pch = pd.to_numeric(m.get("pchange"), errors="coerce").fillna(0)
    m["score"] = m["score"] + 0.05 * _clip01(pch, 1.0, 6.0)

    def sector_klass(sec):
        if sec in pullback_sectors:
            return "PULLBACK"
        if sec in crossing_sectors:
            return "CROSSING"
        return ""
    m["sector_klass"] = m["sector"].apply(sector_klass)
    m["kind"] = m["to_trigger"].apply(lambda g: "through" if pd.notna(g) and g <= 0 else "momentum")

    def why(r):
        return (
            f"Expected range {r['score']:.2f}: RSI {float(r.get('rsi') or 0):.0f}, "
            f"vol {float(r.get('vol_ratio') or 0):.1f}×, "
            f"{float(r['to_trigger'])*100:+.1f}% to the 20-day high. "
            "Ranked for how far it is likely to travel, in either direction."
        )
    m["why"] = m.apply(why, axis=1)

    cols = [c for c in empty_cols if c in m.columns]
    return (m.sort_values("score", ascending=False)[cols]
            .head(top_n).reset_index(drop=True))


def _gate_mask(day: pd.DataFrame) -> pd.Series:
    """For Tom hard gates only — no top-N cut. Used for the track-record base rate."""
    adj = pd.to_numeric(day["adj"], errors="coerce")
    line = (pd.to_numeric(day["prior_trigger"], errors="coerce")
            if "prior_trigger" in day.columns
            else pd.Series(np.nan, index=day.index))
    if "trigger" in day.columns:
        line = line.fillna(pd.to_numeric(day["trigger"], errors="coerce"))
    gap = line / adj.replace(0, np.nan) - 1.0
    trend = True
    if {"ema50", "ema200"}.issubset(day.columns):
        trend = (adj > day["ema50"]) & (day["ema50"] > day["ema200"])
    rsi_ok = True
    if "rsi" in day.columns:
        rsi_ok = day["rsi"].fillna(50) <= MOM_RSI_MAX
    priced = adj >= MOM_MIN_PRICE
    return trend & priced & rsi_ok & gap.between(-MOM_ZONE_ABOVE, MOM_ZONE_BELOW)


def momentum_base_rates(stocks: pd.DataFrame, scan_date=None) -> dict:
    """
    Next-session high vs the prior 20-day high, for every name that passed
    For Tom's hard gates. This is the number the Track Record must beat —
    the top-N score cut is the only thing the list adds on top of the gates.
    """
    empty_leg = {"n": 0, "hit_rate": None, "avg_gain": None}
    out = {
        **empty_leg,
        "scan_date": None,
        "scope": None,
        "pooled": dict(empty_leg),
    }
    if stocks is None or stocks.empty:
        return out
    need = {"symbol", "date", "adj"}
    if not need.issubset(stocks.columns):
        return out
    high_col = "adj_high" if "adj_high" in stocks.columns else (
        "high" if "high" in stocks.columns else None
    )
    if high_col is None:
        return out

    s = stocks.sort_values(["symbol", "date"]).copy()
    g = s.groupby("symbol", sort=False)
    if "prior_trigger" not in s.columns and "trigger" in s.columns:
        s["prior_trigger"] = g["trigger"].shift(1)
    s["_nxt_high"] = g[high_col].shift(-1)
    s["_line"] = (pd.to_numeric(s["prior_trigger"], errors="coerce")
                  if "prior_trigger" in s.columns
                  else pd.Series(np.nan, index=s.index))
    if "trigger" in s.columns:
        s["_line"] = s["_line"].fillna(pd.to_numeric(s["trigger"], errors="coerce"))

    u = s[_gate_mask(s) & s["_nxt_high"].notna() & s["_line"].notna()
          & (pd.to_numeric(s["adj"], errors="coerce") > 0)].copy()

    def _stats(df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return dict(empty_leg)
        hit = df["_nxt_high"] >= df["_line"]
        gain = df["_nxt_high"] / pd.to_numeric(df["adj"], errors="coerce") - 1.0
        return {
            "n": int(len(df)),
            "hit_rate": float(hit.mean()),
            "avg_gain": float(gain.mean()),
        }

    pooled = _stats(u)
    out["pooled"] = pooled
    if scan_date:
        d = pd.Timestamp(scan_date)
        day = _stats(u[u["date"] == d])
        out["scan_date"] = d.strftime("%Y-%m-%d")
        if day["n"]:
            out.update(day)
            out["scope"] = "date"
        else:
            out.update(pooled)
            out["scope"] = "pooled"
    else:
        out.update(pooled)
        out["scope"] = "pooled"
    return out
