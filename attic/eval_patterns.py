#!/usr/bin/env python3
"""
Classic chart patterns on the eval_systems swing ruler.

Each pattern is a daily-bar detector evaluated at the close of the day it
completes (the breakout / trigger bar). Entry is the next open, held a fixed
k sessions; the score is excess over the liquid universe's median k-session
return from the same entry date, Newey-West t at lag k, top TOP_SWING
signals per day ranked by that day's volume multiple. OHLC is put on the
adjusted basis first (factor adj/close) so splits do not draw fake patterns.

Detectors are deliberately simple, rule-based versions of the textbook
shapes -- what a scanner would actually fire on -- not hand-drawn charts.
Bearish patterns are also reported for F&O names only, since a multi-week
short needs futures.

Bar: clear SWING_COST, |t| >= 3 at the pattern's natural horizon, same sign
in every calendar year.

VERDICT (Oct 2023 -> Nov 2026, 22 patterns)
  Survive (all liquid): flat-base breakout (40d range <=15%, new 40d high,
  1.5x vol, above 200-DMA) and VCP-style tight-10d breakout near the 52w
  high: ~+1.7% excess over 20 sessions, t 3.3-4.0, positive every year.
  But the raw median is negative and only 44-46% rise -- they lose less
  than a falling small-cap tape and earn on a fat right tail.
  Survives (F&O vs F&O median): RSI14 > 80 is a BUY, not a short: +1.9%
  over 20 sessions, t 3.7, 66% rise, every year. Against the all-liquid
  median many F&O patterns looked strong; that was large-cap drift.
  Dead: double bottom/top, bull flag, golden/death cross, NR7, inside day,
  20d breakout, pocket pivot, 50-DMA pullback, 200-DMA reclaim, RSI dips,
  hammer/engulfing, 52w-low shorts, squeeze breakdowns.

    cd backend && .venv/bin/python attic/eval_patterns.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import baserates as br  # noqa: E402
import filing_facts  # noqa: E402
from eval_systems import MIN_PRICE, MIN_TURNOVER_CR, SWING_COST, TOP_SWING, features, nw_t  # noqa: E402

HORIZONS = (5, 10, 20, 40)


def roll(g, col, n, fn, shift=0):
    return g[col].transform(lambda s: getattr(s.shift(shift).rolling(n, min_periods=n), fn)())


def pattern_features(f: pd.DataFrame) -> pd.DataFrame:
    k = f["adj"] / f["close"]
    for c in ("open", "high", "low"):
        f[f"a_{c}"] = f[c] * k
    g = f.groupby("symbol", sort=False)
    a = f["adj"]
    f["p_open"] = g["a_open"].shift(1)
    f["p_high"] = g["a_high"].shift(1)
    f["p_low"] = g["a_low"].shift(1)

    # Breakout levels from the bars before today.
    f["hi20_prev"] = roll(g, "a_high", 20, "max", 1)
    f["hi40_prev"] = roll(g, "a_high", 40, "max", 1)
    f["lo40_prev"] = roll(g, "a_low", 40, "min", 1)
    f["lo20"] = roll(g, "a_low", 20, "min")
    f["lo250_prev"] = roll(g, "a_low", 250, "min", 1)
    f["hi10_prev"] = roll(g, "a_high", 10, "max", 1)
    f["lo10_prev"] = roll(g, "a_low", 10, "min", 1)

    # Range contraction.
    rng = f["a_high"] - f["a_low"]
    f["rng"] = rng
    f["nr7_prev"] = g["rng"].shift(1) <= roll(g, "rng", 7, "min", 1)
    f["inside_prev"] = (g["a_high"].shift(1) <= g["a_high"].shift(2)) & (g["a_low"].shift(1) >= g["a_low"].shift(2))

    # Bollinger squeeze.
    sd20 = g["adj"].transform(lambda s: s.rolling(20, min_periods=20).std())
    sma20 = g["adj"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    f["bb_up"] = sma20 + 2 * sd20
    f["bb_lo"] = sma20 - 2 * sd20
    f["bw"] = 4 * sd20 / sma20
    f["bw_q10_prev"] = g["bw"].transform(lambda s: s.shift(1).rolling(120, min_periods=100).quantile(0.10))
    f["p_bw"] = g["bw"].shift(1)

    # RSI (Wilder) 14 and 2.
    d = g["adj"].diff()
    for n in (14, 2):
        up = d.clip(lower=0).groupby(f["symbol"]).transform(lambda s: s.ewm(alpha=1 / n, adjust=False).mean())
        dn = (-d.clip(upper=0)).groupby(f["symbol"]).transform(lambda s: s.ewm(alpha=1 / n, adjust=False).mean())
        f[f"rsi{n}"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["p_rsi14"] = g["rsi14"].shift(1)

    # Moving-average state.
    f["p_sma50"] = g["sma50"].shift(1)
    f["sma50_20ago"] = g["sma50"].shift(20)
    f["sma200_20ago"] = g["sma200"].shift(20)

    # Pocket pivot: today's up-volume beats every down-day volume of the last 10.
    downvol = f["volume"].where(f["r1"] < 0, 0)
    f["downvol"] = downvol
    f["maxdownvol10_prev"] = roll(g, "downvol", 10, "max", 1)

    # Bull flag: strong pole into t-5, tight flag t-5..t-1, break today.
    f["pole"] = g["adj"].shift(5) / g["adj"].transform(lambda s: s.shift(5).rolling(20, min_periods=20).min()) - 1
    f["flag_hi"] = roll(g, "a_high", 5, "max", 1)
    f["flag_lo"] = roll(g, "a_low", 5, "min", 1)

    # Double bottom / top: two extremes within 3%, 15+ sessions apart,
    # neckline (the swing between them) at least 8% away, broken today.
    f["L1"] = roll(g, "a_low", 45, "min", 15)
    f["L2"] = roll(g, "a_low", 15, "min", 1)
    f["neck_hi"] = roll(g, "a_high", 30, "max", 1)
    f["H1"] = roll(g, "a_high", 45, "max", 15)
    f["H2"] = roll(g, "a_high", 15, "max", 1)
    f["neck_lo"] = roll(g, "a_low", 30, "min", 1)
    f["adj_80ago"] = g["adj"].shift(80)

    # Gap held: a >=4% gap-up 3 sessions ago whose gap has not been filled.
    f["gap_3ago"] = g["gap"].shift(3)
    f["pc_3ago"] = g["prev_close"].shift(3) * (g["adj"].shift(3) / g["close"].shift(3))
    f["lo3"] = roll(g, "a_low", 3, "min")
    return f


def patterns():
    vx = lambda d: d.vol_x  # noqa: E731
    above200 = lambda d: d.adj > d.sma200  # noqa: E731
    return [
        # --- breakouts from contraction ---
        ("Flat base breakout (40d range <=15%, new 40d high, 1.5x vol)", +1,
         lambda d: (d.adj > d.hi40_prev) & (d.hi40_prev / d.lo40_prev - 1 <= 0.15) & (d.vol_x >= 1.5) & above200(d)),
        ("VCP-style: tight 10d (<=6%) near 52w high, breaks out on vol", +1,
         lambda d: (d.hi10_prev / d.lo10_prev - 1 <= 0.06) & (d.hi10_prev >= 0.85 * d.hi250)
         & (d.adj > d.hi10_prev) & (d.vol_x >= 1.5)),
        ("Bollinger squeeze, close above upper band", +1,
         lambda d: (d.p_bw <= d.bw_q10_prev) & (d.adj > d.bb_up)),
        ("Bollinger squeeze, close below lower band (short)", -1,
         lambda d: (d.p_bw <= d.bw_q10_prev) & (d.adj < d.bb_lo)),
        ("NR7 then breakout above it, uptrend", +1,
         lambda d: d.nr7_prev & (d.adj > d.p_high) & above200(d)),
        ("Inside day then breakout above it, uptrend", +1,
         lambda d: d.inside_prev & (d.adj > d.p_high) & above200(d)),
        ("20-day high breakout (Donchian), above 200-DMA", +1,
         lambda d: (d.adj > d.hi20_prev) & above200(d)),
        ("Bull flag: +25% pole, <=10% flag, breaks flag high on 1.5x vol", +1,
         lambda d: (d.pole >= 0.25) & (d.flag_hi / d.flag_lo - 1 <= 0.10) & (d.adj > d.flag_hi) & (d.vol_x >= 1.5)),
        ("Pocket pivot above rising 50-DMA", +1,
         lambda d: (d.r1 > 0) & (d.volume > d.maxdownvol10_prev) & (d.adj > d.sma50)
         & (d.sma50 > d.sma50_20ago) & (d.adj <= 1.05 * d.sma50)),
        ("Gap-up >=4% held unfilled for 3 sessions", +1,
         lambda d: (d.gap_3ago >= 0.04) & (d.lo3 > d.pc_3ago)),
        # --- reversal shapes ---
        ("Double bottom, neckline break", +1,
         lambda d: (d.L2 / d.L1 - 1).abs().le(0.03) & (d.neck_hi >= 1.08 * d[["L1", "L2"]].max(axis=1))
         & (d.adj > d.neck_hi) & (d.adj_80ago >= 1.15 * d.L1)),
        ("Double top, neckline break (short)", -1,
         lambda d: (d.H2 / d.H1 - 1).abs().le(0.03) & (d.neck_lo <= 0.92 * d[["H1", "H2"]].min(axis=1))
         & (d.adj < d.neck_lo) & (d.adj_80ago <= 0.87 * d.H1)),
        ("Bullish engulfing at 20d low, above 200-DMA", +1,
         lambda d: (d.p_r1 < 0) & (d.adj > d.a_open) & (d.a_open <= d.p_adj) & (d.adj >= d.p_open)
         & (d.a_low <= d.lo20) & above200(d)),
        ("Hammer at 20d low, above 200-DMA", +1,
         lambda d: ((d[["a_open", "adj"]].min(axis=1) - d.a_low) >= 2 * (d.adj - d.a_open).abs())
         & (d.close_loc >= 0.7) & (d.a_low <= d.lo20) & above200(d)),
        # --- moving averages / oscillators ---
        ("Golden cross (50 over 200)", +1,
         lambda d: (d.sma50 > d.sma200) & (d.p_sma50 <= d.p_sma200)),
        ("Death cross (50 under 200) (short)", -1,
         lambda d: (d.sma50 < d.sma200) & (d.p_sma50 >= d.p_sma200)),
        ("Reclaims 200-DMA on 1.5x volume", +1,
         lambda d: (d.adj > d.sma200) & (d.p_adj <= d.p_sma200) & (d.vol_x >= 1.5)),
        ("Pullback to rising 50-DMA in uptrend (touch, close above)", +1,
         lambda d: (d.a_low <= 1.01 * d.sma50) & (d.adj > d.sma50) & (d.sma50 > d.sma200)
         & (d.sma50 > d.sma50_20ago) & (d.sma200 > d.sma200_20ago)),
        ("RSI14 crosses back above 30, above 200-DMA", +1,
         lambda d: (d.rsi14 > 30) & (d.p_rsi14 <= 30) & above200(d)),
        ("RSI2 < 5 above 200-DMA (Connors)", +1,
         lambda d: (d.rsi2 < 5) & above200(d)),
        ("RSI14 > 80 (overbought) (short)", -1,
         lambda d: d.rsi14 > 80),
        ("New 52-week low on 1.5x volume (short)", -1,
         lambda d: (d.a_low < d.lo250_prev) & (d.vol_x >= 1.5)),
    ], vx


def outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby("symbol", sort=False)
    entry = g["open"].shift(-1) * (g["adj"].shift(-1) / g["close"].shift(-1))
    out = frame[["date", "symbol"]].copy()
    for k in HORIZONS:
        out[f"f{k}"] = g["adj"].shift(-k) / entry - 1
        out[f"up{k}"] = out[f"f{k}"] > 0
    return out


def run(base: pd.DataFrame, bench: dict, restrict: set | None = None) -> pd.DataFrame:
    pats, rank = patterns()
    rows = []
    for name, sign, mask in pats:
        sel = base[mask(base).fillna(False)]
        if restrict is not None:
            sel = sel[sel["symbol"].isin(restrict)]
        if len(sel) < 40:
            continue
        sel = sel.assign(rank=rank(sel)).sort_values(["date", "rank"], ascending=[True, False])
        sel = sel.groupby("date").head(TOP_SWING)
        row = {"pattern": name, "side": "long" if sign > 0 else "short",
               "signals": len(sel), "per_day": len(sel) / base["date"].nunique()}
        for k in HORIZONS:
            x = sel.dropna(subset=[f"f{k}"])
            xs = sign * (x[f"f{k}"] - x["date"].map(bench[k]))
            day = xs.groupby(x["date"]).median()
            row[f"xs{k}"] = day.mean() * 100
            row[f"t{k}"] = nw_t(day, k)
            if k == 20:
                row["raw20"] = sign * x["f20"].median() * 100
                row["hit20"] = ((sign * x["f20"]) > 0).mean() * 100
                yr = day.groupby(day.index.year).mean() * 100
                row.update({f"y{y}": yr.get(y, np.nan) for y in (2024, 2025, 2026)})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    frame = br.load_frame()
    f = pattern_features(features(frame))
    fno = set(filing_facts.fno_symbols() or [])
    liquid = (f["med_to"] >= MIN_TURNOVER_CR) & (f["close"] >= MIN_PRICE) & ~f["frozen"]
    base = f[liquid].merge(outcomes(f), on=["date", "symbol"], how="left")
    bench = {k: base.groupby("date")[f"f{k}"].median() for k in HORIZONS}
    pd.set_option("display.width", 280, "display.max_colwidth", 66, "display.max_rows", 200)
    print(f"Data {f['date'].min().date()} -> {f['date'].max().date()}; "
          f"excess vs universe median, top {TOP_SWING}/day, cost {SWING_COST*100:.1f}% not deducted below")
    cols = ["pattern", "side", "signals", "per_day", "xs5", "t5", "xs10", "t10", "xs20", "t20",
            "xs40", "t40", "raw20", "hit20", "y2024", "y2025", "y2026"]
    r = run(base, bench).sort_values("t20", ascending=False)
    print("\n=== ALL LIQUID")
    print(r[cols].round(2).to_string(index=False))
    fbase = base[base["symbol"].isin(fno)]
    fbench = {k: fbase.groupby("date")[f"f{k}"].median() for k in HORIZONS}
    r2 = run(fbase, fbench)
    print("\n=== F&O NAMES ONLY (excess vs the F&O median, so large-cap drift is not counted)")
    print(r2[cols].sort_values("t20", ascending=False).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
