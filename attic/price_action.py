"""
Do classic price-action concepts change the odds?

For every concept the question is the same: after the signal, does price move
1 ATR in the expected direction before it moves 1 ATR against, within 10
sessions, MORE OFTEN than on comparable days without the signal?

  outcome     entry at the next open; +1 ATR first = 1, -1 ATR first = 0,
              both on the same day or neither within 10 sessions = 0.5
  control     every liquid stock-day, grouped by (5-day return decile x
              distance-from-50-day-average quintile). Each signal is compared
              with the rate of its own group, so "price fell to support"
              is compared with other stocks that fell just as much.
  also shown  median 5-session return from entry vs the same day's median stock

Concepts
  Support hold       low reaches a prior swing low (within 0.5%), close holds above it
  Support, 2+ tests  same, on a level that has already held at least twice
  Resistance reject  high reaches a prior swing high, close stays below  (expects down)
  Support break      close more than 1% below an unbroken swing low     (expects down)
  Resistance break   close more than 1% above an unbroken swing high
  Role reversal      price returns to a broken swing high from above and holds
  Double bottom      two swing lows within 2%, 10-60 sessions apart, close above the peak between
  Hammer             long lower wick after a 5-day fall
  Bullish engulfing  after a 5-day fall;  Bearish engulfing after a 5-day rise (expects down)
  Inside-bar break   yesterday inside the prior bar, today closes above yesterday's high
  NR7 break up       yesterday the narrowest range of 7, today closes above its high
  Round-number rejection   high reaches a round number, close below     (expects down)
  Gap fill           after a 2%+ gap up, does price trade back to the prior close?

Swing points: the extreme of +/- 5 sessions, usable only once confirmed
(5 sessions later), from the last 250 sessions. A level dies on a close 1%
through it.

Data sets: all liquid NSE stocks from bhavcopies (Oct 2023 onward, prices
chained through NSE's adjusted prev_close), and F&O names on Yahoo
split-adjusted daily since 2009 (survivorship-biased: today's list).

Run: ~/anaconda3/bin/python attic/price_action.py 2>/dev/null
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ema_cross_breakout import load_fo  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ETF = r"ETF|BEES|^SETF|GOLD|SILVER|LIQUID|CASE$|NIFTY|SENSEX|IETF|MODEFENCE"
K = 5            # swing point half-width
MAX_AGE = 250
TOUCH = 0.005
BREAK = 0.01
HORIZON = 10

DOWN_CONCEPTS = {"Resistance reject", "Support break", "Bearish engulfing", "Round-number rejection"}


def load_bhav() -> pd.DataFrame:
    cols = ["symbol", "series", "date", "prev_close", "open", "high", "low", "close", "volume", "turnover"]
    fs = sorted(glob.glob(os.path.join(HERE, "..", "data", "raw", "bhav_*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in fs])
    d = d[(d.series == "EQ") & ~d.symbol.str.contains(ETF)]
    d = d[(d.close > 0) & (d.prev_close > 0) & (d.open > 0)]
    d["date"] = pd.to_datetime(d.date)
    d = d.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    d["A"] = (d.close / d.prev_close).groupby(d.symbol).cumprod()
    k = d.A / d.close
    out = pd.DataFrame({"symbol": d.symbol, "date": d.date, "open": d.open * k, "high": d.high * k,
                        "low": d.low * k, "close": d.A, "raw_close": d.close, "turnover_cr": d.turnover / 100})
    return out.reset_index(drop=True)


def per_symbol(g: pd.DataFrame) -> pd.DataFrame | None:
    n = len(g)
    if n < 120:
        return None
    o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    raw = g["raw_close"].to_numpy(float) if "raw_close" in g else c
    scale = raw / c                                              # adjusted -> actual rupees, for round numbers
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    to20 = pd.Series(g["turnover_cr"].to_numpy(float)).rolling(20, min_periods=10).median().to_numpy()
    sma50 = pd.Series(c).rolling(50).mean().to_numpy()
    r5 = c / np.r_[np.full(5, np.nan), c[:-5]] - 1

    # outcome: +1 ATR before -1 ATR from next open, within HORIZON sessions
    out = np.full(n, np.nan)
    f5 = np.full(n, np.nan)
    for t in range(n - 1):
        e = o[t + 1]
        a = atr[t]
        if not (np.isfinite(e) and np.isfinite(a)) or a <= 0:
            continue
        up, dn = e + a, e - a
        res = 0.5
        for j in range(t + 1, min(t + 1 + HORIZON, n)):
            hu, hd = h[j] >= up, l[j] <= dn
            if hu and hd:
                res = 0.5
                break
            if hu:
                res = 1.0
                break
            if hd:
                res = 0.0
                break
        out[t] = res if t + HORIZON < n else np.nan
        if t + 5 < n:
            f5[t] = c[t + 5] / e - 1

    # swing points, confirmed K sessions later
    piv_hi = np.zeros(n, bool)
    piv_lo = np.zeros(n, bool)
    for i in range(K, n - K):
        if h[i] == h[i - K:i + K + 1].max():
            piv_hi[i] = True
        if l[i] == l[i - K:i + K + 1].min():
            piv_lo[i] = True

    sig = {k: np.zeros(n, bool) for k in (
        "Support hold", "Support, 2+ tests", "Resistance reject", "Support break", "Resistance break",
        "Role reversal", "Double bottom", "Round-number rejection")}
    sup = []        # [price, formed_i, holds]
    res_ = []       # [price, formed_i]
    flipped = []    # broken resistance levels now below price: [price, broke_i]
    lows_hist = []  # confirmed swing lows (price, i) for double bottoms
    for t in range(n):
        i = t - K
        if i >= K and piv_lo[i]:
            sup.append([l[i], i, 0])
            lows_hist.append((l[i], i))
        if i >= K and piv_hi[i]:
            res_.append([h[i], i])
        sup = [s for s in sup if t - s[1] <= MAX_AGE]
        res_ = [r for r in res_ if t - r[1] <= MAX_AGE]
        flipped = [f for f in flipped if t - f[1] <= 60]
        keep = []
        for s in sup:
            lvl = s[0]
            if c[t] < lvl * (1 - BREAK):
                sig["Support break"][t] = True
                continue
            if l[t] <= lvl * (1 + TOUCH) and c[t] > lvl and t - s[1] > K + 5 and c[t - 1] > lvl * (1 + BREAK):
                sig["Support hold"][t] = True
                if s[2] >= 2:
                    sig["Support, 2+ tests"][t] = True
                s[2] += 1
            keep.append(s)
        sup = keep
        keep = []
        for r in res_:
            lvl = r[0]
            if c[t] > lvl * (1 + BREAK):
                sig["Resistance break"][t] = True
                flipped.append([lvl, t])
                continue
            if h[t] >= lvl * (1 - TOUCH) and c[t] < lvl and t - r[1] > K + 5 and c[t - 1] < lvl * (1 - BREAK):
                sig["Resistance reject"][t] = True
            keep.append(r)
        res_ = keep
        keep = []
        for f in flipped:
            lvl = f[0]
            if c[t] < lvl * (1 - BREAK):
                continue
            if t - f[1] >= 3 and l[t] <= lvl * (1 + TOUCH) and c[t] > lvl:
                sig["Role reversal"][t] = True
                continue
            keep.append(f)
        flipped = keep
        # double bottom: last two confirmed lows within 2%, 10-60 apart, close above the peak between
        if len(lows_hist) >= 2:
            (p2, i2), (p1, i1) = lows_hist[-1], lows_hist[-2]
            if 10 <= i2 - i1 <= 60 and abs(p2 / p1 - 1) <= 0.02 and t - i2 <= 20:
                neck = h[i1:i2 + 1].max()
                if c[t] > neck and c[t - 1] <= neck:
                    sig["Double bottom"][t] = True
        # round numbers in actual rupees
        rp = c[t] * scale[t]
        step = 100 if rp >= 1000 else 50 if rp >= 200 else 10 if rp >= 50 else 5
        rn = np.ceil(rp / step) * step
        if h[t] * scale[t] >= rn * (1 - 0.002) and rp < rn and rp > rn * 0.97:
            sig["Round-number rejection"][t] = True

    body = np.abs(c - o)
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    rng = h - l
    po, pcl, ph, plo = (np.r_[np.nan, x[:-1]] for x in (o, c, h, l))
    pr = np.r_[np.nan, rng[:-1]]
    fell = r5 < -0.03
    rose = r5 > 0.03
    sig["Hammer"] = fell & (lower >= 2 * body) & (upper <= body) & (rng > 0.6 * atr)
    sig["Bullish engulfing"] = fell & (pcl < po) & (c > o) & (o <= pcl) & (c >= po)
    sig["Bearish engulfing"] = rose & (pcl > po) & (c < o) & (o >= pcl) & (c <= po)
    pph, ppl = np.r_[np.nan, ph[:-1]], np.r_[np.nan, plo[:-1]]
    inside_prev = (ph < pph) & (plo > ppl)
    sig["Inside-bar break"] = inside_prev & (c > ph)
    nr7_prev = np.r_[False, (pd.Series(rng).rolling(7).min().to_numpy() == rng)[:-1]]
    sig["NR7 break up"] = nr7_prev & (c > ph)

    # gap fill
    gap = o / pc - 1
    fill1 = np.full(n, np.nan)
    fill5 = np.full(n, np.nan)
    fill20 = np.full(n, np.nan)
    for t in np.flatnonzero(gap >= 0.02):
        target = pc[t]
        lows = l[t:t + 20]
        if len(lows) < 20:
            continue
        hit = np.flatnonzero(lows <= target)
        first = hit[0] if len(hit) else 99
        fill1[t], fill5[t], fill20[t] = first == 0, first < 5, first < 20

    df = pd.DataFrame({"symbol": g.symbol.iat[0], "date": g.date.to_numpy(), "out": out, "f5": f5, "r5": r5,
                       "pos50": c / sma50 - 1, "liq": (to20 >= 2) & (c * scale >= 50) & np.isfinite(sma50),
                       "gap": gap, "fill1": fill1, "fill5": fill5, "fill20": fill20})
    for k, v in sig.items():
        df[k] = v
    return df


def run(df: pd.DataFrame, label: str) -> None:
    parts = [per_symbol(g) for _, g in df.groupby("symbol", sort=False)]
    x = pd.concat([p for p in parts if p is not None], ignore_index=True)
    x = x[x.liq & x.out.notna()].copy()
    x["b_r5"] = pd.qcut(x.r5, 10, labels=False, duplicates="drop")
    x["b_pos"] = pd.qcut(x.pos50, 5, labels=False, duplicates="drop")
    cell = x.groupby(["b_r5", "b_pos"]).out.mean().rename("ctrl")
    x = x.join(cell, on=["b_r5", "b_pos"])
    x["f5_ex"] = x.f5 - x.groupby("date").f5.transform("median")
    mid = x.date.min() + (x.date.max() - x.date.min()) / 2

    print(f"\n{'=' * 24} {label}: {x.date.min().date()} -> {x.date.max().date()}, "
          f"{x.symbol.nunique()} stocks, {len(x):,} stock-days {'=' * 24}")
    print(f"Overall chance of +1 ATR before -1 ATR on any day: {100 * x.out.mean():.1f}%\n")
    print(f"{'concept':26s} {'expects':>7s} {'n':>8s} {'right way':>10s} {'similar days':>13s} {'edge':>7s}"
          f" {'1st/2nd half edge':>18s} {'5d vs typical':>14s}")
    concepts = ["Support hold", "Support, 2+ tests", "Resistance reject", "Support break", "Resistance break",
                "Role reversal", "Double bottom", "Hammer", "Bullish engulfing", "Bearish engulfing",
                "Inside-bar break", "NR7 break up", "Round-number rejection"]
    for k in concepts:
        e = x[x[k]]
        if len(e) < 30:
            print(f"{k:26s} too few ({len(e)})")
            continue
        down = k in DOWN_CONCEPTS
        p = 1 - e.out if down else e.out
        q = 1 - e.ctrl if down else e.ctrl
        edge = p - q
        h1, h2 = edge[e.date < mid].mean(), edge[e.date >= mid].mean()
        fx = (-e.f5_ex if down else e.f5_ex).median()
        print(f"{k:26s} {'down' if down else 'up':>7s} {len(e):>8,d} {100 * p.mean():>9.1f}% {100 * q.mean():>12.1f}%"
              f" {100 * edge.mean():>+6.1f}  {100 * h1:>+7.1f} / {100 * h2:>+5.1f}  {100 * fx:>+12.2f}%")
    gp = x[x.gap >= 0.02]
    print(f"\nGap fill after a 2%+ gap up (n={len(gp):,}): same day {100 * gp.fill1.mean():.0f}%, "
          f"within 5 sessions {100 * gp.fill5.mean():.0f}%, within 20 sessions {100 * gp.fill20.mean():.0f}%")


if __name__ == "__main__":
    run(load_bhav(), "All liquid NSE stocks")
    run(load_fo(), "F&O stocks (Yahoo, adjusted)")
