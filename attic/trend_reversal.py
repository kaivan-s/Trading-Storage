"""
Do "trend reversal" signals mark real bottoms?

Downtrend context on day t: close below a falling 200-day EMA (20-session
slope < 0) and at least 15% below the 52-week high.

Signals (each fires once per stock per 20 sessions)
  Reclaims 50 EMA      in a downtrend, first close above the 50 EMA after 20+ sessions below it
  20/50 EMA cross      in a downtrend, EMA20 crosses above EMA50
  Structure shift      in a downtrend, a higher swing low, then a close above the last swing high
  RSI divergence       in a downtrend, a new lower swing low with RSI(14) at least 3 points higher
  V-bounce             in a downtrend 10 sessions ago, now 15%+ above the 10-session low
  Reclaims 200 EMA     first close above a still-falling 200 EMA
  Golden cross         EMA50 crosses above EMA200

Entry at the next open. Outcomes at 20 and 60 sessions, compared with
  (a) the median liquid stock over the same dates
  (b) the median stock that was ALSO in a downtrend that day but did not signal
Also: how often price went on to undercut the low of the 20 sessions before
the signal within 60 sessions (the "reversal" failed).

Data: all liquid NSE stocks (bhavcopy, Oct 2023 onward) and F&O names on
Yahoo since 2009 (survivorship-biased).

Run: ~/anaconda3/bin/python attic/trend_reversal.py 2>/dev/null
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ema_cross_breakout import load_fo  # noqa: E402
from price_action import load_bhav  # noqa: E402

COOL = 20
SIGNALS = ["Reclaims 50 EMA", "20/50 EMA cross", "Structure shift", "RSI divergence", "V-bounce",
           "Reclaims 200 EMA", "Golden cross"]


def per_symbol(g: pd.DataFrame) -> pd.DataFrame | None:
    g = g.sort_values("date").reset_index(drop=True)
    n = len(g)
    if n < 300:
        return None
    o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    raw = g["raw_close"].to_numpy(float) if "raw_close" in g else c
    s = pd.Series(c)
    e20, e50, e200 = (s.ewm(span=k, adjust=False, min_periods=k).mean().to_numpy() for k in (20, 50, 200))
    slope = e200 / np.r_[np.full(20, np.nan), e200[:-20]] - 1
    hi250 = pd.Series(h).rolling(250, min_periods=200).max().to_numpy()
    ctx = (c < e200) & (slope < 0) & (c / hi250 - 1 <= -0.15)
    d = s.diff()
    rsi = (100 - 100 / (1 + d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
                        / (-d).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean())).to_numpy()
    to20 = pd.Series(g["turnover_cr"].to_numpy(float)).rolling(20, min_periods=10).median().to_numpy()
    liq = (to20 >= 2) & (raw >= 50) & np.isfinite(e200)

    below50_20 = pd.Series(c < e50).rolling(20).sum().shift(1).to_numpy() == 20
    sig = {k: np.zeros(n, bool) for k in SIGNALS}
    sig["Reclaims 50 EMA"] = ctx & (c > e50) & below50_20
    sig["20/50 EMA cross"] = ctx & (e20 > e50) & (np.r_[np.nan, e20[:-1]] <= np.r_[np.nan, e50[:-1]])
    lo10 = pd.Series(c).rolling(10).min().to_numpy()
    ctx10 = np.r_[np.zeros(10, bool), ctx[:-10]]
    sig["V-bounce"] = ctx10 & (c / lo10 - 1 >= 0.15)
    prev_c, prev_e200 = np.r_[np.nan, c[:-1]], np.r_[np.nan, e200[:-1]]
    sig["Reclaims 200 EMA"] = (c > e200) & (prev_c <= prev_e200) & (slope < 0)
    sig["Golden cross"] = (e50 > e200) & (np.r_[np.nan, e50[:-1]] <= np.r_[np.nan, e200[:-1]])

    k = 5
    lows, highs = [], []
    for t in range(2 * k, n):
        i = t - k
        if l[i] == l[i - k:i + k + 1].min():
            lows.append(i)
            if len(lows) >= 2:
                a, b = lows[-2], lows[-1]
                if l[b] < l[a] and b - a <= 60 and rsi[b] >= rsi[a] + 3 and ctx[t]:
                    sig["RSI divergence"][t] = True
        if h[i] == h[i - k:i + k + 1].max():
            highs.append(i)
        if len(lows) >= 2 and highs and ctx[t - 1]:
            a, b = lows[-2], lows[-1]
            sh = highs[-1]
            if l[b] > l[a] and sh > a and c[t] > h[sh] and c[t - 1] <= h[sh]:
                sig["Structure shift"][t] = True

    def fwd(hh):
        out = np.full(n, np.nan)
        out[:n - hh - 1] = c[hh + 1:] / o[1:n - hh] if n - hh - 1 > 0 else np.nan
        return out - 1
    f20, f60 = fwd(20), fwd(60)
    low20 = pd.Series(l).rolling(20).min().to_numpy()
    fut_min = pd.Series(l[::-1]).rolling(60, min_periods=60).min().to_numpy()[::-1]
    fut_min = np.r_[fut_min[1:], np.nan]                     # next 60 sessions, starting tomorrow
    failed = fut_min < low20

    df = pd.DataFrame({"symbol": g.symbol.iat[0], "date": g.date, "liq": liq, "ctx": ctx,
                       "f20": f20, "f60": f60, "failed": np.where(np.isfinite(fut_min), failed, np.nan)})
    for key, v in sig.items():
        keep = np.zeros(n, bool)
        last = -10**9
        for t in np.flatnonzero(v & liq):
            if t - last > COOL:
                keep[t] = True
                last = t
        df[key] = keep
    return df


def run(df: pd.DataFrame, label: str, start: str) -> None:
    parts = [per_symbol(g) for _, g in df.groupby("symbol", sort=False)]
    x = pd.concat([p for p in parts if p is not None], ignore_index=True)
    x = x[x.liq & (x.date >= pd.Timestamp(start))].copy()
    any_sig = x[SIGNALS].any(axis=1)
    for h in (20, 60):
        x[f"m{h}"] = x.date.map(x.groupby("date")[f"f{h}"].median())
        dt = x[x.ctx & ~any_sig]
        x[f"d{h}"] = x.date.map(dt.groupby("date")[f"f{h}"].median())
    dt = x[x.ctx & ~any_sig]
    base_fail = dt.failed.mean()
    mid = x.date.min() + (x.date.max() - x.date.min()) / 2
    print(f"\n{'=' * 18} {label}: signals {x.date.min().date()} -> {x.date.max().date()}, {x.symbol.nunique()} stocks {'=' * 18}")
    print(f"Downtrend stocks with no signal undercut their prior 20-day low within 60 sessions {100 * base_fail:.0f}% of the time\n")
    print(f"{'signal':18s} {'n':>6s} {'20d median':>10s} {'vs typical':>10s} {'vs downtrend':>12s}  "
          f"{'60d median':>10s} {'vs typical':>10s} {'vs downtrend':>12s} {'beat dt 60d':>11s}  {'1st/2nd half 60d vs dt':>22s} {'failed':>7s}")
    for k in SIGNALS:
        e = x[x[k]].dropna(subset=["f60", "d60"])
        if len(e) < 30:
            print(f"{k:18s} too few ({len(e)})")
            continue
        ex60 = e.f60 - e.d60
        print(f"{k:18s} {len(e):>6,d} {100 * e.f20.median():>+9.1f}% {100 * (e.f20 - e.m20).median():>+9.1f}% "
              f"{100 * (e.f20 - e.d20).median():>+11.1f}%  {100 * e.f60.median():>+9.1f}% {100 * (e.f60 - e.m60).median():>+9.1f}% "
              f"{100 * ex60.median():>+11.1f}% {100 * (ex60 > 0).mean():>10.0f}%  "
              f"{100 * ex60[e.date < mid].median():>+9.1f}% / {100 * ex60[e.date >= mid].median():>+5.1f}%  "
              f"{100 * e.failed.mean():>6.0f}%")


if __name__ == "__main__":
    run(load_fo(), "F&O stocks (Yahoo, adjusted)", "2010-01-01")
    run(load_bhav(), "All liquid NSE stocks", "2024-09-01")
