"""
Backtest of the "fresh 10/20 EMA cross + recent breakout" screen.

Setup on day t (all at the close):
  1. Close above the 10, 20, 50 and 200-day EMAs.
  2. EMA10 crossed above EMA20 within the last CROSS_WIN sessions.
  3. A close above the prior 20-session high within the last BO_WIN sessions,
     and the close is still above that breakout level.
  4. Close 3-15% above the 20-day EMA (near the average, not stretched).
  5. At least 5 of the 7 trend-template checks (Minervini-style).
A stock signals on the first day it passes, then not again for COOLDOWN sessions.

Entry at the next session's open. Forward returns at 5/10/20/60 sessions are
compared with the median liquid stock's return over the same entry-to-exit
window, and with a baseline of "any stock above all four EMAs" so the cross
and breakout are tested on top of plain trend.

A rule-based exit is also tested: sell at the next open after a close below
the 20-day EMA, with a 60-session cap.

Data sets:
  fo   - today's F&O names, Yahoo split-adjusted daily, 2010 onward
  all  - every liquid NSE EQ stock from bhavcopies (late 2023 onward),
         names with split-sized jumps are dropped

Run: ~/anaconda3/bin/python attic/ema_cross_breakout.py 2>/dev/null
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
YAHOO = os.path.join(HERE, "..", "data", "cache", "yahoo_daily_fo.parquet")
CROSS_WIN, BO_WIN, COOLDOWN = 10, 10, 20
HORIZONS = (5, 10, 20, 60)
COST = 0.0025
ETF = r"ETF|BEES|^SETF|GOLD|SILVER|LIQUID|CASE$|NIFTY|SENSEX|IETF|MODEFENCE"


def load_fo() -> pd.DataFrame:
    df = pd.read_parquet(YAHOO).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["turnover_cr"] = df.close * df.volume / 1e7
    return df[["symbol", "date", "open", "high", "low", "close", "volume", "turnover_cr"]]


def load_all() -> pd.DataFrame:
    cols = ["symbol", "series", "date", "open", "high", "low", "close", "volume", "turnover"]
    fs = sorted(glob.glob(os.path.join(HERE, "..", "data", "raw", "bhav_*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in fs])
    df = df[(df.series == "EQ") & ~df.symbol.str.contains(ETF)].drop(columns="series")
    df["date"] = pd.to_datetime(df["date"])
    df["turnover_cr"] = df.turnover / 100
    df = df[df.close > 0].sort_values(["symbol", "date"])
    jump = df.open / df.groupby("symbol").close.shift(1)
    bad = df.loc[(jump < 0.6) | (jump > 1.6), "symbol"].unique()
    return df[~df.symbol.isin(bad)].drop(columns="turnover")


def features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c, h, l = g.close, g.high, g.low
    ema = {n: c.ewm(span=n, adjust=False, min_periods=n).mean() for n in (10, 20, 50, 200)}
    sma = {n: c.rolling(n).mean() for n in (50, 150, 200)}
    up = ema[10] > ema[20]
    cross = up & ~up.shift(1, fill_value=False)
    recent_cross = cross.rolling(CROSS_WIN, min_periods=1).max().astype(bool)
    hi20 = h.shift(1).rolling(20).max()
    bo = c > hi20
    bo_lvl = hi20.where(bo).ffill(limit=BO_WIN - 1)
    recent_bo = bo.rolling(BO_WIN, min_periods=1).max().astype(bool) & (c > bo_lvl)
    hi52, lo52 = h.rolling(250, min_periods=200).max(), l.rolling(250, min_periods=200).min()
    stage = ((c > sma[150]) & (c > sma[200])).astype(int) + (sma[150] > sma[200]) + \
        (sma[200] > sma[200].shift(21)) + ((sma[50] > sma[150]) & (sma[50] > sma[200])) + \
        (c > sma[50]) + (c >= 1.3 * lo52) + (c >= 0.75 * hi52)
    above_all = (c > ema[10]) & (c > ema[20]) & (c > ema[50]) & (c > ema[200])
    from20 = c / ema[20] - 1
    g["above_all"] = above_all & ema[200].notna()
    g["setup"] = g.above_all & recent_cross & recent_bo & from20.between(0.03, 0.15) & (stage >= 5)
    vol_x = g.volume / g.volume.shift(1).rolling(50, min_periods=30).median()
    g["setup_vol"] = g.setup & (vol_x >= 5).rolling(10, min_periods=1).max().astype(bool)
    g["below20"] = c < ema[20]
    g["liq"] = (g.turnover_cr.rolling(20).median() >= 2) & (c >= 50)
    return g


def first_signals(g: pd.DataFrame, col: str) -> np.ndarray:
    idx, last = [], -10**9
    for i in np.flatnonzero(g[col].to_numpy() & g.liq.to_numpy()):
        if i - last > COOLDOWN:
            idx.append(i)
            last = i
    return np.array(idx, dtype=int)


def run(df: pd.DataFrame, start: str) -> None:
    frames = [features(g) for _, g in df.groupby("symbol", sort=False)]
    dates = sorted(df.date.unique())
    # median liquid stock's open-to-close-at-horizon return, by entry date
    o = pd.concat(frames)
    for k in HORIZONS:
        o[f"fw{k}"] = o.groupby("symbol").close.shift(-k) / o.groupby("symbol").open.shift(-1) - 1
    o["entry_date"] = o.groupby("symbol").date.shift(-1)
    bench = o[o.liq].groupby("date")[[f"fw{k}" for k in HORIZONS]].median()

    out = {"setup": [], "setup_vol": [], "above_all": []}
    exits = []
    for g in frames:
        for col in out:
            for i in first_signals(g, col):
                if i + 1 >= len(g) or g.date.iat[i] < pd.Timestamp(start):
                    continue
                e = g.open.iat[i + 1]
                r = dict(symbol=g.symbol.iat[0], date=g.date.iat[i])
                for k in HORIZONS:
                    r[f"r{k}"] = g.close.iat[i + k] / e - 1 if i + k < len(g) else np.nan
                out[col].append(r)
                if col == "setup":
                    j = i + 1
                    while j < len(g) - 1 and j - i < 60 and not g.below20.iat[j]:
                        j += 1
                    px = g.open.iat[j + 1] if j + 1 < len(g) and g.below20.iat[j] else g.close.iat[j]
                    exits.append(dict(date=g.date.iat[i], ret=px / e - 1 - COST, days=j - i))

    print(f"{df.date.min().date()} -> {df.date.max().date()}, {df.symbol.nunique()} stocks, signals from {start}")
    for col, label in (("setup", "SETUP (cross + breakout + trend)"),
                       ("setup_vol", "SETUP + a 5x-volume day in the last 10 sessions (catalyst proxy)"),
                       ("above_all", "BASELINE (any stock above all 4 EMAs)")):
        t = pd.DataFrame(out[col])
        t = t.join(bench, on="date")
        print(f"\n{label}: {len(t)} signals")
        for k in HORIZONS:
            x = t[f"r{k}"].dropna()
            ex = (t[f"r{k}"] - t[f"fw{k}"]).dropna()
            print(f"  {k:>2} sessions: median {100 * x.median():+5.1f}%  avg {100 * x.mean():+5.1f}%  rose {100 * (x > 0).mean():3.0f}%"
                  f"   vs typical stock: median {100 * ex.median():+5.1f}%  avg {100 * ex.mean():+5.1f}%  beat {100 * (ex > 0).mean():3.0f}%")
        if col == "setup":
            yr = (t.r20 - t.fw20).groupby(t.date.dt.year).median() * 100
            print("  20-session excess by year: " + "  ".join(f"{y}: {v:+.1f}" for y, v in yr.items()))
    x = pd.DataFrame(exits)
    print(f"\nEXIT RULE (sell after a close below 20 EMA, max 60 sessions, {COST * 100:.2f}% cost): "
          f"{len(x)} trades  win {100 * (x.ret > 0).mean():.0f}%  avg {100 * x.ret.mean():+.1f}%  "
          f"median {100 * x.ret.median():+.1f}%  avg hold {x.days.mean():.0f} sessions  "
          f"avg win {100 * x.ret[x.ret > 0].mean():+.1f}%  avg loss {100 * x.ret[x.ret <= 0].mean():+.1f}%")


def bracket(o, h, l, c, i, tgt, stp, days=5):
    """Enter at the next open; exit at target or stop (stop first if both hit the same day), else the close on day `days`."""
    e = o[i + 1]
    for j in range(i + 1, min(i + 1 + days, len(c))):
        if l[j] <= e * (1 - stp):
            return min(o[j], e * (1 - stp)) / e - 1 if j > i + 1 else -stp
        if h[j] >= e * (1 + tgt):
            return max(o[j], e * (1 + tgt)) / e - 1 if j > i + 1 else tgt
    return c[min(i + days, len(c) - 1)] / e - 1


def short_term(df: pd.DataFrame, start: str) -> None:
    frames = [features(g) for _, g in df.groupby("symbol", sort=False)]
    cols = ("setup", "setup_vol", "above_all")
    res = {k: [] for k in cols + ("all",)}
    rng = np.random.default_rng(0)
    for g in frames:
        o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        dts = g.date.to_numpy()
        ok = np.flatnonzero(g.liq.to_numpy() & (dts >= np.datetime64(start)))
        sets = {k: first_signals(g, k) for k in cols}
        sets["all"] = rng.choice(ok, size=min(len(ok), 40), replace=False) if len(ok) else np.array([], int)
        for k, idx in sets.items():
            for i in idx:
                if dts[i] < np.datetime64(start) or i + 6 >= len(c):
                    continue
                e = o[i + 1]
                res[k].append(dict(
                    gap=e / c[i] - 1, d1_oc=c[i + 1] / e - 1, r1=c[i + 1] / e - 1, r2=c[i + 2] / e - 1,
                    r3=c[i + 3] / e - 1, r5=c[i + 5] / e - 1,
                    b32=bracket(o, h, l, c, i, 0.03, 0.02) - COST, b53=bracket(o, h, l, c, i, 0.05, 0.03) - COST))
    print(f"{'':52s} {'next-day open':>13s} {'1d':>6s} {'2d':>6s} {'3d':>6s} {'5d':>6s}   {'+3/-2 bracket':>16s}   {'+5/-3 bracket':>16s}")
    names = {"setup": "Setup", "setup_vol": "Setup + 5x-volume day", "above_all": "Above all 4 EMAs",
             "all": "Random liquid stock-days"}
    for k in cols + ("all",):
        t = pd.DataFrame(res[k])
        if t.empty:
            continue
        m = lambda s: f"{100 * t[s].mean():+5.2f}%"
        print(f"{names[k] + f' (n={len(t)})':52s} {m('gap'):>13s} {m('r1'):>6s} {m('r2'):>6s} {m('r3'):>6s} {m('r5'):>6s}"
              f"   {m('b32')} win {100 * (t.b32 > 0).mean():.0f}%   {m('b53')} win {100 * (t.b53 > 0).mean():.0f}%")
        print(f"{'':52s} {'':13s} {'win':>6s} " + " ".join(f"{100 * (t[s] > 0).mean():5.0f}%" for s in ("r2", "r3", "r5")))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "short":
        print("=" * 20, "F&O stocks, 2010 onward (averages; returns from next open)", "=" * 20)
        short_term(load_fo(), "2010-01-01")
        print("\n" + "=" * 20, "All liquid NSE stocks, Sep 2024 onward", "=" * 20)
        short_term(load_all(), "2024-09-01")
    else:
        print("=" * 30, "F&O stocks, Yahoo adjusted", "=" * 30)
        run(load_fo(), "2010-01-01")
        print("\n" + "=" * 30, "All liquid NSE stocks, bhavcopy", "=" * 30)
        run(load_all(), "2024-09-01")
