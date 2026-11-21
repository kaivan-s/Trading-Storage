"""
Stocks that keep opening higher and closing lower (RVNL-style): does the habit persist?

Pattern measures, from the last 10 / 20 sessions up to today's close:
  fades10   days that opened above the previous close and closed below the open
  oc20      average open->close move (the intraday part)
  gap20     average gap (the overnight part)
Outcome: next session's open->close, shorted, vs the typical stock that day
(cash shorts are intraday only), and the next 5 sessions as five separate
day-trades. Cost 0.25% per trade. Universe: price >= 50, 20-day median
turnover >= 5 cr, 60+ sessions of history.

Run: ~/anaconda3/bin/python attic/intraday_fade.py [SYMBOL] 2>/dev/null
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from tab_audit import COST, load


def build() -> tuple[pd.DataFrame, pd.Series]:
    d = load()
    g = d.groupby("symbol", sort=False)
    d["fade"] = ((d.gap > 0) & (d.oc < 0)).astype(float)
    roll = lambda col, n: g[col].transform(lambda s: s.rolling(n, min_periods=n).mean())
    d["red"] = (d.oc < 0).astype(float)
    d["red10"] = roll("red", 10) * 10
    d["red20"] = roll("red", 20) * 20
    d["red_streak"] = g.red.transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumsum()) * d.red
    d["fades10"] = roll("fade", 10) * 10
    d["fades20"] = roll("fade", 20) * 20
    d["oc20"] = roll("oc", 20)
    d["gap20"] = roll("gap", 20)
    d["oc_sd20"] = g.oc.transform(lambda s: s.rolling(20, min_periods=20).std())
    d["oc_z"] = d.oc20 / d.oc_sd20 * np.sqrt(20)
    uni = ((d.close >= 50) & (d.to20 >= 500) & (d.n_hist >= 60) & d.n_oc.notna()).fillna(False)
    d["bench"] = d.date.map(d[uni].groupby("date").n_oc.median())
    d["ex"] = d.bench - d.n_oc                      # short's edge over the typical stock
    for k in range(1, 6):
        d[f"oc_k{k}"] = g.oc.shift(-k)
        d[f"b_k{k}"] = d.date.map(d[uni].groupby("date")[f"oc_k{k}"].median()) if k > 1 else d.bench
    d.loc[uni, "b_k2"] = d.loc[uni, "b_k2"]
    return d, uni


def show_symbol(d: pd.DataFrame, sym: str) -> None:
    s = d[d.symbol == sym].tail(20)
    print(f"{sym}, last 20 sessions (gap = open vs previous close, oc = close vs open):")
    print(s[["date", "prev_close", "open", "close", "gap", "oc", "ret"]].assign(
        gap=lambda x: (100 * x.gap).round(2), oc=lambda x: (100 * x.oc).round(2),
        ret=lambda x: (100 * x.ret).round(2)).to_string(index=False))
    t = s.iloc[-1]
    print(f"closed below its open on {int(t.red20)} of the last 20 sessions "
          f"({int(t.red10)} of the last 10, current streak {int(t.red_streak)}); "
          f"opened higher and closed lower on {int(t.fades20)}; "
          f"average gap {100*t.gap20:+.2f}%, average open->close {100*t.oc20:+.2f}%\n")


def line(name: str, x: pd.DataFrame) -> None:
    if len(x) < 100:
        print(f"  {name:<40} n={len(x):>7}  (too few)")
        return
    ex, sh = x.ex, -x.n_oc - COST
    mid = x.date.min() + (x.date.max() - x.date.min()) / 2
    a, b = ex[x.date < mid], ex[x.date >= mid]
    five = sum(x[f"b_k{k}"] - x[f"oc_k{k}"] for k in range(1, 6)) / 5
    ok = ex.mean() > COST and a.mean() > COST and b.mean() > COST
    print(f"  {name:<40} n={len(x):>7}  next day: short avg {100*sh.mean():+.2f}% med {100*sh.median():+.2f}%  "
          f"vs typical avg {100*ex.mean():+.2f}% beat {100*(ex > 0).mean():.0f}%  halves {100*a.mean():+.2f}/{100*b.mean():+.2f}  |  "
          f"next 5 days, per day vs typical {100*five.mean():+.2f}%  {'PASS' if ok else ''}")


def main() -> None:
    d, uni = build()
    sym = sys.argv[1] if len(sys.argv) > 1 else "RVNL"
    show_symbol(d, sym)
    u = d[uni].copy()
    print(f"Pattern test, {u.date.min().date()} -> {u.date.max().date()}, {len(u):,} stock-days; "
          f"positive = shorting at the open beat the typical stock\n")
    red_next = lambda x: f"closed below open next day {100*(x.n_oc < 0).mean():.0f}%"
    print("By count of days that closed below the open (red candles):")
    line("all stock-days", u)
    print(f"    {red_next(u)}")
    for n in (6, 7, 8, 9):
        x = u[u.red10 >= n]
        line(f"10-day red count >= {n}", x)
        print(f"    {red_next(x)}")
    for n in (13, 15, 17):
        x = u[u.red20 >= n]
        line(f"20-day red count >= {n}", x)
        print(f"    {red_next(x)}")
    for n in (3, 4, 5, 6, 7):
        x = u[u.red_streak >= n]
        line(f"red streak >= {n} days in a row", x)
        print(f"    {red_next(x)}")
    x = u[u.fno & (u.red10 >= 7)]
    line("F&O, 10-day red count >= 7", x)
    print(f"    {red_next(x)}")
    print("\nBy count of 'opened higher, closed lower' days:")
    for n in (0, 2, 4, 5, 6, 7, 8):
        line(f"10-day count >= {n}" if n else "all stock-days", u[u.fades10 >= n])
    for n in (10, 12, 14):
        line(f"20-day count >= {n}", u[u.fades20 >= n])
    print("\nBy average open->close over 20 sessions (decile 1 = most negative):")
    q = pd.qcut(u.oc20, 10, labels=False, duplicates="drop")
    for k in range(10):
        line(f"decile {k + 1}", u[q == k])
    print("\nTug of war: gaps up on average AND falls on average after the open (20 sessions):")
    line("avg gap > 0 and avg open->close < 0", u[(u.gap20 > 0) & (u.oc20 < 0)])
    line("  ... strongest third of both", u[(u.gap20 > u.gap20.quantile(0.67)) & (u.oc20 < u.oc20.quantile(0.33))])
    line("  ... and opened higher again today", u[(u.gap20 > u.gap20.quantile(0.67)) & (u.oc20 < u.oc20.quantile(0.33)) & (u.n_gap > 0)])
    print("\nSame, F&O stocks only:")
    f = u[u.fno]
    line("F&O, 10-day count >= 6", f[f.fades10 >= 6])
    line("F&O, oc20 bottom decile", f[f.oc20 <= u.oc20.quantile(0.1)])


if __name__ == "__main__":
    main()
