"""
Three tested patterns for stocks that keep falling, judged by one bar set in
advance: a short must beat the typical stock on AVERAGE after costs, and do so
in both halves of the period. Win rate alone doesn't count.

Every trade enters at the next session's open and is measured to the close H
sessions later. "vs typical" is the short's return minus the median stock's
short over the same window and universe. Each symbol gets one signal per H
sessions (cooldown), so a stock sitting in a list isn't counted every day.

1  Weak results keep falling (post-results drift).
   Cash market, results filings from data/cache/filings_hist (Apr 2024 - Oct 2026).
   Reaction = the results day's return minus the median stock's. A filing at or
   after 14:30 is treated as reacting the next session.
2  The weakest stocks keep underperforming (12-1 and 6-1 momentum, monthly).
   F&O stocks 2010-26 (Yahoo, today's list: survivors, which flatters longs and
   hurts shorts) and all liquid cash stocks 2024-26.
3  Support break on heavy volume (close 1%+ below an unbroken swing low on
   2x / 3x+ the prior 20-day median volume). F&O stocks can be shorted overnight
   with futures, so the full move is capturable there.

Run: ~/anaconda3/bin/python attic/falling_patterns.py [1|2|3] 2>/dev/null
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from ema_cross_breakout import load_fo
from price_action import per_symbol
from tab_audit import CACHE, COST, load

FILINGS = os.path.join(CACHE, "filings_hist")


# --------------------------------------------------------------------------
# shared measurement
# --------------------------------------------------------------------------

def add_forward(d: pd.DataFrame, horizons, close="close", open_="open") -> pd.DataFrame:
    g = d.groupby("symbol", sort=False)
    o1 = g[open_].shift(-1)
    d["rank"] = g.cumcount()
    for h in horizons:
        d[f"fwd{h}"] = g[close].shift(-h) / o1 - 1
    return d


def benchmark(d: pd.DataFrame, uni: pd.Series, horizons) -> pd.DataFrame:
    for h in horizons:
        d[f"bench{h}"] = d.date.map(d[uni].groupby("date")[f"fwd{h}"].median())
    return d


def cooldown(ev: pd.DataFrame, h: int) -> pd.DataFrame:
    ev = ev.sort_values(["symbol", "date"])
    keep, last = [], {}
    for i, s, r in zip(ev.index, ev.symbol, ev["rank"]):
        if r - last.get(s, -10**9) >= h:
            keep.append(i)
            last[s] = r
    return ev.loc[keep]


def report(name: str, ev: pd.DataFrame, horizons, cool=True) -> None:
    print(f"  {name}")
    for h in horizons:
        e = ev.dropna(subset=[f"fwd{h}", f"bench{h}"])
        if cool:
            e = cooldown(e, h)
        if len(e) < 30:
            print(f"    {h:>3}d  n={len(e):>5}  (too few)")
            continue
        short = -e[f"fwd{h}"] - COST
        ex = e[f"bench{h}"] - e[f"fwd{h}"]
        mid = e.date.min() + (e.date.max() - e.date.min()) / 2
        a, b = ex[e.date < mid], ex[e.date >= mid]
        m = ex.groupby(e.date.dt.to_period("M")).mean()
        ok = ex.mean() - COST > 0 and a.mean() - COST > 0 and b.mean() - COST > 0
        print(f"    {h:>3}d  n={len(e):>5}  short avg {100*short.mean():+6.2f}% med {100*short.median():+6.2f}%  |  "
              f"vs typical avg {100*ex.mean():+6.2f}% med {100*ex.median():+6.2f}% beat {100*(ex > 0).mean():3.0f}%  |  "
              f"halves {100*a.mean():+5.2f}% / {100*b.mean():+5.2f}%  months+ {100*(m > 0).mean():3.0f}% of {len(m)}"
              f"  {'PASS' if ok else ''}")


# --------------------------------------------------------------------------
# 1  weak results
# --------------------------------------------------------------------------

def results_events(dates) -> set:
    nxt = {a: b for a, b in zip(dates[:-1], dates[1:])}
    out = set()
    for day in dates:
        p = os.path.join(FILINGS, f"{day:%Y-%m-%d}.json")
        if not os.path.exists(p):
            continue
        for sym, rows in json.load(open(p)).items():
            for r in rows:
                if (r.get("category") or "") != "Results":
                    continue
                late = not r.get("after_close") and (r.get("time") or "00:00") >= "14:30"
                rd = nxt.get(day) if late else day
                if rd is not None:
                    out.add((rd, sym))
                break
    return out


def test_results() -> None:
    H = (5, 20, 40, 60)
    d = load()
    d["open_a"] = d.open * d.A / d.close
    d = add_forward(d, H, close="A", open_="open_a")
    uni = ((d.close >= 50) & (d.to20 >= 500) & (d.n_hist >= 60)).fillna(False)
    d = benchmark(d, uni, H)
    d["react"] = d.ret - d.date.map(d[uni].groupby("date").ret.median())
    dates = sorted(d.date.unique())
    ev = results_events(dates)
    have = {pd.Timestamp(f[:10]) for f in os.listdir(FILINGS)}
    u = d[uni & d.date.isin(have)].copy()
    u["results"] = [k in ev for k in zip(u.date, u.symbol)]
    print(f"TEST 1  Weak results keep falling  ({u.date.min().date()} -> {u.date.max().date()}, "
          f"{u.results.sum()} results days)\n  reaction = results-day return minus the typical stock's\n")
    r = u[u.results]
    for lab, m in [("reaction -8% or worse", r.react <= -0.08),
                   ("reaction -8% to -4%", r.react.between(-0.08, -0.04, inclusive="right")),
                   ("reaction -4% to 0", r.react.between(-0.04, 0, inclusive="right")),
                   ("reaction 0 to +4%", r.react.between(0, 0.04, inclusive="right")),
                   ("reaction +4% or better", r.react > 0.04)]:
        report(lab, r[m], H)
    print()
    report("control: -4% or worse WITHOUT results", u[~u.results & (u.react <= -0.04)], H)
    print()


# --------------------------------------------------------------------------
# 2  weakest stocks (momentum losers)
# --------------------------------------------------------------------------

def momentum(d: pd.DataFrame, uni: pd.Series, label: str, n_bottom) -> None:
    H = 21
    g = d.groupby("symbol", sort=False)
    d["m12"] = g.close.shift(21) / g.close.shift(252) - 1
    d["m6"] = g.close.shift(21) / g.close.shift(126) - 1
    d["fwd"] = g.close.shift(-H) / g.open.shift(-1) - 1
    first = d.groupby(d.date.dt.to_period("M")).date.transform("min")
    m = d[uni & (d.date == first) & d.fwd.notna()].copy()
    print(f"  {label}: {m.date.min().date()} -> {m.date.max().date()}")
    for sig in ("m12", "m6"):
        rows = []
        for dt, x in m.dropna(subset=[sig]).groupby("date"):
            if len(x) < 40:
                continue
            k = n_bottom if isinstance(n_bottom, int) else max(10, int(len(x) * n_bottom))
            x = x.sort_values(sig)
            bench = x.fwd.median()
            rows.append({"date": dt, "bot": bench - x.fwd.head(k).mean(), "top": x.fwd.tail(k).mean() - bench,
                         "bot_raw": -x.fwd.head(k).mean() - COST, "k": k})
        r = pd.DataFrame(rows)
        if r.empty:
            continue
        half = len(r) // 2
        a, b = r.bot.iloc[:half], r.bot.iloc[half:]
        ok = r.bot.mean() - COST > 0 and a.mean() - COST > 0 and b.mean() - COST > 0
        print(f"    {sig.replace('m', '')}-1 momentum, weakest {r.k.iloc[-1]} each month, {len(r)} months: "
              f"short vs typical avg {100*r.bot.mean():+.2f}%/month, beat in {100*(r.bot > 0).mean():.0f}% of months, "
              f"halves {100*a.mean():+.2f}% / {100*b.mean():+.2f}%  |  short raw avg {100*r.bot_raw.mean():+.2f}%  |  "
              f"(strongest, long vs typical: {100*r.top.mean():+.2f}%, beat {100*(r.top > 0).mean():.0f}%)"
              f"  {'PASS' if ok else ''}")
    print()


def test_losers() -> None:
    print("TEST 2  The weakest stocks keep underperforming (rebalanced monthly, hold 21 sessions)\n")
    fo = load_fo().sort_values(["symbol", "date"]).reset_index(drop=True)
    fo = fo[fo.date >= "2009-01-01"]
    to20 = fo.groupby("symbol").turnover_cr.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    momentum(fo, ((fo.close >= 50) & (to20 >= 10)).fillna(False), "F&O stocks", 20)
    d = load()
    d["open_a"] = d.open * d.A / d.close
    d = d.rename(columns={"open": "open_raw", "close": "close_raw"}).rename(columns={"A": "close", "open_a": "open"})
    uni = ((d.close_raw >= 50) & (d.to20 >= 500) & (d.n_hist >= 60)).fillna(False)
    momentum(d, uni, "All liquid cash stocks", 0.1)


# --------------------------------------------------------------------------
# 3  support break on heavy volume
# --------------------------------------------------------------------------

def breaks(df: pd.DataFrame, label: str, liq_cr: float) -> None:
    H = (5, 10, 20)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    parts = [per_symbol(g) for _, g in df.groupby("symbol", sort=False)]
    x = pd.concat([p for p in parts if p is not None], ignore_index=True)[["symbol", "date", "Support break"]]
    df = df.merge(x, on=["symbol", "date"], how="left")
    df["Support break"] = df["Support break"].fillna(False).astype(bool)
    g = df.groupby("symbol", sort=False)
    df["vol_x"] = df.volume / g.volume.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    to20 = g.turnover_cr.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    df = add_forward(df, H)
    price = df.raw_close if "raw_close" in df else df.close
    uni = ((price >= 50) & (to20 >= liq_cr)).fillna(False)
    df = benchmark(df, uni, H)
    u = df[uni]
    print(f"  {label}: {u.date.min().date()} -> {u.date.max().date()}")
    report("support break, any volume", u[u["Support break"]], H)
    report("support break, 2-3x volume", u[u["Support break"] & u.vol_x.between(2, 3, inclusive="left")], H)
    report("support break, 3x+ volume", u[u["Support break"] & (u.vol_x >= 3)], H)
    report("control: 3x+ volume down day 3%+, no break", u[~u["Support break"] & (u.vol_x >= 3)
                                                        & (u.close / g.close.shift(1).loc[u.index] <= 0.97)], H)
    print()


def test_breaks() -> None:
    print("TEST 3  Support break on heavy volume\n")
    fo = load_fo()
    breaks(fo[fo.date >= "2009-01-01"], "F&O stocks (shortable overnight via futures)", 10)
    from price_action import load_bhav
    b = load_bhav()
    vol = pd.concat([pd.read_parquet(f, columns=["symbol", "series", "date", "volume"])
                     for f in sorted(__import__("glob").glob(os.path.join(CACHE, "..", "raw", "bhav_*.parquet")))])
    vol = vol[vol.series == "EQ"].drop(columns="series")
    vol["date"] = pd.to_datetime(vol.date)
    b = b.merge(vol.drop_duplicates(["symbol", "date"]), on=["symbol", "date"])
    breaks(b, "All liquid cash stocks (intraday-only shorts in practice)", 5)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    pd.set_option("display.width", 250)
    if "1" in which:
        test_results()
    if "2" in which:
        test_losers()
    if "3" in which:
        test_breaks()
