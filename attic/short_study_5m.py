"""
Re-price short_study.py's picks from a 09:20 entry with Groww 5-minute bars
(short at the 09:20 bar open, cover at the 15:10 bar close, 0.25% cost).

Run after short_study.py: ~/anaconda3/bin/python attic/short_study_5m.py 2>/dev/null
"""
from __future__ import annotations

import os

import pandas as pd

import gap_5m as G
from tab_audit import CACHE, load


def main() -> None:
    p = pd.read_parquet(os.path.join(CACHE, "short_study_picks.parquet"))
    d = load()[["symbol", "date", "open", "high", "low", "close"]]
    # picks are dated on the signal day; the trade is the next session
    nxt = d.assign(sig_date=d.groupby("symbol").date.shift(1)).dropna(subset=["sig_date"])
    t = p[["date", "symbol", "split"]].rename(columns={"date": "sig_date"}).merge(nxt, on=["symbol", "sig_date"])
    t = t.assign(trend="-", news="-")
    bars = G.get_bars(t)
    bars["date"] = pd.to_datetime(bars.date)
    r = G.price(t, bars).merge(t[["symbol", "date", "split"]], on=["symbol", "date"])
    print(f"\npriced {len(r)} of {len(t)} picks; positive = the short made money, after cost")
    print(f"median move from official open to 09:20: {100*r.drift_0920.median():+.2f}%\n")
    for name, g in [("all", r)] + list(r.groupby("split")):
        day = g.groupby("date").e0920.mean()
        print(f"  {name:<26} n={len(g):>5}  09:15 open: med {100*g.auction.median():+.2f}% avg {100*g.auction.mean():+.2f}%  |  "
              f"09:20: med {100*g.e0920.median():+.2f}% avg {100*g.e0920.mean():+.2f}% won {100*(g.e0920 > 0).mean():.0f}%  |  "
              f"book up on {100*(day > 0).mean():.0f}% of days, avg day {100*day.mean():+.2f}%")


if __name__ == "__main__":
    main()
