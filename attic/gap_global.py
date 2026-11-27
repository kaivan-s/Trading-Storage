"""
Gap-Ups in Downtrends: does an overnight global move in a stock's driver
(copper for metals, Nasdaq for IT, crude for oil, gold for jewellery) change
how the gap behaves? Same signals and trade as gap_traits.py (short 09:20,
cover 15:15, after cost; positive = the short made money).

Overnight move for Indian session D = the driver's last close before D
versus the close before that (the US session that ended in the early hours
of D, IST). Also splits every candidate by the overnight S&P 500 move.

Run: ~/anaconda3/bin/python attic/gap_global.py 2>/dev/null
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_traits import build  # noqa: E402
from tab_audit import CACHE  # noqa: E402

DRIVERS = {
    "Copper (non-ferrous)": ("HG=F", ["Aluminium", "Aluminium Copper & Zinc Products", "Copper", "Zinc",
                                      "Diversified Metals", "Trading - Metals"]),
    "Copper (steel, proxy)": ("HG=F", ["Iron & Steel", "Iron & Steel Products", "Sponge Iron", "Pig Iron"]),
    "Nasdaq (IT)": ("^IXIC", ["IT Enabled Services", "Computers - Software & Consulting", "Software Products",
                              "Computers Hardware & Equipments"]),
    "Crude (oil upstream)": ("CL=F", ["Oil Exploration & Production", "Oil Equipment & Services"]),
    "Gold (jewellery, precious)": ("GC=F", ["Gems Jewellery And Watches", "Precious Metals"]),
}
BROAD = "^GSPC"


def overnight(ticker: str, days: pd.Series) -> pd.Series:
    c = yf.download(ticker, start="2023-11-01", progress=False, auto_adjust=False)["Close"].squeeze().dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None)
    ret = c.pct_change()
    pos = c.index.searchsorted(days.values) - 1
    return pd.Series(np.where(pos >= 1, ret.values[np.clip(pos, 0, None)], np.nan), index=days.index)


def line(name: str, s: pd.DataFrame, mid) -> None:
    if len(s) < 25:
        print(f"  {name:<34} n={len(s):>5}  (too few)")
        return
    a, b = s[s.date < mid].ret, s[s.date >= mid].ret
    print(f"  {name:<34} n={len(s):>5}  median {100*s.ret.median():+5.2f}%  avg {100*s.ret.mean():+5.2f}%  "
          f"win {100*(s.ret > 0).mean():3.0f}%  halves {100*a.median():+.2f} / {100*b.median():+.2f}")


def main() -> None:
    r = build()
    smap = pd.read_json(os.path.join(CACHE, "sector_map.json"), orient="index")["basic_industry"]
    r["industry"] = r.symbol.map(smap)
    mid = r.date.min() + (r.date.max() - r.date.min()) / 2
    print(f"{len(r)} downtrend gap-ups; positive = the short made money\n")

    r["spx"] = overnight(BROAD, r.date)
    print("every candidate, by the overnight S&P 500 move")
    for lo, hi in [(-1, -0.01), (-0.01, 0), (0, 0.01), (0.01, 1)]:
        line(f"S&P {lo*100:+.0f}%..{hi*100:+.0f}%", r[(r.spx >= lo) & (r.spx < hi)], mid)
    print("  same, only mornings that were not a broad gap-up open")
    q = r[r.mkt_gap < 0.0075]
    for lo, hi in [(-1, 0), (0, 0.01), (0.01, 1)]:
        line(f"  S&P {lo*100:+.0f}%..{hi*100:+.0f}%", q[(q.spx >= lo) & (q.spx < hi)], mid)

    linked = []
    for name, (tk, inds) in DRIVERS.items():
        s = r[r.industry.isin(inds)].copy()
        s["drv"] = overnight(tk, s.date)
        linked.append(s.assign(group=name))
        print(f"\n{name} [{tk}], {len(s)} gap-ups")
        line("driver up >= 1% overnight", s[s.drv >= 0.01], mid)
        line("driver -1%..+1%", s[s.drv.between(-0.01, 0.01, inclusive="left")], mid)
        line("driver down <= -1%", s[s.drv < -0.01], mid)

    s = pd.concat(linked)
    print("\nall driver-linked sectors pooled")
    line("driver up >= 1% (gap has a global reason)", s[s.drv >= 0.01], mid)
    line("driver up >= 2%", s[s.drv >= 0.02], mid)
    line("driver flat or down", s[s.drv < 0.01], mid)
    line("not driver-linked (other industries)", r[~r.symbol.isin(s.symbol) | ~r.date.isin(s.date)], mid)
    t = s[s.mkt_gap < 0.0075]
    line("  driver up >= 1%, not a broad open", t[t.drv >= 0.01], mid)
    line("  driver flat/down, not a broad open", t[t.drv < 0.01], mid)

    r = r.sort_values(["date", "gap"], ascending=[True, False])
    top = r[r.groupby("date").cumcount() < 5]
    k = top.merge(s[["date", "symbol", "drv"]], on=["date", "symbol"], how="left")
    print("\ntop 5 by gap only")
    line("driver-linked, driver up >= 1%", k[k.drv >= 0.01], mid)
    line("driver-linked, driver flat/down", k[k.drv < 0.01], mid)
    line("everything else in the top 5", k[k.drv.isna() | (k.drv < 0.01)], mid)
    for yr, g in k[k.drv >= 0.01].groupby(k.date.dt.year):
        line(f"  driver up >= 1%, {yr}", g, mid)


if __name__ == "__main__":
    main()
