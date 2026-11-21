"""
Do gap-ups with news fade less than gap-ups without news?

Same gap rule and benchmark as tab_audit.py (open 2-9% up, non-F&O, price >= 50,
turnover >= 10 cr; trend split on the prior close vs its 50-day average).
"News" = a medium/high-importance NSE filing for that symbol after the previous
close and before 09:15 on the gap day (data/cache/filings_hist). Days with no
filings file are dropped rather than counted as "no news".

Profit is the fade: short open -> close, so positive = the stock fell.

Run: ~/anaconda3/bin/python attic/gap_news.py 2>/dev/null
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from tab_audit import CACHE, COST, excess, halves, load

FILINGS = os.path.join(CACHE, "filings_hist")


def news_flags(days) -> tuple[dict, set]:
    out, have = {}, set()
    for day in days:
        p = os.path.join(FILINGS, f"{day:%Y-%m-%d}.json")
        if not os.path.exists(p):
            continue
        have.add(day)
        for sym, rows in json.load(open(p)).items():
            pre = [r for r in rows if r.get("after_close") or (r.get("time") or "99:99") < "09:15"]
            if pre:
                out[(day, sym)] = {
                    "high": any(r["importance"] == "high" for r in pre),
                    "cat": pre[0].get("category") or "",
                }
    return out, have


def summary(name, d, m, liq):
    oc = -d.loc[m, "oc"]
    if len(oc) < 10:
        print(f"  {name:<34} n={len(oc):>4}  (too few)")
        return
    ex = excess(d, m, "oc", liq, -1)
    h = halves(d, ex)
    print(f"  {name:<34} n={len(oc):>4}  fell {100*(oc > 0).mean():4.0f}%  "
          f"median fade {100*oc.median():+5.2f}%  after cost {100*(oc.median()-COST):+5.2f}%  "
          f"vs typical stock {100*ex.median():+5.2f}%  halves {h}")


def main() -> None:
    d = load()
    liq = (d.to20 >= 100) & (d.close >= 10) & (d.n_hist >= 30)
    flags, have = news_flags(sorted(d.date.unique()))
    d = d[d.date.isin(have)].copy()
    liq = liq.loc[d.index]
    key = list(zip(d.date, d.symbol))
    d["news"] = [k in flags for k in key]
    d["news_high"] = [flags.get(k, {}).get("high", False) for k in key]
    d["news_cat"] = [flags.get(k, {}).get("cat", "") for k in key]
    print(f"{d.date.min().date()} -> {d.date.max().date()}, {d.date.nunique()} sessions with filings data\n")

    base = (d.gap.between(0.02, 0.09, inclusive="left") & ~d.fno & (d.prev_close >= 50)
            & (d.to20 >= 1000) & d.oc.notna()).fillna(False)
    for trend, tm in [("Downtrend (the tab)", d.A_prev < d.sma50_prev),
                      ("Uptrend", d.A_prev >= d.sma50_prev),
                      ("All gap-ups", pd.Series(True, index=d.index))]:
        m = base & tm.fillna(False)
        print(trend)
        summary("all", d, m, liq)
        summary("no news", d, m & ~d.news, liq)
        summary("news before open", d, m & d.news, liq)
        summary("  high-importance news", d, m & d.news_high, liq)
        summary("  normal news only", d, m & d.news & ~d.news_high, liq)
        print()

    m = base & (d.A_prev < d.sma50_prev).fillna(False) & d.news
    print("Downtrend + news, by filing category (n >= 10):")
    for cat, g in d[m].groupby("news_cat"):
        if len(g) >= 10:
            oc = -g.oc
            print(f"  {cat[:32]:<32} n={len(g):>3}  fell {100*(oc > 0).mean():3.0f}%  median fade {100*oc.median():+5.2f}%")


if __name__ == "__main__":
    main()
