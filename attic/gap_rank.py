"""
Gap-Ups in Downtrends: does a different ranking pick a better top five than
gap size? Same signals and trade as gap_traits.py (short 09:20, cover 15:15,
after cost; positive = the short made money).

Rankings known by 09:08 (when the list is published): gap, depth below the
50-day average, prior 20-day return, overnight filing. Rankings that need the
first 5-minute bar (early volume, 09:20 vs open) are shown separately because
the list would have to be re-ranked at 09:20.

Run: ~/anaconda3/bin/python attic/gap_rank.py 2>/dev/null
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_traits import build  # noqa: E402

TOP = 5


def pick(r: pd.DataFrame, key: pd.Series, ascending: bool) -> pd.DataFrame:
    return (r.assign(_k=key).sort_values(["date", "_k", "gap"], ascending=[True, ascending, False])
            .groupby("date").head(TOP))


def line(name: str, s: pd.DataFrame, mid) -> None:
    a, b = s[s.date < mid].ret, s[s.date >= mid].ret
    yrs = " ".join(f"{y}:{100*g.ret.median():+.2f}" for y, g in s.groupby(s.date.dt.year) if len(g) >= 30)
    print(f"  {name:<38} n={len(s):>5}  median {100*s.ret.median():+5.2f}%  avg {100*s.ret.mean():+5.2f}%  "
          f"win {100*(s.ret > 0).mean():3.0f}%  blow-ups {100*(s.ret < -0.05).mean():4.1f}%  "
          f"halves {100*a.median():+.2f} / {100*b.median():+.2f}  | {yrs}")


def main() -> None:
    r = build()
    r["filed"] = (r.news == "yes").astype(int)
    mid = r.date.min() + (r.date.max() - r.date.min()) / 2
    print(f"{len(r)} candidates on {r.date.nunique()} sessions; top {TOP} per session under each ranking\n")

    print("baseline")
    line("every candidate (no ranking)", r, mid)

    print("\nknown by 09:08")
    line("largest gap (today's rule)", pick(r, r.gap, False), mid)
    line("smallest gap", pick(r, r.gap, True), mid)
    line("deepest below 50-day avg", pick(r, r.depth, True), mid)
    line("shallowest below 50-day avg", pick(r, r.depth, False), mid)
    line("weakest prior 20 days", pick(r, r.ret20, True), mid)
    line("strongest prior 20 days", pick(r, r.ret20, False), mid)
    line("overnight filing first, then gap", pick(r, -r.filed, True), mid)

    print("\nneeds the 09:15 bar (re-rank at 09:20)")
    line("heaviest early volume", pick(r, r.early_vol, False), mid)
    line("lightest early volume", pick(r, r.early_vol, True), mid)
    line("most extended at 09:20", pick(r, r.drift, False), mid)
    line("most faded by 09:20", pick(r, r.drift, True), mid)

    print("\nsame rankings on mornings that were not a broad gap-up open (median open < +0.75%)")
    q = r[r.mkt_gap < 0.0075]
    line("largest gap (today's rule)", pick(q, q.gap, False), mid)
    line("overnight filing first, then gap", pick(q, -q.filed, True), mid)
    line("heaviest early volume", pick(q, q.early_vol, False), mid)
    line("filing first, then early volume", pick(q, -q.filed * 10 - q.early_vol.rank(pct=True), True), mid)


if __name__ == "__main__":
    main()
