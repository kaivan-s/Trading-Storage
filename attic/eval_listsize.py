#!/usr/bin/env python3
"""
How short can the daily list be, and does shortening it pay?

63 coils plus 81 leaders is not a shortlist, it is a screener. Three questions:

  1. How many INDEPENDENT decisions is each list really? A list of 40 names in
     6 sectors is 6 bets, not 40.
  2. Does cutting to the top N improve the per-name edge, or just shrink the
     sample? Only worth cutting if the kept names measure better.
  3. Can one ranked list replace both? The coil score does not order
     outcomes, but 12-1 momentum does -- so ranking the COILS by momentum,
     or the union of both pools, may beat either tab alone.

VERDICT: show ~20, ranked by 12-1 momentum, drawn from the coil pool.

Coils ranked by momentum, top 20, measured +0.78% at a 7-session hold
(NW t=3.37) and +1.30% at 10 (t=4.21, 84% of days positive) -- roughly double
the unfiltered coil pool's +0.47%/+0.83%, and better than the leaders list's
+0.95%/+1.15%, on a third of the names. It is also the most stable
construction tested, the only one whose halves are both solidly positive and
of comparable size (+0.94% / +0.59%).

Cutting BELOW 20 breaks it, which is the part that matters for product
decisions. Top 10 goes +1.98% / +0.12% across halves and top 5 flips sign
(+1.68% / -0.21%). The headline +2.03% on "leaders top 5" is first-half only
(+3.19% / +0.17%) -- with five names a day the daily median is too unstable to
trust, and any single-digit list will produce numbers like that. So "fewer
names is better" holds down to about 20 and is wrong below it.

Two incidental findings:
  - Ranking the UNION of both pools by momentum exactly reproduces ranking the
    leaders alone, because the top names by momentum are top-decile by
    construction. A union list adds nothing over the leaders list.
  - These lists are NOT concentrated: 20 names spread across 14 sectors. The
    "the sector is the position" framing that applies to Setups does not
    transfer here, so 20 rows really is closer to 20 decisions than to 5.

Caveat, the same one as everywhere else in this repo: 67 measurable sessions,
roughly 9 independent 7-day periods, and eleven constructions were compared.
The stability split is what makes the top-20 result worth acting on; it is not
enough to advertise a number from.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import eval_reversal as ev
import eval_strategies as es


def stats(sel, G, dates, h, window):
    """Day-paired median excess with a Newey-West t, plus list geometry."""
    exc = (G["fwd"][h] - G["bench"][h]) * 100
    ok = np.isfinite(exc)
    per_day, sizes = [], []
    for i in range(len(dates)):
        if not window[i]:
            continue
        r = sel[i] & ok[i]
        sizes.append(int(sel[i].sum()))
        if r.sum() >= 2:
            per_day.append(np.median(exc[i][r]))
    if len(per_day) < 12:
        return None
    ser = pd.Series(per_day)
    t, _ = ev.nw_t(ser, h)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if np.isfinite(t) else np.nan
    return {"med": ser.median(), "t": t, "p": p, "win": 100 * (ser > 0).mean(),
            "n": len(ser), "size": np.mean(sizes) if sizes else 0}


def top_n(score, pool, n):
    """Keep the n highest-scoring names in `pool` each day."""
    sc = np.where(pool, score, -np.inf)
    out = np.zeros_like(pool, dtype=bool)
    for i in range(sc.shape[0]):
        k = min(n, int(np.isfinite(sc[i]).sum()), int(pool[i].sum()))
        if k <= 0:
            continue
        idx = np.argpartition(-sc[i], k - 1)[:k]
        idx = idx[np.isfinite(sc[i][idx])]
        out[i, idx] = True
    return out


def sectors_per_day(sel, sec_codes, window):
    """Mean distinct sectors represented, on days the list is non-empty."""
    counts = []
    for i in range(sel.shape[0]):
        if not window[i] or not sel[i].any():
            continue
        counts.append(len(np.unique(sec_codes[sel[i]])))
    return np.mean(counts) if counts else np.nan


def main() -> None:
    G, s, like = es.load()
    S = es.strategies(G, s, like)
    dates = like.index.to_numpy()

    coil = np.nan_to_num(S["OURS: Coiled Bases"][0], nan=0).astype(bool)
    lead = S["Mom 12-1 + uptrend"][0]
    live = coil.any(axis=1) & lead.any(axis=1)

    sec = s.groupby("symbol")["sector"].last().reindex(like.columns)
    sec_codes = pd.factorize(sec.fillna("?"))[0]

    mom = np.nan_to_num(G["mom250_ex20"].values, nan=-np.inf)
    union = coil | lead

    L = "=" * 88
    print(f"\n{L}\nHOW MANY NAMES, AND IS SHORTER BETTER?\n{L}")
    print(f"  {live.sum()} sessions where both pools exist. Excess vs the liquid")
    print("  universe median, day-paired, Newey-West t.\n")

    cands = [
        ("Setups pool: all coils", coil, None),
        ("Leaders: all qualifying", lead, None),
        ("Leaders: top 20 by mom", top_n(mom, lead, 20), None),
        ("Leaders: top 10 by mom", top_n(mom, lead, 10), None),
        ("Leaders: top 5 by mom", top_n(mom, lead, 5), None),
        ("Coils ranked by mom: top 20", top_n(mom, coil, 20), None),
        ("Coils ranked by mom: top 10", top_n(mom, coil, 10), None),
        ("Coils ranked by mom: top 5", top_n(mom, coil, 5), None),
        ("UNION ranked by mom: top 20", top_n(mom, union, 20), None),
        ("UNION ranked by mom: top 10", top_n(mom, union, 10), None),
        ("UNION ranked by mom: top 5", top_n(mom, union, 5), None),
    ]

    for h in (7, 10):
        print(f"  --- {h}-session hold " + "-" * 62)
        print(f"  {'list':<30}{'shown':>7}{'sectors':>9}{'med':>8}"
              f"{'NW t':>7}{'p':>7}{'days won':>10}{'days':>6}")
        for lab, sel, _ in cands:
            r = stats(sel, G, dates, h, live)
            if r is None:
                print(f"  {lab:<30}{'too few days':>7}")
                continue
            sd = sectors_per_day(sel, sec_codes, live)
            print(f"  {lab:<30}{r['size']:>7.0f}{sd:>9.1f}{r['med']:>+7.2f}%"
                  f"{r['t']:>7.2f}{r['p']:>7.3f}{r['win']:>9.0f}%{r['n']:>6}")
        print()

    halves(cands, G, dates, live)


def halves(cands, G, dates, live):
    """
    Split the measurable window in two. A construction picked out of eleven
    candidates needs to hold in both halves before it means anything.
    """
    idx = np.where(live)[0]
    mid = idx[len(idx) // 2]
    first = live.copy(); first[mid:] = False
    second = live.copy(); second[:mid] = False

    print("=" * 88)
    print("STABILITY: DOES IT HOLD IN BOTH HALVES? (7-session)")
    print("=" * 88)
    print(f"  {'list':<30}{'1st half':>11}{'2nd half':>11}{'agree':>8}")
    for lab, sel, _ in cands:
        a = stats(sel, G, dates, 7, first)
        b = stats(sel, G, dates, 7, second)
        if a is None or b is None:
            print(f"  {lab:<30}{'too few days':>11}")
            continue
        agree = "yes" if (a["med"] > 0) == (b["med"] > 0) else "NO"
        print(f"  {lab:<30}{a['med']:>+10.2f}%{b['med']:>+10.2f}%{agree:>8}")
    print()


if __name__ == "__main__":
    main()
