#!/usr/bin/env python3
"""
Re-measure the three numbers in STAT_STRIPS.movers on the warmed-up window.

The shipped strip quotes "+9pp more likely to travel 3% intraday", "t=13,
holding on 87-92% of sessions" and a next-day return edge of "+0.02%,
p=0.73". Those came from a run whose gated pool was only alive on ~50 days,
so the comparison needs redoing on the 123 clean days.

Measures exactly what the copy claims: the shortlist (top 40 by score)
against the REST OF THE GATED POOL, not against a decile split and not
against the whole market.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import eval_tom
import fetch  # noqa: F401
import tom as tomscan

TOP_N = 40
MIN_POOL = 20


def clip01(s, lo, hi):
    return ((pd.to_numeric(s, errors="coerce") - lo) / (hi - lo)).clip(0, 1)


def tstat(v):
    v = pd.Series(v).dropna()
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
    return t, 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))), len(v)


def main() -> None:
    s = eval_tom.load(10 ** 6).sort_values(["symbol", "date"])
    g = s.groupby("symbol", sort=False)
    hi = "adj_high" if "adj_high" in s.columns else "high"
    s["reach"] = (g[hi].shift(-1) / s["adj"] - 1.0) * 100
    s["r1"] = (g["adj"].shift(-1) / s["adj"] - 1.0) * 100
    s["x1"] = s["r1"] - s.groupby("date")["r1"].transform("median")

    trig = pd.to_numeric(s["prior_trigger"], errors="coerce").fillna(
        pd.to_numeric(s["trigger"], errors="coerce"))
    s["to_trigger"] = trig / s["adj"].replace(0, np.nan) - 1.0
    s["gated"] = (
        (s["adj"] > s["ema50"]) & (s["ema50"] > s["ema200"])
        & s["to_trigger"].between(-tomscan.MOM_ZONE_ABOVE, tomscan.MOM_ZONE_BELOW)
        & (s["adj"] >= tomscan.MOM_MIN_PRICE)
        & (s["rsi"].fillna(50) <= tomscan.MOM_RSI_MAX)
    )
    W = tomscan.MOM_W
    r = s["rsi"].fillna(50)
    s_rsi = pd.Series(np.where(r <= 72, clip01(r, 55, 72), 1.0 - clip01(r, 72, 85)),
                      index=s.index).clip(0, 1)
    s["score"] = (W["vol"] * clip01(s["vol_ratio"].fillna(0), 0.8, 2.5)
                  + W["rsi"] * s_rsi
                  + W["atr"] * clip01(s["atr_pct"], 0.015, 0.06)
                  + W["ext"] * clip01(s["ext_ema20"], 0.0, 0.15)
                  + W["cmf"] * clip01(s["cmf"], 0.0, 0.25))

    pool = s.groupby("date")["gated"].transform("sum")
    d = s[(pool >= MIN_POOL) & s["r1"].notna() & s["gated"]].copy()
    d["rank"] = d.groupby("date")["score"].rank(ascending=False, method="first")
    top, rest = d[d["rank"] <= TOP_N], d[d["rank"] > TOP_N]

    L = "=" * 74
    print(f"\n{L}\nMOVERS STAT STRIP — shortlist vs rest of the gated pool\n{L}")
    print(f"{d['date'].nunique()} scan days, {d['date'].min():%d %b %Y} -> "
          f"{d['date'].max():%d %b %Y}")
    print(f"pool {len(d) / d['date'].nunique():.0f}/day, "
          f"shortlist {len(top) / d['date'].nunique():.0f}/day\n")

    print(f"{'threshold':<22}{'shortlist':>11}{'rest':>9}{'gap':>8}"
          f"{'t':>7}{'p':>9}{'days held':>11}")
    for thr in (1, 2, 3, 5):
        a = top.groupby("date").apply(lambda x: (x["reach"] >= thr).mean() * 100)
        b = rest.groupby("date").apply(lambda x: (x["reach"] >= thr).mean() * 100)
        j = a.index.intersection(b.index)
        diff = a.loc[j] - b.loc[j]
        t, p, _ = tstat(diff)
        print(f"reached +{thr}% intraday{'':<4}{a.mean():>10.1f}%{b.mean():>8.1f}%"
              f"{diff.mean():>+7.1f}{t:>7.1f}{p:>9.4f}"
              f"{100 * (diff > 0).mean():>10.0f}%")

    print()
    for label, col in (("next-day return edge", "x1"),):
        a = top.groupby("date")[col].median()
        b = rest.groupby("date")[col].median()
        j = a.index.intersection(b.index)
        t, p, _ = tstat(a.loc[j] - b.loc[j])
        print(f"{label:<22}{a.median():>+10.2f}%{b.median():>+8.2f}%"
              f"{(a.loc[j] - b.loc[j]).median():>+7.2f}{t:>7.1f}{p:>9.4f}"
              f"{100 * ((a.loc[j] - b.loc[j]) > 0).mean():>10.0f}%")
    print("\n(return edge is median excess vs the all-stock median, per day)")


if __name__ == "__main__":
    main()
