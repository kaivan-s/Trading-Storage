"""
Which stocks open higher and then close below the open, by how much, and does
it persist? Railway names vs everything else.

For every gap-up day (open >= GAP_MIN above previous close):
  red%      share of those days closing below the open
  avg o->c  open-to-close return (negative = faded)
  >=3% fade share of days closing 3%+ below the open
  max dn    average low vs open (best case for a short from the open)
  max up    average high vs open (what a short from the open had to sit through)

Persistence: the first half of the window picks stocks, the second half grades them.

    Run:  ~/anaconda3/bin/python attic/gapup_fade_stocks.py 2>/dev/null
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intraday_backtest import load  # noqa: E402

GAP_MIN = 0.005
MIN_TURNOVER_CR = 10
MIN_PRICE = 50
MIN_DAYS = 20
COST = 0.001
RAILWAY = ["IRFC", "RVNL", "IRCON", "RAILTEL", "IRCTC", "RITES", "TITAGARH", "JWL",
           "TEXRAIL", "BEML", "KERNEX", "CONCOR", "HBLENGINE", "ORIENTRAIL", "RKFORGE"]


def per_stock(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "days": len(g),
        "red%": 100 * (g.oc < 0).mean(),
        "avg_oc%": 100 * g.oc.mean(),
        "med_oc%": 100 * g.oc.median(),
        "fade3%": 100 * (g.oc <= -0.03).mean(),
        "avg_gap%": 100 * g.gap.mean(),
        "max_dn%": 100 * g.dn.mean(),
        "max_up%": 100 * g.up.mean(),
    })


def main() -> None:
    df = load()
    g = df.groupby("symbol", sort=False)
    df["med_to"] = g["turnover"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).median()) / 100
    df["gap"] = df.open / df.prev_close - 1
    df["oc"] = df.close / df.open - 1
    df["dn"] = df.low / df.open - 1
    df["up"] = df.high / df.open - 1
    # ETFs' first trade of the day is often a stray print away from NAV, which
    # looks like a gap that "fades" every day.
    etf = df.symbol.str.contains(r"ETF|BEES|^SETF|GOLD|SILVER|LIQUID|CASE$|NIFTY|SENSEX|IETF", regex=True)
    ups = df[(df.gap >= GAP_MIN) & (df.prev_close >= MIN_PRICE) & (df.med_to >= MIN_TURNOVER_CR) & ~etf]
    start, end = ups.date.min(), ups.date.max()
    mid = start + (end - start) / 2
    print(f"window {start.date()} -> {end.date()}, split at {mid.date()}; "
          f"{ups.symbol.nunique()} liquid stocks with gap-up days\n")

    base = per_stock(ups)
    print("All liquid stocks, every gap-up day:")
    print("  " + "  ".join(f"{k} {v:.1f}" for k, v in base.items()), "\n")

    stats = ups.groupby("symbol").apply(per_stock)
    stats = stats[stats.days >= MIN_DAYS]
    h1 = ups[ups.date < mid].groupby("symbol").apply(per_stock)
    h2 = ups[ups.date >= mid].groupby("symbol").apply(per_stock)

    pd.set_option("display.width", 220)
    rail = stats.loc[stats.index.intersection(RAILWAY)].copy()
    rail["red% H1"] = h1["red%"].reindex(rail.index)
    rail["red% H2"] = h2["red%"].reindex(rail.index)
    rail["avg_oc% H1"] = h1["avg_oc%"].reindex(rail.index)
    rail["avg_oc% H2"] = h2["avg_oc%"].reindex(rail.index)
    print("Railway stocks (gap-up days):")
    print(rail.sort_values("avg_oc%").round(1).to_string(), "\n")
    rail_days = ups[ups.symbol.isin(RAILWAY)]
    print("Railway group pooled:  " + "  ".join(f"{k} {v:.1f}" for k, v in per_stock(rail_days).items()), "\n")

    print("Top 20 over the full window by average open->close on gap-up days (most faded):")
    top = stats.sort_values("avg_oc%").head(20).copy()
    top["red% H1"], top["red% H2"] = h1["red%"].reindex(top.index), h2["red%"].reindex(top.index)
    top["avg_oc% H1"], top["avg_oc% H2"] = h1["avg_oc%"].reindex(top.index), h2["avg_oc%"].reindex(top.index)
    print(top.round(1).to_string(), "\n")

    # Persistence: pick on H1, grade on H2
    both = h1[h1.days >= 10].join(h2[h2.days >= 10], lsuffix="_1", rsuffix="_2", how="inner")
    picks = both.sort_values("avg_oc%_1").head(20)
    rest = both.drop(picks.index)
    corr = both["avg_oc%_1"].corr(both["avg_oc%_2"], method="spearman")
    print(f"Persistence ({len(both)} stocks with 10+ gap-up days in each half):")
    print(f"  rank correlation of fade, first half vs second half: {corr:.2f}")
    for label, fr in (("20 most-faded in H1", picks), ("all other stocks", rest)):
        print(f"  {label:<22} H1 avg o->c {fr['avg_oc%_1'].mean():+.2f}%  red {fr['red%_1'].mean():.0f}%   ->   "
              f"H2 avg o->c {fr['avg_oc%_2'].mean():+.2f}%  red {fr['red%_2'].mean():.0f}%  "
              f"(net of {100 * COST:.1f}% cost as a short: {-fr['avg_oc%_2'].mean() - 100 * COST:+.2f}%)")
    stable = both[(both["red%_1"] >= 65) & (both["red%_2"] >= 65)
                  & (both["avg_oc%_1"] <= -1) & (both["avg_oc%_2"] <= -1)]
    print(f"\nStocks faded in BOTH halves (red >= 65% and avg o->c <= -1% each half): {len(stable)}")
    if len(stable):
        cols = ["days_1", "red%_1", "avg_oc%_1", "days_2", "red%_2", "avg_oc%_2", "fade3%_2", "max_up%_2"]
        print(stable.sort_values("avg_oc%_2")[cols].round(1).to_string())
    print(f"\n(For scale: with {len(both)} stocks, noise alone leaves some names passing both halves; "
          f"the persistence numbers above matter more than the list length.)")


if __name__ == "__main__":
    main()
