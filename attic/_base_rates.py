#!/usr/bin/env python3
"""
Re-measure the setups and coils base rates that evidence.js advertises.

Claims under test, from ui/src/evidence.js:
    setups  n=151  win 66.2%  median excess +3.67%
    coils   n=377  win 59.2%  median excess +1.70%
    window  "Jun 2025 - Sep 2026"

Both scans require 200 sessions of a symbol's own history (CoilParams.
min_sessions, and ema200 built with min_periods=200), so nothing can qualify
until that warmup clears. With 324 cached sessions the earliest qualifying
date is ~200 sessions in, which is why the movers window turned out to be
Mar-Sep 2026 rather than the advertised range. This checks whether the
setups and coils numbers carry the same problem.

Method mirrors the stated one: each symbol counted once, on the first day it
qualified, held 20 sessions, measured as excess over the same session's
all-stock median return.

The seven coil gates are elementwise, so they vectorise across the whole
panel; only the two per-symbol eligibility rules (own history, trailing
liquidity) need rolling windows. Sector classification is not vectorisable
and runs per day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fetch
import panel as pnl
import scan as sc
import stocks as st

HOLD = 20
SETUP_KLASSES = ("CROSSING", "PULLBACK")


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(fetch.RAW_DIR.glob("bhav_*"))
    print(f"Loading {len(paths)} cached sessions "
          f"({paths[0].stem[5:]} -> {paths[-1].stem[5:]})...")
    raw = pd.concat([fetch._read_cache(p) for p in paths], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["symbol", "date"])
    smap = (pd.read_json(fetch.CACHE_DIR / "sector_map.json", orient="index")
              .rename_axis("symbol").reset_index())
    print("Building panel...")
    stocks, panel = pnl.build(raw, smap)
    print("Computing indicators...")
    return st.add_indicators(stocks), panel


def coil_mask(s: pd.DataFrame, p: st.CoilParams) -> pd.Series:
    """The seven elementwise gates plus the two per-symbol eligibility rules."""
    g = s.groupby("symbol", sort=False)
    own = g.cumcount() + 1
    liq = g["turnover"].transform(
        lambda x: x.rolling(p.liq_lookback, min_periods=1).median())
    return (
        (s["adj"] >= p.min_price)
        & (s["adj"] > s["ema50"])
        & (s["ema50"] > s["ema200"])
        & s["pos_hi"].between(p.near_high_min, p.near_high_max)
        & s["rsi"].between(p.rsi_low, p.rsi_high)
        & (s["ext_ema20"].abs() <= p.max_ext_ema20)
        & (s["vol_ratio"] <= p.vol_dryup_max)
        & (s["range20"] <= p.range_max)
        & (own >= p.min_sessions)
        & (liq >= p.min_median_turnover_lacs)
    ).fillna(False)


def setup_sectors_by_day(panel: pd.DataFrame, days) -> dict:
    """{date -> set of sectors classified CROSSING or PULLBACK that day}."""
    out = {}
    for i, day in enumerate(days, 1):
        rows = sc.classify(panel, as_of=day)
        out[day] = (set() if rows.empty else
                    set(rows[rows["klass"].isin(SETUP_KLASSES)]["sector"]))
        if i % 25 == 0 or i == len(days):
            print(f"  classified {i}/{len(days)} days")
    return out


def report(label: str, first: pd.DataFrame, claim: dict) -> None:
    n = len(first)
    if n == 0:
        print(f"{label}: nothing qualified")
        return
    wr = 100 * (first["excess"] > 0).mean()
    mx = first["excess"].median()
    print(f"\n{label}")
    print(f"  window measured     {first['date'].min():%d %b %Y} -> "
          f"{first['date'].max():%d %b %Y}  ({first['date'].nunique()} distinct days)")
    print(f"  {'':<20}{'measured':>12}{'claimed':>12}{'delta':>10}")
    print(f"  {'names (n)':<20}{n:>12,}{claim['n']:>12,}{n - claim['n']:>+10,}")
    print(f"  {'win rate':<20}{wr:>11.1f}%{100 * claim['wr']:>11.1f}%"
          f"{wr - 100 * claim['wr']:>+9.1f}")
    print(f"  {'median excess':<20}{mx:>11.2f}%{100 * claim['mx']:>11.2f}%"
          f"{mx - 100 * claim['mx']:>+9.2f}")
    print(f"  mean excess         {first['excess'].mean():>11.2f}%"
          f"   (mean is skew-inflated; median is the honest one)")


def main() -> None:
    s, panel = load_all()
    p = st.CoilParams()
    s = s.sort_values(["symbol", "date"])
    g = s.groupby("symbol", sort=False)

    # Forward 20-session excess return over the all-stock median.
    s["fwd"] = (g["adj"].shift(-HOLD) / s["adj"] - 1.0) * 100
    s["excess"] = s["fwd"] - s.groupby("date")["fwd"].transform("median")

    s["coil"] = coil_mask(s, p)
    L = "=" * 78
    print(f"\n{L}\nBASE RATES — setups and coils, {HOLD}-session hold\n{L}")

    cd = s[s["coil"]]["date"]
    print(f"cache spans           {s['date'].min():%d %b %Y} -> "
          f"{s['date'].max():%d %b %Y} ({s['date'].nunique()} sessions)")
    print(f"first coil qualifies  {cd.min():%d %b %Y}  "
          f"— {s['date'].nunique() - s[s['date'] >= cd.min()]['date'].nunique()} "
          f"sessions consumed by the 200-session warmup")
    print(f"coil name-days        {int(s['coil'].sum()):,} over "
          f"{cd.nunique()} days ({s['coil'].sum() / cd.nunique():.0f}/day)")

    # Setups need the sector verdict, so classify only days that have coils.
    days = sorted(cd.unique())
    print(f"\nClassifying sectors for {len(days)} days...")
    ss = setup_sectors_by_day(panel, days)
    s["setup"] = s["coil"] & [
        r.sector in ss.get(r.date, ()) for r in s[["date", "sector"]].itertuples()
    ]

    # Each symbol counted once, on the first day it qualified, and only where
    # a full 20-session forward window exists.
    for label, col, claim in (
        ("SETUPS (coils in actionable sectors)", "setup",
         {"n": 151, "wr": 0.662, "mx": 0.0367}),
        ("COILED BASES (all coils)", "coil",
         {"n": 377, "wr": 0.592, "mx": 0.017}),
    ):
        q = s[s[col] & s["excess"].notna()]
        first = q.sort_values("date").groupby("symbol", as_index=False).first()
        report(label, first, claim)

    print(f"\n{L}\nSanity: does the claimed window exist in the data?\n{L}")
    print("Both scans need 200 sessions of a symbol's own history, so the")
    print("earliest possible qualifying date is fixed by the cache start, not")
    print("by market conditions. Any window label starting before that date")
    print("describes the data range rather than the measurement range.")


if __name__ == "__main__":
    main()
