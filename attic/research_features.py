#!/usr/bin/env python3
"""
Which features predict forward returns, and at what horizon?

eval_tom.py showed the current For-Tom score has no return edge at any
horizon. Before writing another scorer, this measures candidate features
against forward returns so the weights can come from evidence.

Two families are missing from the current feature set entirely:

  RELATIVE STRENGTH  — return ranked against the market and against the
      stock's own sector. Cross-sectional momentum is the effect with the
      most out-of-sample support in equities, and nothing in stocks.py
      measures it. Expected to pay at 10-20 sessions.

  SHORT-TERM REVERSAL — recent losers bouncing. eval_tom found extension
      and RSI flat-to-negative next day, which is the reversal signature.
      Expected to pay at 1-5 sessions, opposite to how For-Tom scores today.

Method: for each session, rank every liquid stock cross-sectionally on each
feature, then correlate that rank with forward return (Spearman). Averaging
the daily correlations gives the information coefficient (IC); the spread of
the daily values gives its t-stat, which is what says whether an IC is real
or one lucky month. Every feature is also reported on a first-half /
second-half split, because an edge that does not survive that is curve fit.

Usage:
    python research_features.py
    python research_features.py --min-turnover 100 --dump ic.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import eval_tom
import fetch  # noqa: F401  (eval_tom.load reads the cache through it)

HORIZONS = (1, 5, 10, 20)


# --------------------------------------------------------------------------
# Candidate features
# --------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate predictors, added to the add_indicators output.

    Everything here is backward-looking on `adj` (split-adjusted), so
    computing on full history and slicing by date introduces no lookahead.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)

    # --- absolute momentum over several windows -------------------------
    for n in (5, 20, 60, 120, 252):
        df[f"mom{n}"] = g["adj"].transform(lambda s, n=n: s.pct_change(n))

    # Classic 12-1 momentum: 12-month return skipping the last month, which
    # removes the short-term reversal that contaminates raw 12-month return.
    df["mom_12_1"] = (df["mom252"] + 1.0) / (df["mom20"] + 1.0) - 1.0

    # --- distance from the 52-week high ---------------------------------
    hi252 = g["adj_high"].transform(
        lambda s: s.rolling(252, min_periods=120).max()
    )
    df["from_52w_high"] = df["adj"] / hi252 - 1.0
    lo252 = g["adj_low"].transform(
        lambda s: s.rolling(252, min_periods=120).min()
    )
    df["from_52w_low"] = df["adj"] / lo252 - 1.0

    # --- trend quality / persistence ------------------------------------
    above20 = (df["adj"] > df["ema20"]).astype(float)
    df["persist60"] = (g.apply(lambda x: above20.loc[x.index]
                               .rolling(60, min_periods=20).mean(),
                               include_groups=False)
                        .reset_index(level=0, drop=True))
    df["ext_ema200"] = df["adj"] / df["ema200"] - 1.0
    # Realised vol: the denominator that turns raw momentum into risk-adjusted.
    ret1 = g["adj"].transform(lambda s: s.pct_change())
    df["vol60"] = g.apply(lambda x: ret1.loc[x.index]
                          .rolling(60, min_periods=30).std(),
                          include_groups=False
                          ).reset_index(level=0, drop=True)
    df["mom_sharpe"] = df["mom60"] / df["vol60"].replace(0, np.nan)

    # --- short-term reversal (sign flipped so higher = more oversold) ----
    df["rev5"] = -df["mom5"]
    df["below_ema20"] = -df["ext_ema20"]

    # --- volume / participation trend -----------------------------------
    v20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    v60 = g["volume"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    df["vol_trend"] = v20 / v60.replace(0, np.nan)
    if "deliv_pct" in df.columns:
        d20 = g["deliv_pct"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        df["deliv20"] = d20

    # --- relative strength ----------------------------------------------
    # vs the market: cross-sectional percentile of 60-day return that day.
    df["rs_mkt"] = df.groupby("date")["mom60"].rank(pct=True)
    # vs own sector: 60-day return minus the sector's median that day.
    if "sector" in df.columns:
        sec_med = df.groupby(["date", "sector"])["mom60"].transform("median")
        df["rs_sector"] = df["mom60"] - sec_med
        # Sector momentum itself, as a standalone feature.
        df["sector_mom"] = sec_med
    return df


FEATURES = [
    # relative strength family
    "rs_mkt", "rs_sector", "sector_mom",
    # absolute momentum family
    "mom20", "mom60", "mom120", "mom252", "mom_12_1", "mom_sharpe",
    # position / trend quality
    "from_52w_high", "from_52w_low", "persist60", "ext_ema200", "pos_hi",
    # short-term reversal family
    "rev5", "below_ema20",
    # current For-Tom inputs, for comparison
    "vol_ratio", "rsi", "atr_pct", "ext_ema20", "cmf",
    # participation
    "vol_trend", "deliv20", "deliv_quality_rel", "turnover",
]


# --------------------------------------------------------------------------
# IC machinery
# --------------------------------------------------------------------------
def build_dataset(stocks: pd.DataFrame, min_turnover: float) -> pd.DataFrame:
    """Forward returns + a tradability filter, one row per symbol-day."""
    df = stocks.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)
    for h in HORIZONS:
        df[f"fwd{h}"] = g["adj"].transform(
            lambda s, h=h: s.shift(-h) / s - 1.0
        ) * 100.0

    # Tradability: rolling median turnover, computed from the past only.
    if "turnover" in df.columns:
        df["turn_med"] = g["turnover"].transform(
            lambda s: s.rolling(60, min_periods=20).median()
        )
        df = df[df["turn_med"].fillna(0) >= min_turnover]
    df = df[df["adj"] >= 20.0]
    return df


def daily_ic(df: pd.DataFrame, feature: str, horizon: int) -> pd.Series:
    """Per-day Spearman between the feature and forward return."""
    col = f"fwd{horizon}"
    sub = df[["date", feature, col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)

    def one(part: pd.DataFrame) -> float:
        if len(part) < 50:
            return np.nan
        return part[feature].rank().corr(part[col].rank())

    return (sub.groupby("date", sort=True)
               .apply(one, include_groups=False)
               .dropna())


def report(df: pd.DataFrame, features: list[str], dump: str | None) -> None:
    dates = np.array(sorted(df["date"].unique()))
    mid = dates[len(dates) // 2]
    rows = []

    print(f"\nRows {len(df):,}   symbols {df['symbol'].nunique():,}   "
          f"sessions {len(dates):,} "
          f"({pd.Timestamp(dates[0]).date()} -> {pd.Timestamp(dates[-1]).date()})")
    print("\nIC = mean daily rank correlation with forward return.")
    print("t = IC / standard error of the daily ICs. |t| > 2 is the usual bar.")
    print("h1 / h2 = IC in the first and second half of the period.\n")

    for h in HORIZONS:
        print("=" * 78)
        print(f"HORIZON: {h} session{'s' if h > 1 else ''}")
        print("=" * 78)
        print(f"{'feature':<20}{'IC':>8}{'t':>8}{'h1':>8}{'h2':>8}"
              f"{'D10-D1 %':>11}{'days':>7}")
        print("-" * 70)

        scored = []
        for f in features:
            if f not in df.columns:
                continue
            ics = daily_ic(df, f, h)
            if len(ics) < 30:
                continue
            ic = ics.mean()
            t = ic / (ics.std() / np.sqrt(len(ics))) if ics.std() else np.nan
            h1 = ics[ics.index <= mid].mean()
            h2 = ics[ics.index > mid].mean()

            # Decile spread, in return points, on daily cross-sectional ranks.
            sub = df[["date", f, f"fwd{h}"]].dropna()
            dec = sub.groupby("date")[f].transform(
                lambda s: pd.qcut(s.rank(method="first"), 10,
                                  labels=False, duplicates="drop")
            )
            spread = np.nan
            if dec.notna().any():
                by = sub.groupby(dec)[f"fwd{h}"].mean()
                if 9 in by.index and 0 in by.index:
                    spread = by.loc[9] - by.loc[0]

            scored.append((f, ic, t, h1, h2, spread, len(ics)))
            rows.append({"horizon": h, "feature": f, "ic": ic, "t": t,
                         "ic_h1": h1, "ic_h2": h2, "spread": spread,
                         "days": len(ics)})

        # Strongest absolute IC first: sign tells direction, magnitude edge.
        for f, ic, t, h1, h2, spread, n in sorted(
                scored, key=lambda x: -abs(x[1])):
            flag = ""
            if abs(t) > 2 and np.sign(h1) == np.sign(h2) and abs(ic) > 0.01:
                flag = "  <- stable"
            print(f"{f:<20}{ic:>8.4f}{t:>8.1f}{h1:>8.4f}{h2:>8.4f}"
                  f"{spread:>11.2f}{n:>7}{flag}")
        print()

    if dump and rows:
        pd.DataFrame(rows).to_csv(dump, index=False)
        print(f"IC table -> {dump}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature IC research")
    ap.add_argument("--sessions", type=int, default=322,
                    help="cached sessions to load")
    ap.add_argument("--min-turnover", type=float, default=30.0,
                    help="min rolling median turnover, lacs")
    ap.add_argument("--dump", type=str, help="save the IC table to CSV")
    args = ap.parse_args()

    stocks = eval_tom.load(args.sessions)
    print("Adding candidate features...")
    stocks = add_features(stocks)
    print("Computing forward returns...")
    df = build_dataset(stocks, args.min_turnover)
    report(df, FEATURES, args.dump)


if __name__ == "__main__":
    main()
