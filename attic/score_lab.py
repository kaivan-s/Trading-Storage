#!/usr/bin/env python3
"""
Build and validate two composite scores, one per holding horizon.

research_features.py measured single features. This combines the ones that
survived (|t| > 2 with the same sign in both halves of the sample) and tests
the composites the only way that means anything: weights are set looking at
the FIRST half only, then scored on the second half, which is never used to
choose anything.

SWING  — 10-20 session hold. Cross-sectional momentum and quality, the
         features whose IC grows with horizon.
QUICK  — 1-5 session hold. Near-high position, delivery, sector strength,
         and a PENALTY on ATR and volume expansion, both of which measured
         negative at short horizons. This is close to the inverse of how
         tom.for_tomorrow_momentum scores today.

Both are compared against the current production score as the baseline, and
against the base pool, net of a round-trip cost assumption.

Usage:
    python score_lab.py
    python score_lab.py --cost 0.30 --top 40
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import eval_tom
import research_features as rf
import tom as tomscan

# Weights are signs and rough magnitudes taken from the first-half ICs in
# research_features.py, deliberately coarse. Fine-tuning them on this data
# is how you curve fit; the second half is the check that they generalise.
SWING_W = {
    "mom_sharpe": 0.20,      # risk-adjusted 60d momentum, stable at every horizon
    "from_52w_high": 0.18,   # proximity to the 52-week high
    "sector_mom": 0.16,      # the stock's sector, not the stock
    "persist60": 0.16,       # how consistently it has held above the 20-EMA
    "deliv20": 0.15,         # delivery share: the most stable single feature
    "rs_mkt": 0.15,          # 60d return ranked against the market
}

QUICK_W = {
    "from_52w_high": 0.22,
    "deliv20": 0.20,
    "pos_hi": 0.16,
    "deliv_quality_rel": 0.14,
    "sector_mom": 0.12,
    "persist60": 0.08,
    "atr_pct": -0.20,        # NEGATIVE: high ATR underperformed, t = -6.0
    "vol_ratio": -0.10,      # NEGATIVE: t = -2.8 at 1 session
}

# What the production score uses today, for a like-for-like comparison.
CURRENT_W = {
    "vol_ratio": tomscan.MOM_W["vol"],
    "rsi": tomscan.MOM_W["rsi"],
    "atr_pct": tomscan.MOM_W["atr"],
    "ext_ema20": tomscan.MOM_W["ext"],
    "cmf": tomscan.MOM_W["cmf"],
}


def composite(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted sum of per-day cross-sectional percentile ranks.

    Ranking within each day removes level shifts between market regimes and
    makes the components comparable without assuming any distribution.
    """
    total = pd.Series(0.0, index=df.index)
    wsum = 0.0
    for feat, w in weights.items():
        if feat not in df.columns:
            continue
        r = df.groupby("date")[feat].rank(pct=True)
        total = total.add(r.fillna(0.5) * w, fill_value=0.0)
        wsum += abs(w)
    return total / (wsum or 1.0)


def basket(df: pd.DataFrame, score: str, horizon: int, top: int,
           cost: float) -> dict:
    """Equal-weight top-N by score each day, held `horizon` sessions."""
    col = f"fwd{horizon}"
    sub = df[["date", score, col]].dropna()
    if sub.empty:
        return {}
    rank = sub.groupby("date")[score].rank(ascending=False, method="first")
    picks = sub[rank <= top]
    daily = picks.groupby("date")[col].mean()
    pool = sub.groupby("date")[col].mean()

    # Cost is charged once per round trip, so annualising by horizon is what
    # makes a 1-session and a 20-session strategy comparable.
    gross = daily.mean()
    net = gross - cost
    per_session = net / horizon
    return {
        "gross": gross,
        "net": net,
        "per_session": per_session,
        "pool": pool.mean(),
        "lift": gross - pool.mean(),
        "win_days": 100.0 * (daily > pool).mean(),
        "days": len(daily),
        "sharpe": (daily.mean() / daily.std() * np.sqrt(252 / horizon)
                   if daily.std() else np.nan),
    }


def show(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n{title}")
    print(f"{'score':<12}{'gross %':>9}{'pool %':>9}{'lift %':>9}"
          f"{'net %':>8}{'/session':>10}{'beat pool':>11}{'ann.Sharpe':>12}")
    print("-" * 80)
    for name, r in rows:
        if not r:
            continue
        print(f"{name:<12}{r['gross']:>9.2f}{r['pool']:>9.2f}{r['lift']:>9.2f}"
              f"{r['net']:>8.2f}{r['per_session']:>10.3f}"
              f"{r['win_days']:>10.1f}%{r['sharpe']:>12.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Composite score validation")
    ap.add_argument("--sessions", type=int, default=322)
    ap.add_argument("--min-turnover", type=float, default=100.0,
                    help="min rolling median turnover, lacs")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--cost", type=float, default=0.30,
                    help="round-trip cost assumption, %%")
    args = ap.parse_args()

    stocks = eval_tom.load(args.sessions)
    print("Adding candidate features...")
    stocks = rf.add_features(stocks)
    print("Computing forward returns...")
    df = rf.build_dataset(stocks, args.min_turnover)

    print("Scoring...")
    df["swing"] = composite(df, SWING_W)
    df["quick"] = composite(df, QUICK_W)
    df["current"] = composite(df, CURRENT_W)

    dates = np.array(sorted(df["date"].unique()))
    mid = dates[len(dates) // 2]
    first = df[df["date"] <= mid]
    second = df[df["date"] > mid]
    print(f"\nweights chosen on : {pd.Timestamp(dates[0]).date()} -> "
          f"{pd.Timestamp(mid).date()}  ({first['date'].nunique()} sessions)")
    print(f"HELD OUT test half: {pd.Timestamp(mid).date()} -> "
          f"{pd.Timestamp(dates[-1]).date()}  ({second['date'].nunique()} sessions)")
    print(f"top {args.top} per day, {args.cost:.2f}% round-trip cost, "
          f"turnover >= {args.min_turnover:.0f} lacs")

    for horizon in (1, 5, 10, 20):
        for label, part in (("IN-SAMPLE half", first),
                            ("HELD-OUT half", second)):
            rows = [(s, basket(part, s, horizon, args.top, args.cost))
                    for s in ("swing", "quick", "current")]
            show(f"=== {horizon} session hold — {label} ===", rows)

    # Decile monotonicity on the held-out half only.
    print("\n" + "=" * 80)
    print("HELD-OUT half: mean forward return by score decile")
    print("=" * 80)
    for s in ("swing", "quick", "current"):
        sub = second[["date", s, "fwd10", "fwd20"]].dropna()
        if sub.empty:
            continue
        dec = sub.groupby("date")[s].transform(
            lambda x: pd.qcut(x.rank(method="first"), 10, labels=False,
                              duplicates="drop")
        )
        by = sub.groupby(dec)[["fwd10", "fwd20"]].mean()
        print(f"\n{s}:")
        print("  decile " + "".join(f"{int(d) + 1:>7}" for d in by.index))
        for h in ("fwd10", "fwd20"):
            print(f"  {h:<7}" + "".join(f"{v:>7.2f}" for v in by[h]))
        if 9 in by.index and 0 in by.index:
            print(f"  D10-D1: fwd10 {by.loc[9, 'fwd10'] - by.loc[0, 'fwd10']:+.2f}%"
                  f"   fwd20 {by.loc[9, 'fwd20'] - by.loc[0, 'fwd20']:+.2f}%")


if __name__ == "__main__":
    main()
