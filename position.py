"""
12-1 momentum: the ranking behind Leaders at rest.

WHAT IS LIVE HERE

`add_position_features` and `leaders_at_rest` are in the daily path. They add
a 12-1 momentum reading to every stock and use it to order the coil pool,
which is what the Setups page shows under "Leaders at rest" -- the 20 coiled
names with the strongest last year. See eval_listsize.py for why 20.

`scan()` is DORMANT. It produced the standalone Position Trades list (top
decile 12-1 above the 50-EMA, no coil requirement), which was removed from
the product: ranking the coil pool by the same factor measured better
(+1.30% at a 10-session hold against +1.15%, and steadier across sample
halves) on a quarter of the names. It is kept because it rests on a far
stronger prior than our own construction does -- see the note at the bottom
-- so it is the fallback if Leaders at rest disappoints live. Reviving it
needs a table; the DDL was removed from db.py along with it.

Everything below documents the factor itself, which both paths depend on.

WHY THIS SPECIFICATION

Two conditions, both earned rather than fitted. `eval_strategies.py` ran this
against 21 alternatives on one ruler; `backend` history is thin, so the
reasoning matters more than the local t-stat.

  12-1 MOMENTUM   Trailing 12-month return EXCLUDING the most recent month.
                  Skipping the last month is not a detail -- recent returns
                  reverse, so including them contaminates the signal. Inside
                  the trend population the rank orders outcomes cleanly at
                  both horizons:

                      top 5%        +0.73% @ 7d   +1.26% @ 10d
                      90-95%        +0.63%        +0.82%
                      70-90%        +0.41%        +0.74%
                      50-70%        +0.21%        +0.06%
                      bottom half   -0.85%       -1.07%

                  That monotonicity is why ranking is meaningful here and a
                  top-N cut is defensible. Neither of our other scans has it:
                  the coil score showed no relationship with forward returns
                  and the Expected Movers score was inversely related, which
                  is why both are uncapped.

  ABOVE 50-EMA    Tested head to head -- the names it keeps against the names
                  it discards, same days. +0.87% vs -0.03% at 7 sessions, a
                  +0.80% separation (t=2.02, p=0.044) winning 70% of days. At
                  5 sessions +0.71% (p=0.012).

WHAT IS DELIBERATELY ABSENT

  No golden cross. Requiring the 50-EMA above the 200 filters nothing here --
  all 106 top-decile names already satisfy it -- so it is dead weight.
  No pullback filter. It looked like +0.17pp in the comparison table and
  measured p=0.745 head to head, with no monotonicity across thresholds
  (+0.11/+0.34/+0.32/+0.22/+0.05 from 2% to 7%) and 53-61% days won. It was
  noise, and it is the reason every gate above had to pass a direct test.
  No six-month variant. Measured negative (-0.33%).

DO NOT PRINT A BASE RATE FROM OUR OWN DATA

12-1 momentum needs 250 sessions of warmup and the cache holds 324, leaving a
64-session measurable window -- roughly 9 independent 7-session periods inside
a single quarter. Our data is not what justifies this scan; it only confirms
the factor is not broken on NSE. The justification is the published record
(Jegadeesh & Titman 1993 and the replications since, across decades and
dozens of markets), which is why this is shipped where the mean-reversion
finding was not. Advertising a measured win rate off 9 periods is precisely
the mistake Expected Movers made.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FORMATION = 250   # sessions in the trailing window
SKIP = 20         # most recent sessions excluded (the reversal month)

# Why 20 and not 10 or 5 -- see eval_listsize.py. Ranking the coil pool by
# 12-1 momentum and keeping 20 measured +0.78% at a 7-session hold (NW t=3.37)
# and +1.30% at 10 (t=4.21, 84% of days positive), roughly double the
# unfiltered pool's +0.47%/+0.83%, on a third of the names. Cutting further
# breaks it: top 10 runs +1.98% then +0.12% across sample halves and top 5
# flips sign. 20 is where the list is short enough to act on and still long
# enough for the daily median to mean something.
REST_TOP_N = 20


@dataclass(frozen=True)
class PositionParams:
    min_price: float = 20.0
    min_median_turnover_lacs: float = 100.0
    liq_lookback: int = 60
    top_pct: float = 0.10        # keep the top decile of 12-1 momentum
    min_sessions: int = FORMATION + SKIP


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The two inputs this scan needs, on top of `stocks.add_indicators`.

    Both are backward-looking on `adj`, so computing across full history and
    slicing by date introduces no lookahead.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)

    r_long = g["adj"].transform(lambda s: s.pct_change(FORMATION))
    r_recent = g["adj"].transform(lambda s: s.pct_change(SKIP))
    # Return from t-250 to t-20: the long window with the recent month divided
    # back out, rather than subtracted, so it compounds correctly.
    df["mom12_1"] = (1 + r_long) / (1 + r_recent).replace(0, np.nan) - 1
    df["mom20"] = r_recent

    if "med_turn60" not in df.columns:
        df["med_turn60"] = g["turnover"].transform(
            lambda s: s.rolling(60, min_periods=20).median()
        )
    df["own_sessions"] = g.cumcount() + 1
    return df


def eligible(day: pd.DataFrame, p: PositionParams) -> pd.Series:
    """Tradable and old enough to have a 12-1 reading. Not a view on the stock."""
    adj = pd.to_numeric(day["adj"], errors="coerce")
    turn = pd.to_numeric(day.get("med_turn60"), errors="coerce").fillna(0)
    own = pd.to_numeric(day.get("own_sessions"), errors="coerce").fillna(0)
    return (
        (adj >= p.min_price)
        & (turn >= p.min_median_turnover_lacs)
        & (own >= p.min_sessions)
        & pd.to_numeric(day["mom12_1"], errors="coerce").notna()
    ).fillna(False)


def scan(stocks: pd.DataFrame, as_of=None, top_n: int | None = 40,
         params: PositionParams | None = None) -> pd.DataFrame:
    """
    Position Trade candidates for one session, best first.

    `stocks` = `add_indicators` output. Pass `as_of` to score a historical
    session; defaults to the latest date present. `top_n=None` returns the
    whole qualifying decile.

    The momentum cut is cross-sectional among eligible names ON THAT DAY, so
    the list adapts to the market rather than to a fixed return threshold.
    """
    cols = ["symbol", "sector", "adj", "mom12_1", "mom20", "ext_ema50",
            "rsi", "atr_pct", "med_turn60", "mom_rank", "why"]
    if stocks is None or stocks.empty:
        return pd.DataFrame(columns=cols)

    p = params or PositionParams()
    d = stocks if "mom12_1" in stocks.columns else add_position_features(stocks)
    as_of = d["date"].max() if as_of is None else pd.Timestamp(as_of)
    day = d[d["date"] == as_of].copy()
    if day.empty:
        return pd.DataFrame(columns=cols)

    day = day[eligible(day, p)]
    if day.empty:
        return pd.DataFrame(columns=cols)

    # Rank within the eligible set, then keep the top decile.
    day["mom_rank"] = day["mom12_1"].rank(pct=True)
    day = day[day["mom_rank"] >= 1.0 - p.top_pct]

    # The trend condition: holding above the 50-EMA. Measured head to head.
    if "ema50" in day.columns:
        e50 = pd.to_numeric(day["ema50"], errors="coerce")
        day["ext_ema50"] = pd.to_numeric(day["adj"], errors="coerce") / e50 - 1
        day = day[day["ext_ema50"] > 0]
    else:
        day["ext_ema50"] = np.nan
    if day.empty:
        return pd.DataFrame(columns=cols)

    def why(r):
        pctile = float(r["mom_rank"]) * 100
        return (
            f"Up {float(r['mom12_1']) * 100:.0f}% over the year to last month "
            f"({pctile:.0f}th percentile), and still "
            f"{float(r['ext_ema50']) * 100:.1f}% above its 50-day average. "
            "Hold 7 to 10 sessions — this is a trend to sit with, not a "
            "next-day move."
        )
    day["why"] = day.apply(why, axis=1)

    out = day.sort_values("mom12_1", ascending=False)
    if top_n is not None:
        out = out.head(top_n)
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


def leaders_at_rest(coil_rows: pd.DataFrame, stocks: pd.DataFrame,
                    as_of=None, top_n: int | None = REST_TOP_N) -> pd.DataFrame:
    """
    The coil pool ranked by 12-1 momentum, strongest year first.

    A name qualifies on the coil gates -- quiet, tight, near its highs -- and
    is then ORDERED by how strong its last year was. So this is a proven
    leader taking a rest, which is why it reads better than either parent
    list: the coil gates time the entry and momentum picks which bases are
    worth waiting on.

    This is the one list in the app where sort order carries information. The
    coil score does not rank outcomes at all and the Expected Movers score
    ranked them backwards, but 12-1 momentum orders them monotonically (top 5%
    +0.73% at 7 sessions down to -0.85% for the bottom half).

    `coil_rows` = stocks.scan() output. `stocks` needs `mom12_1`, so pass an
    add_position_features() frame. Names without a 12-1 reading -- anything
    under 270 sessions of history -- rank last rather than being dropped, so a
    thin cache degrades to the unranked pool instead of an empty list.
    """
    if coil_rows is None or coil_rows.empty:
        return coil_rows if coil_rows is not None else pd.DataFrame()

    out = coil_rows.copy()
    if stocks is not None and not stocks.empty and "mom12_1" in stocks.columns:
        as_of = stocks["date"].max() if as_of is None else pd.Timestamp(as_of)
        day = stocks[stocks["date"] == as_of]
        mom = day.set_index("symbol")["mom12_1"]
        out["mom12_1"] = out["symbol"].map(mom)
    elif "mom12_1" not in out.columns:
        out["mom12_1"] = np.nan

    # na_position="last" is the degradation path: no reading means unranked,
    # not excluded.
    out = out.sort_values("mom12_1", ascending=False, na_position="last")
    if top_n is not None:
        out = out.head(top_n)
    return out.reset_index(drop=True)
