"""
Short-term reversal: the winners Expected Movers deliberately excludes.

The catch study showed 75% of names that rose the next session were NOT in
an uptrend near their 20-day high — they were quiet, RSI ~51, sitting ~8%
below the high. Expected Movers drops those by design. This module scores
that population instead.

The hypothesis is NOT "buy anything that fell". Two features in the IC work
pull in opposite directions and both were stable:

    rev5            recent 5-day loser        IC +0.019 @ 5d, t=3.8
    from_52w_high   close to the 52-week high IC +0.042 @ 5d, t=4.0

They only coexist in one shape: a structurally strong name having a short
bad patch. That is a dip inside strength, not a falling knife — so the
scorer rewards recent weakness only while long-term position stays good,
and penalises volatility (atr_pct IC -0.044 @ 5d, t=-6.0) so the dip is
orderly rather than a collapse.

The GATES BELOW ARE THE WRONG ONES -- see eval_meanrev.py.
`eval_reversal.py` showed this construction is flat at every horizon
(1-session +0.08%, t=0.61) and that `score` ranks below its own pool. But the
reason turned out to be the design, not the premise: REV_MIN_DROP_5D of 2%
over 5 days is a mild pullback, and it lands in the bucket that measures
zero. Re-tested on 20-day formation with a severity sweep, the extreme tail
DOES pay -- worst 2% of 20-day performers, held 7 sessions, +0.58% median
excess (p=0.003, +0.28% net of costs), monotonic in severity and liquid.

So if this is revived, the changes are: formation on mom20 rather than mom5,
severity as a cross-sectional rank in the worst 2-5% rather than an absolute
2% threshold, and a fixed 7-10 session hold rather than a reversion exit
(those measured as artifacts). Nothing here is wired into the product yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- gates ----------------------------------------------------------------
REV_MIN_PRICE = 20.0
REV_MIN_TURNOVER = 100.0   # Rs lakh, 60-day median. Fillability, not edge.
REV_MAX_FROM_52W = -0.35   # skip broken names: no worse than 35% off the high
REV_MIN_FROM_52W = -0.02   # and not already AT the high (that is momentum)
REV_RSI_MAX = 55.0         # must actually be soft
REV_MIN_DROP_5D = -0.02    # at least a 2% five-day fall

# --- score weights (sum 1.0), signed by measured IC at 5 sessions ---------
REV_W = {
    "drop": 0.30,      # how oversold on 5 days
    "position": 0.24,  # still near the 52-week high
    "calm": 0.18,      # low ATR — orderly dip, not a collapse
    "deliv": 0.16,     # real buyers still taking delivery
    "liquid": 0.12,    # turnover
}


def _clip01(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((pd.to_numeric(s, errors="coerce") - lo) / (hi - lo)).clip(0, 1)


def add_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features the reversal scan needs, on top of `stocks.add_indicators`.

    Everything is backward-looking on `adj`, so computing on full history and
    slicing by date introduces no lookahead.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)

    df["mom5"] = g["adj"].transform(lambda s: s.pct_change(5))
    df["mom20"] = g["adj"].transform(lambda s: s.pct_change(20))

    hi_col = "adj_high" if "adj_high" in df.columns else "adj"
    hi252 = g[hi_col].transform(lambda s: s.rolling(252, min_periods=120).max())
    df["from_52w_high"] = df["adj"] / hi252 - 1.0

    if "deliv_pct" in df.columns:
        df["deliv20"] = g["deliv_pct"].transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )

    df["med_turn60"] = g["turnover"].transform(
        lambda s: s.rolling(60, min_periods=20).median()
    )

    # Consecutive down sessions, for the "how deep is the patch" read.
    down = (g["adj"].transform(lambda s: s.pct_change()) < 0).astype(float)
    df["down_streak"] = down.groupby(df["symbol"]).transform(
        lambda s: s.groupby((s == 0).cumsum()).cumsum()
    )
    return df


def gate_mask(d: pd.DataFrame) -> pd.Series:
    """Hard gates. A name must be soft, but structurally intact and tradable."""
    adj = pd.to_numeric(d["adj"], errors="coerce")
    f52 = pd.to_numeric(d.get("from_52w_high"), errors="coerce")
    rsi = d["rsi"].fillna(50) if "rsi" in d.columns else pd.Series(50.0, index=d.index)
    m5 = pd.to_numeric(d.get("mom5"), errors="coerce")
    turn = pd.to_numeric(d.get("med_turn60"), errors="coerce").fillna(0)

    return (
        (adj >= REV_MIN_PRICE)
        & (turn >= REV_MIN_TURNOVER)
        & f52.between(REV_MAX_FROM_52W, REV_MIN_FROM_52W)
        & (rsi <= REV_RSI_MAX)
        & (m5 <= REV_MIN_DROP_5D)
    )


def score(d: pd.DataFrame) -> pd.Series:
    """0-1. Higher = more oversold inside a still-intact structure."""
    # Deeper 5-day fall scores higher, saturating at -12%.
    s_drop = _clip01(-pd.to_numeric(d["mom5"], errors="coerce"), 0.02, 0.12)
    # -2% from the 52w high scores 1.0, -30% scores 0.
    s_pos = _clip01(pd.to_numeric(d["from_52w_high"], errors="coerce"), -0.30, -0.02)
    s_calm = (
        1.0 - _clip01(d["atr_pct"], 0.02, 0.07)
        if "atr_pct" in d.columns
        else pd.Series(0.5, index=d.index)
    )
    s_deliv = (
        _clip01(d["deliv20"], 30.0, 70.0)
        if "deliv20" in d.columns
        else pd.Series(0.5, index=d.index)
    )
    s_liq = _clip01(np.log10(pd.to_numeric(d["med_turn60"], errors="coerce").clip(lower=1)), 2.0, 4.0)

    return (
        REV_W["drop"] * s_drop.fillna(0)
        + REV_W["position"] * s_pos.fillna(0)
        + REV_W["calm"] * s_calm.fillna(0)
        + REV_W["deliv"] * s_deliv.fillna(0)
        + REV_W["liquid"] * s_liq.fillna(0)
    )


def scan(stocks: pd.DataFrame, as_of=None, top_n: int | None = None) -> pd.DataFrame:
    """
    Reversal candidates for one session.

    `stocks` = add_indicators output. Pass `as_of` to score a historical
    session; defaults to the latest date present.
    """
    cols = [
        "symbol", "sector", "adj", "rsi", "mom5", "mom20", "from_52w_high",
        "atr_pct", "deliv20", "med_turn60", "down_streak", "rev_score", "why",
    ]
    if stocks is None or stocks.empty:
        return pd.DataFrame(columns=cols)

    d = stocks if "from_52w_high" in stocks.columns else add_reversal_features(stocks)
    as_of = d["date"].max() if as_of is None else pd.Timestamp(as_of)
    day = d[d["date"] == as_of].copy()
    if day.empty:
        return pd.DataFrame(columns=cols)

    day = day[gate_mask(day).fillna(False)].copy()
    if day.empty:
        return pd.DataFrame(columns=cols)

    day["rev_score"] = score(day)

    def why(r):
        return (
            f"Down {float(r['mom5']) * 100:.1f}% over 5 sessions but only "
            f"{abs(float(r['from_52w_high'])) * 100:.0f}% off its 52-week high, "
            f"RSI {float(r.get('rsi') or 0):.0f}. Oversold inside an intact "
            "structure — a dip, not a breakdown."
        )
    day["why"] = day.apply(why, axis=1)

    out = day.sort_values("rev_score", ascending=False)
    keep = [c for c in cols if c in out.columns]
    if top_n is not None:
        out = out.head(top_n)
    return out[keep].reset_index(drop=True)
