"""
Stock-level pre-breakout scan: find coiled bases before they resolve.

Where a breakout screener asks "is it AT its high, on RISING volume, closing
strong" — all of which are confirmations — this asks the inverse:

    near the high but not at it, on FALLING volume, in a NARROWING range,
    with money quietly coming in.

That is the phase before the breakout. Most of these never break out; they drift
sideways or roll over. The output is a ranked list to study, not a signal.

All technicals are computed on a split-adjusted series. Bhavcopy prices are raw,
and a 1:10 split would otherwise read as a 90% crash and wreck every EMA, the
RSI and the distance-from-high in one go.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CoilParams:
    """
    Starting values. Same warning as the sector thresholds: these were reasoned,
    not measured. Run the backtest before believing any of them.
    """
    high_lookback: int = 85         # ~4 months, matching the usual breakout scan
    near_high_min: float = 0.85     # at least 85% of the period high…
    # near_high_max is NOT an extension filter. A flat base sits at its highs by
    # definition; excluding those threw away the cleanest setups in the first
    # real run (Laurus at 0.997 with a 10% range and CMF 0.38). Extension is
    # measured by max_ext_ema20 against the 20-EMA, which is the right tool.
    near_high_max: float = 0.99     # was 0.95 — see comment below
    rsi_low: float = 45.0           # below this the trend is damaged
    rsi_high: float = 68.0          # above this the move already happened
    max_ext_ema20: float = 0.06     # was 0.07, tightened to compensate
    vol_dryup_max: float = 1.00     # 5-day volume must be under the 20-day
    range_max: float = 0.14         # 20-day high/low spread under 14%
    min_median_turnover_lacs: float = 30.0
    min_price: float = 20.0
    # Sessions of the symbol's OWN history required before its gates mean
    # anything. The 200-EMA is an ewm, so it is defined on a stock's first day;
    # without this a recent listing clears the trend gate on a 40-day average
    # wearing the name ema200.
    min_sessions: int = 200
    # Liquidity is judged on this trailing window, not the whole loaded panel.
    liq_lookback: int = 60


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI. ewm(alpha=1/n) is Wilder smoothing, not a plain EMA."""
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = (up.ewm(alpha=1 / n, adjust=False).mean()
          / dn.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan))
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame, p: CoilParams | None = None) -> pd.DataFrame:
    """
    Expects the output of panel.stock_metrics (needs `tri`, `cmf`,
    `deliv_quality`). Adds EMAs, RSI, ATR, range contraction and position.
    """
    p = p or CoilParams()
    if df.empty:
        return df
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)

    # Rescale the return index to today's actual price so EMAs, the 20-day
    # range and the breakout trigger all come out in real rupees while staying
    # split-adjusted.
    last_close = g["close"].transform("last")
    last_tri = g["tri"].transform("last")
    df["adj"] = df["tri"] * (last_close / last_tri)
    scale = df["adj"] / df["close"].replace(0, np.nan)
    df["adj_high"] = df["high"] * scale
    df["adj_low"] = df["low"] * scale

    g = df.groupby("symbol", sort=False)
    # min_periods matters: an ewm without it returns a value on the first row,
    # so ema200 for a 40-session listing is a 40-day average under a name that
    # claims 200. Every downstream trend gate reads `adj > ema50 > ema200`,
    # which is False against NaN — fail closed is the right answer here.
    for n in (10, 20, 50, 200):
        df[f"ema{n}"] = g["adj"].transform(
            lambda s, n=n: s.ewm(span=n, adjust=False, min_periods=n).mean()
        )
    df["rsi"] = g["adj"].transform(_rsi)

    df["atr"] = (df.groupby("symbol", sort=False)
                   .apply(lambda x: _atr(x["adj_high"], x["adj_low"], x["adj"]),
                          include_groups=False)
                   .reset_index(level=0, drop=True))
    df["atr_pct"] = df["atr"] / df["adj"]

    g = df.groupby("symbol", sort=False)

    # --- position --------------------------------------------------------
    df["hi_n"] = g["adj_high"].transform(
        lambda s, n=p.high_lookback: s.rolling(n, min_periods=int(n * 0.6)).max()
    )
    df["pos_hi"] = df["adj"] / df["hi_n"]              # 1.0 = at the high
    df["trigger"] = g["adj_high"].transform(
        lambda s: s.rolling(20, min_periods=10).max()
    )
    # The breakout line a *later* session can clear. `trigger` includes this
    # bar's high, so a close on the same session can never print through it.
    g = df.groupby("symbol", sort=False)
    df["prior_trigger"] = g["trigger"].shift(1)
    df["to_trigger"] = df["trigger"] / df["adj"] - 1.0  # % move to break out

    # --- volume dry-up ---------------------------------------------------
    v5 = g["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    v20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["vol_ratio"] = v5 / v20.replace(0, np.nan)

    # --- range contraction ----------------------------------------------
    # A base IS a narrowing range. This is the one pre-breakout marker that
    # cannot be manufactured by churn, which makes it a useful independent
    # check against the delivery-percentage confound.
    hi20 = g["adj_high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    lo20 = g["adj_low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    df["range20"] = hi20 / lo20.replace(0, np.nan) - 1.0
    df["contraction"] = df["range20"] / g["range20"].transform(lambda s: s.shift(20))

    # --- how long has it been based? ------------------------------------
    # 0.80 saturated: every candidate returned exactly 60.0, making this term a
    # constant. 0.90 actually discriminates between a long tight base and a
    # stock that only recently climbed into range.
    near = (df["pos_hi"] > 0.90).astype(float)
    df["base_days"] = g.apply(
        lambda x: near.loc[x.index].rolling(60, min_periods=20).sum(),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Delivery quality is confounded: delivery PERCENTAGE rises mechanically
    # when volume falls, so on a quiet market day almost every stock looks like
    # it is being accumulated. Normalise against the cross-sectional median that
    # day to strip out the common component.
    df["deliv_quality_rel"] = df["deliv_quality"] / df.groupby("date")[
        "deliv_quality"
    ].transform("median")

    df["ext_ema20"] = df["adj"] / df["ema20"] - 1.0
    return _add_coil_streak(df, p)


def _add_coil_streak(df: pd.DataFrame, p: CoilParams) -> pd.DataFrame:
    """
    `coil_days`: consecutive sessions this row has satisfied the seven gates.

    The scan only ever evaluates one cross-section, so a name coiled for six
    weeks is indistinguishable from one that qualified this morning — even
    though the score rewards duration through `base_days`. Counting the run
    lets the reader tell a fresh setup from a stale one.

    This applies the elementwise gates only. The eligibility restrictions in
    `_day_frame` (own history, recent liquidity) are per-symbol rather than
    per-row, so they cannot lengthen or shorten a streak.
    """
    ok = (
        (df["adj"] >= p.min_price)
        & (df["adj"] > df["ema50"])
        & (df["ema50"] > df["ema200"])
        & df["pos_hi"].between(p.near_high_min, p.near_high_max)
        & df["rsi"].between(p.rsi_low, p.rsi_high)
        & (df["ext_ema20"].abs() <= p.max_ext_ema20)
        & (df["vol_ratio"] <= p.vol_dryup_max)
        & (df["range20"] <= p.range_max)
    ).astype(int)

    # Every non-coil session opens a new block, so consecutive qualifying
    # sessions share a block id and a cumulative sum inside it is the run.
    blocks = (1 - ok).groupby(df["symbol"], sort=False).cumsum()
    df["coil_days"] = ok.groupby([df["symbol"], blocks], sort=False).cumsum()
    return df


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _norm(s: pd.Series, lo: float, hi: float, invert: bool = False) -> pd.Series:
    """Clip into [lo, hi] then scale to 0-1."""
    x = (s.clip(lo, hi) - lo) / (hi - lo)
    return (1.0 - x) if invert else x


FILTER_ORDER = ["price", "trend", "pos", "rsi", "ext", "vol", "range"]

FILTER_TEXT = {
    "price": "Price is at least ₹20",
    "trend": "Close above the 50-EMA, and the 50 above the 200 (uptrend)",
    "pos": "At 85–99% of the 85-day high — near it, not through it",
    "rsi": "RSI between 45 and 68 — trend intact, move not already spent",
    "ext": "Within 6% of the 20-EMA (not extended)",
    "vol": "5-day volume at or below the 20-day average (dry-up)",
    "range": "20-day high-to-low range at or under 14% (tight)",
}


def _cell(row, key):
    if row is None:
        return None
    if isinstance(row, dict):
        v = row.get(key)
    elif key in getattr(row, "index", []):
        v = row[key]
    else:
        v = None
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if pd.isna(v):
        return None
    return v


def evaluate_filters(row, p: CoilParams | None = None) -> list[dict]:
    """Pass/fail the coil gates for one stock row. Missing inputs fail closed."""
    p = p or CoilParams()
    adj = _cell(row, "adj")
    ema50 = _cell(row, "ema50")
    ema200 = _cell(row, "ema200")
    pos = _cell(row, "pos_hi")
    rsi = _cell(row, "rsi")
    ext = _cell(row, "ext_ema20")
    vol = _cell(row, "vol_ratio")
    rng = _cell(row, "range20")

    flags = {
        "price": adj is not None and adj >= p.min_price,
        "trend": (adj is not None and ema50 is not None and ema200 is not None
                  and adj > ema50 and ema50 > ema200),
        "pos": pos is not None and p.near_high_min <= pos <= p.near_high_max,
        "rsi": rsi is not None and p.rsi_low <= rsi <= p.rsi_high,
        "ext": ext is not None and abs(ext) <= p.max_ext_ema20,
        "vol": vol is not None and vol <= p.vol_dryup_max,
        "range": rng is not None and rng <= p.range_max,
    }
    return [{"id": k, "ok": flags[k], "text": FILTER_TEXT[k]} for k in FILTER_ORDER]


def coil_score(row) -> float | None:
    """Same equal-weight coil score the scan uses, even if filters failed."""
    parts = []
    mapping = [
        ("vol_ratio", 0.40, 1.00, True),
        ("range20", 0.04, CoilParams().range_max, True),
        ("contraction", 0.4, 1.2, True),
        ("cmf", -0.05, 0.20, False),
        ("deliv_quality_rel", 0.85, 1.40, False),
        ("base_days", 25, 55, False),
    ]
    defaults = {"contraction": 1.0, "cmf": 0.0, "deliv_quality_rel": 1.0, "base_days": 0.0}
    for key, lo, hi, invert in mapping:
        v = _cell(row, key)
        if v is None:
            v = defaults.get(key)
        if v is None:
            return None
        x = (min(hi, max(lo, float(v))) - lo) / (hi - lo)
        parts.append(1.0 - x if invert else x)
    return float(np.mean(parts) * 100.0)


FLAG_COLS = ["f_price", "f_trend", "f_pos", "f_rsi", "f_ext", "f_vol", "f_range"]


def _day_frame(stocks: pd.DataFrame, p: CoilParams, as_of) -> pd.DataFrame:
    """
    The `as_of` cross-section with the seven gates evaluated, restricted to
    names that could actually be traded on a breakout.

    Two restrictions the gates themselves cannot express:

      - Liquidity is the median turnover over the trailing `liq_lookback`
        sessions. A full-history median lets a name that traded ₹80L a day
        eight months ago and ₹5L now clear the floor while being unbuyable.
      - A symbol needs `min_sessions` of its own history, or `hi_n`, `base_days`
        and the EMAs are all measuring a window the stock has not lived through.

    Comparisons against NaN are False, so a missing input fails its gate.
    """
    hist = stocks[stocks["date"] <= as_of]
    if hist.empty:
        return hist.copy()

    sessions = hist.groupby("symbol")["date"].nunique()
    dates = np.sort(hist["date"].unique())
    window = dates[-p.liq_lookback:] if len(dates) > p.liq_lookback else dates
    liq = hist[hist["date"].isin(window)].groupby("symbol")["turnover"].median()

    keep = sessions.index[
        (sessions >= p.min_sessions)
        & (liq.reindex(sessions.index) >= p.min_median_turnover_lacs)
    ]

    d = hist[(hist["date"] == as_of) & hist["symbol"].isin(keep)].copy()
    if d.empty:
        return d

    d["f_price"] = d["adj"] >= p.min_price
    d["f_trend"] = (d["adj"] > d["ema50"]) & (d["ema50"] > d["ema200"])
    d["f_pos"] = d["pos_hi"].between(p.near_high_min, p.near_high_max)
    d["f_rsi"] = d["rsi"].between(p.rsi_low, p.rsi_high)
    d["f_ext"] = d["ext_ema20"].abs() <= p.max_ext_ema20
    d["f_vol"] = d["vol_ratio"] <= p.vol_dryup_max
    d["f_range"] = d["range20"] <= p.range_max
    d["n_fail"] = (~d[FLAG_COLS]).sum(axis=1)
    return d


def gate_flags(stocks: pd.DataFrame, p: CoilParams | None = None,
               as_of=None) -> pd.DataFrame:
    """
    The `as_of` cross-section with the seven gates evaluated, passing or not.

    `scan` keeps only `n_fail == 0` and ranks it. Episode tracking needs the
    rows that fail too, so a base leaving the list can name the condition it
    lost instead of just vanishing. Same `_day_frame`, so membership here is
    the same membership the published list is drawn from.
    """
    p = p or CoilParams()
    as_of = pd.Timestamp(as_of) if as_of is not None else stocks["date"].max()
    return _day_frame(stocks, p, as_of)


def scan(stocks: pd.DataFrame, p: CoilParams | None = None,
         as_of=None, top: int = 40) -> pd.DataFrame:
    """
    Hard filters first, then rank whatever survives by how tightly coiled it is.

    Filters are binary because they encode "this is the wrong phase". The score
    is continuous because among stocks in the right phase, tighter is better —
    but there is no threshold that makes one a signal.
    """
    p = p or CoilParams()
    as_of = pd.Timestamp(as_of) if as_of is not None else stocks["date"].max()

    d = _day_frame(stocks, p, as_of)
    if d.empty:
        return d
    passed = d[d["n_fail"] == 0].copy()

    if passed.empty:
        return passed

    # Equal weights, deliberately. Any other weighting would be a claim about
    # relative importance that nothing here supports.
    passed["s_vol"] = _norm(passed["vol_ratio"], 0.40, 1.00, invert=True)
    passed["s_range"] = _norm(passed["range20"], 0.04, p.range_max, invert=True)
    passed["s_contract"] = _norm(passed["contraction"].fillna(1.0), 0.4, 1.2, invert=True)
    passed["s_cmf"] = _norm(passed["cmf"].fillna(0.0), -0.05, 0.20)
    passed["s_deliv"] = _norm(passed["deliv_quality_rel"].fillna(1.0), 0.85, 1.40)
    passed["s_base"] = _norm(passed["base_days"].fillna(0), 25, 55)

    parts = ["s_vol", "s_range", "s_contract", "s_cmf", "s_deliv", "s_base"]
    passed["coil"] = passed[parts].mean(axis=1) * 100.0

    cols = ["symbol", "sector", "adj", "coil", "coil_days", "pos_hi", "to_trigger",
            "trigger", "rsi", "vol_ratio", "range20", "contraction", "cmf",
            "deliv_quality_rel", "deliv_pct", "base_days", "atr_pct", "ext_ema20"]
    cols = [c for c in cols if c in passed.columns]
    return (passed.sort_values("coil", ascending=False)[cols]
                  .head(top).reset_index(drop=True))


def near_miss(stocks: pd.DataFrame, p: CoilParams | None = None,
              as_of=None, top: int = 20) -> pd.DataFrame:
    """
    Stocks failing exactly one filter.

    Worth reading daily: it tells you which condition is doing the excluding.
    If everything is failing on f_vol, volume has not dried up anywhere and the
    market is not offering this setup at all.
    """
    p = p or CoilParams()
    as_of = pd.Timestamp(as_of) if as_of is not None else stocks["date"].max()
    d = _day_frame(stocks, p, as_of)
    if d.empty:
        return d
    one = d[d["n_fail"] == 1].copy()
    if one.empty:
        return one
    one["missing"] = one[FLAG_COLS].apply(
        lambda r: [f[2:] for f in FLAG_COLS if not r[f]][0], axis=1
    )
    cols = ["symbol", "sector", "adj", "missing", "pos_hi", "rsi",
            "vol_ratio", "range20", "cmf"]
    cols = [c for c in cols if c in one.columns]
    return one.sort_values("pos_hi", ascending=False)[cols].head(top).reset_index(drop=True)


def forward_test(stocks: pd.DataFrame, p: CoilParams | None = None,
                 horizons=(5, 10, 20), min_history: int = 210) -> pd.DataFrame:
    """
    Run the scan on every historical date and attach forward returns, alongside
    the all-stock average on the same dates.

    The second number is the base rate. Without it the first means nothing.
    """
    p = p or CoilParams()
    s = stocks.sort_values(["symbol", "date"]).copy()
    for h in horizons:
        s[f"fwd_{h}"] = s.groupby("symbol", sort=False)["tri"].transform(
            lambda x, h=h: x.shift(-h) / x - 1.0
        )

    dates = sorted(s["date"].unique())[min_history:]
    rows = []
    for as_of in dates:
        hits = scan(s[s["date"] <= as_of], p, as_of=as_of, top=10_000)
        day = s[s["date"] == as_of]
        rec = {"date": as_of, "n_flagged": len(hits), "n_universe": len(day)}
        for h in horizons:
            col = f"fwd_{h}"
            rec[f"flag_{h}"] = (day[day["symbol"].isin(hits["symbol"])][col].mean()
                                if len(hits) else np.nan)
            rec[f"base_{h}"] = day[col].mean()
            rec[f"edge_{h}"] = rec[f"flag_{h}"] - rec[f"base_{h}"]
        rows.append(rec)
    return pd.DataFrame(rows)