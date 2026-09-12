"""
Turn raw bhavcopy rows into a sector x day panel carrying T, T_rel, B, CMF and RS.

Everything the scan needs is computed here. The gotchas that silently corrupt
these numbers are handled explicitly and commented, because each of them
produces a plausible-looking wrong answer rather than an error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean(df: pd.DataFrame, sector_map: pd.DataFrame,
          sector_level: str = "basic_industry",
          min_median_turnover_lacs: float = 20.0,
          min_sector_stocks: int = 4) -> pd.DataFrame:
    """
    Filter to real equities, attach sector, drop illiquid noise.

    The liquidity floor is not cosmetic. A microcap that traded once at its
    circuit price produces a valid-looking money-flow multiplier of +/-1 and,
    in a plain sector average, counts exactly as much as a stock that turned
    over 200 crore. Sectors like Sugar and Paper Products have long tails of
    these.
    """
    df = df.copy()

    # SERIES carries SME, ETFs, debt and rights alongside equities.
    df = df[df["series"].str.upper() == "EQ"]

    df = df.merge(sector_map[["symbol", sector_level]], on="symbol", how="left")
    df = df.rename(columns={sector_level: "sector"})
    df = df[df["sector"].notna() & (df["sector"].astype(str).str.strip() != "")]

    df = df[(df["close"] > 0) & (df["volume"] > 0) & (df["turnover"] > 0)]

    med = df.groupby("symbol")["turnover"].median()
    keep = med[med >= min_median_turnover_lacs].index
    df = df[df["symbol"].isin(keep)]

    counts = df.groupby("sector")["symbol"].nunique()
    df = df[df["sector"].isin(counts[counts >= min_sector_stocks].index)]

    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Per-stock metrics
# --------------------------------------------------------------------------

def stock_metrics(df: pd.DataFrame, cmf_window: int = 20,
                  rs_lookback: int = 55,
                  deliv_baseline: int = 60) -> pd.DataFrame:
    """
    Direction, money-flow volume, delivery value, delivery quality, 55-day return.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)

    # --- direction -------------------------------------------------------
    # PREV_CLOSE ships in the file, so no shift is needed. This matters:
    # a shift would misalign across suspensions, and Indian smallcaps get
    # suspended often.
    df["chg"] = df["close"] - df["prev_close"]
    df["dir"] = np.sign(df["chg"]).fillna(0)

    # --- corporate actions ----------------------------------------------
    # Bhavcopy prices are unadjusted, BUT NSE adjusts PREV_CLOSE on ex-dates.
    # So close/prev_close is already a clean daily total return, and a
    # cumulative product of it gives an adjusted price series for free.
    # Without this a 1:10 split reads as a 90% crash and drops the stock — and
    # its whole sector — to the bottom of the RS ranking.
    ret = (df["close"] / df["prev_close"]) - 1.0
    df["ret"] = ret.where(ret.abs() < 0.6)          # guard against bad prints
    df["tri"] = g["ret"].transform(lambda s: (1.0 + s.fillna(0)).cumprod())
    df["ret_lb"] = g["tri"].transform(
        lambda s: s / s.shift(rs_lookback) - 1.0
    )

    # --- Chaikin money flow ---------------------------------------------
    rng = df["high"] - df["low"]
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    # Circuit-locked stocks have high == low. Without this guard the division
    # yields inf, which propagates through the rolling sum and blanks the
    # sector. Indian smallcaps hit circuits constantly.
    df["mfm"] = mfm.where(rng > 0, 0.0)
    df["mfv"] = df["mfm"] * df["volume"]

    g = df.groupby("symbol", sort=False)
    num = g["mfv"].transform(lambda s: s.rolling(cmf_window, min_periods=int(cmf_window * 0.75)).sum())
    den = g["volume"].transform(lambda s: s.rolling(cmf_window, min_periods=int(cmf_window * 0.75)).sum())
    df["cmf"] = (num / den).replace([np.inf, -np.inf], np.nan)

    # --- delivery --------------------------------------------------------
    # Delivery VALUE is the honest weight for breadth: it ignores intraday
    # churn and counts only money that actually took possession of shares.
    df["deliv_val"] = df["turnover"] * df["deliv_pct"] / 100.0

    # Delivery QUALITY = today's delivery % against the stock's own norm.
    # This is the discriminator that separates accumulation from a churn day:
    # delivery % falls mechanically on huge volume as day traders inflate the
    # denominator, so raw delivery % on a spike day is misleading on its own.
    base = g["deliv_pct"].transform(
        lambda s: s.shift(1).rolling(deliv_baseline, min_periods=15).mean()
    )
    df["deliv_quality"] = df["deliv_pct"] / base

    return df


# --------------------------------------------------------------------------
# Sector aggregation
# --------------------------------------------------------------------------

def sector_panel(df: pd.DataFrame, turnover_window: int = 9) -> pd.DataFrame:
    """
    Collapse stock rows into one row per sector per day.

    Returns a long frame indexed by (date, sector) with the four inputs plus
    the quality and width columns the gates need.
    """
    df = df.copy()
    key = ["date", "sector"]

    # --- cross-sectional RS rank ----------------------------------------
    # Ranking by return alone gives the same ordering as return / index return,
    # because the index return is a common divisor on any given day. So no
    # benchmark series is needed.
    df["rs"] = df.groupby("date")["ret_lb"].rank(pct=True) * 100.0

    adv = df["dir"] > 0
    dec = df["dir"] < 0

    # Turnover-weighted sector return, so forward performance can be measured.
    df["_wret"] = df["ret"] * df["turnover"]

    grouped = df.groupby(key, sort=True)

    out = pd.DataFrame({
        "ret": grouped["_wret"].sum() / grouped["turnover"].sum(),
        "turnover": grouped["turnover"].sum(),
        "volume": grouped["volume"].sum(),
        "deliv_val": grouped["deliv_val"].sum(),
        "n_stocks": grouped["symbol"].nunique(),
        "cmf": grouped["cmf"].mean(),
        "rs": grouped["rs"].mean(),
        # Median is deliberate: one microcap printing 4x should not carry a
        # sector, and the mean lets it.
        "deliv_quality": grouped["deliv_quality"].median(),
        "top_share": grouped["turnover"].max() / grouped["turnover"].sum(),
    })

    adv_t = df[adv].groupby(key)["turnover"].sum()
    dec_t = df[dec].groupby(key)["turnover"].sum()
    out["n_adv"] = df[adv].groupby(key)["symbol"].nunique()
    out["n_dec"] = df[dec].groupby(key)["symbol"].nunique()

    adv_t = adv_t.reindex(out.index).fillna(0.0)
    dec_t = dec_t.reindex(out.index).fillna(0.0)
    denom = (adv_t + dec_t).replace(0, np.nan)
    out["B"] = (adv_t - dec_t) / denom

    # Delivery-weighted breadth: same split, but on money that took delivery.
    adv_d = df[adv].groupby(key)["deliv_val"].sum().reindex(out.index).fillna(0.0)
    dec_d = df[dec].groupby(key)["deliv_val"].sum().reindex(out.index).fillna(0.0)
    dd = (adv_d + dec_d).replace(0, np.nan)
    out["B_deliv"] = (adv_d - dec_d) / dd

    out[["n_adv", "n_dec"]] = out[["n_adv", "n_dec"]].fillna(0).astype(int)
    out = out.reset_index()

    # --- turnover ratio --------------------------------------------------
    wide = out.pivot(index="date", columns="sector", values="turnover").sort_index()
    # shift(1) is load-bearing. Including today in its own baseline drags the
    # denominator up and shrinks the very spike you are trying to detect:
    # a genuine 3x day reads as roughly 2.4x without it.
    base = wide.shift(1).rolling(turnover_window, min_periods=5).mean()
    T = wide / base

    # T_rel divides by the cross-sector median that day. On a market-wide
    # heavy-volume day every sector's raw T rises together, so raw T carries no
    # sector-specific information; the median rises too, and T_rel does not.
    T_rel = T.div(T.median(axis=1), axis=0)

    T_long = T.stack().rename("T").reset_index()
    Tr_long = T_rel.stack().rename("T_rel").reset_index()
    out = out.merge(T_long, on=["date", "sector"], how="left")
    out = out.merge(Tr_long, on=["date", "sector"], how="left")

    # --- delivery quality, cross-sectionally normalised ------------------
    # Raw delivery quality is confounded: delivery PERCENTAGE rises mechanically
    # whenever volume falls, because day traders inflate the denominator on busy
    # days and vanish on quiet ones. So on any market-wide quiet day nearly every
    # sector prints "improving delivery" and the signal is a tautology.
    # Dividing by the cross-sector median that day removes the common component,
    # exactly as T_rel does for turnover. Use this, not `deliv_quality`, in gates.
    out["deliv_quality_rel"] = out["deliv_quality"] / out.groupby("date")[
        "deliv_quality"
    ].transform("median")

    # CMF carries a large market-wide component: in a weak tape nearly every
    # sector prints negative, and an absolute test against zero then reports the
    # market rather than discriminating between sectors. Compare against peers.
    out["cmf_rel"] = out["cmf"] - out.groupby("date")["cmf"].transform("median")

    # --- sector-level history -------------------------------------------
    out = out.sort_values(["sector", "date"])
    gs = out.groupby("sector", sort=False)
    out["rs_chg_5"] = gs["rs"].transform(lambda s: s - s.shift(5))
    out["cmf_chg_5"] = gs["cmf"].transform(lambda s: s - s.shift(5))
    out["cmf_rel_chg_5"] = gs["cmf_rel"].transform(lambda s: s - s.shift(5))
    out["B_green_10"] = gs["B"].transform(
        lambda s: (s > 0).rolling(10, min_periods=6).sum()
    )
    out["T_quiet_10"] = gs["T"].transform(
        lambda s: (s < 1.0).rolling(10, min_periods=6).sum()
    )
    out["T_quiet_prior_5"] = gs["T"].transform(
        lambda s: (s.shift(1) < 1.2).rolling(5, min_periods=3).sum()
    )
    out["T_max_prior_5"] = gs["T"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).max()
    )

    return out.reset_index(drop=True)


def build(raw: pd.DataFrame, sector_map: pd.DataFrame, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw bhavcopy + sector map -> (stock frame, sector panel)."""
    cleaned = clean(raw, sector_map,
                    sector_level=kw.get("sector_level", "basic_industry"),
                    min_median_turnover_lacs=kw.get("min_median_turnover_lacs", 20.0),
                    min_sector_stocks=kw.get("min_sector_stocks", 4))
    stocks = stock_metrics(cleaned,
                           cmf_window=kw.get("cmf_window", 20),
                           rs_lookback=kw.get("rs_lookback", 55),
                           deliv_baseline=kw.get("deliv_baseline", 60))
    panel = sector_panel(stocks, turnover_window=kw.get("turnover_window", 9))
    return stocks, panel
