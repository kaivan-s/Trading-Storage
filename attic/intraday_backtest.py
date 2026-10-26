"""
Intraday (open -> close) strategy backtest on NSE cash equities, daily bhavcopy.

Every signal uses only information available at or before the 09:15 open:
prior-day bars plus today's open price. Trades exit at the close. Stop-loss
rules are exact on daily bars (a long is out once the low touches the stop);
breakout-entry rules enter at the trigger level only if the day's range
reaches it.

Run:  ~/anaconda3/bin/python attic/intraday_backtest.py 2>/dev/null
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")

TEST_START = pd.Timestamp("2026-04-14")   # last 6 months
PRIOR_START = pd.Timestamp("2025-10-14")  # prior 6 months, as a robustness check
LOAD_FROM = "20250301"

COST = 0.0010          # round-trip: brokerage + STT + charges + slippage (10 bps)
MIN_TURNOVER_CR = 10   # 20-day median turnover, rupees crore
MIN_PRICE = 50
MAX_PER_DAY = 5        # portfolio takes at most this many trades a day


def load() -> pd.DataFrame:
    files = sorted(f for f in glob.glob(os.path.join(RAW, "bhav_*.parquet"))
                   if os.path.basename(f)[5:13] >= LOAD_FROM)
    cols = ["symbol", "series", "date", "prev_close", "open", "high", "low",
            "close", "vwap", "volume", "turnover", "deliv_pct"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    df = df[df["series"] == "EQ"].drop(columns="series")
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["prev_close"] > 0)]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def fno_symbols() -> set[str]:
    files = sorted(glob.glob(os.path.join(RAW, "fo", "fo_*.csv.gz")))
    if not files:
        return set()
    fo = pd.read_csv(files[-1])
    col = next(c for c in fo.columns if c.lower() in ("tckrsymb", "symbol"))
    return set(fo[col].astype(str).str.strip())


def features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("symbol", sort=False)
    df["gap"] = df["open"] / df["prev_close"] - 1
    df["oc"] = df["close"] / df["open"] - 1                # the intraday trade
    df["up_x"] = df["high"] / df["open"] - 1               # max favourable for long
    df["dn_x"] = df["low"] / df["open"] - 1                # max adverse for long

    # Prior-day facts (shifted, so known before today's open).
    df["r1"] = g["close"].shift(1) / g["close"].shift(2) - 1
    df["r5"] = g["close"].shift(1) / g["close"].shift(6) - 1
    prev_high, prev_low = g["high"].shift(1), g["low"].shift(1)
    df["prev_high"], df["prev_low"] = prev_high, prev_low
    df["prev_close_loc"] = (g["close"].shift(1) - prev_low) / (prev_high - prev_low)
    rng = (df["high"] - df["low"]) / df["prev_close"]
    df["atr"] = rng.groupby(df["symbol"]).transform(
        lambda s: s.shift(1).rolling(14, min_periods=10).mean())
    df["vol_x"] = df["volume"] / g["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    df["prev_vol_x"] = g["vol_x"].shift(1)
    df["med_to"] = g["turnover"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).median()) / 100  # lakhs -> crore
    c1 = g["close"].shift(1)
    df["sma20"] = c1.groupby(df["symbol"]).transform(lambda s: s.rolling(20).mean())
    df["sma50"] = c1.groupby(df["symbol"]).transform(lambda s: s.rolling(50).mean())
    df["hi20"] = g["high"].transform(lambda s: s.shift(1).rolling(20).max())
    df["nr7"] = rng.groupby(df["symbol"]).transform(
        lambda s: s.shift(1) <= s.shift(1).rolling(7).min())
    df["prev_deliv"] = g["deliv_pct"].shift(1)

    # Market context: median prior-day return and today's median gap.
    df["mkt_r1"] = df.groupby("date")["r1"].transform("median")
    df["mkt_gap"] = df.groupby("date")["gap"].transform("median")
    df["frozen"] = df["high"] == df["low"]
    return df


# --------------------------------------------------------------------------
# Trade return helpers. All return gross fractional P&L per unit.
# --------------------------------------------------------------------------

def long_oc(d, stop=None):
    r = d["oc"].copy()
    if stop:
        r = np.where(d["dn_x"] <= -stop, -stop, r)
    return pd.Series(r, index=d.index)


def short_oc(d, stop=None):
    r = -d["oc"].copy()
    if stop:
        r = np.where(d["up_x"] >= stop, -stop, r)
    return pd.Series(r, index=d.index)


def long_breakout(d, level):
    """Buy-stop at `level` (above open). Filled only if the high reaches it."""
    hit = d["high"] >= level
    entry = np.maximum(level, d["open"])
    r = d["close"] / entry - 1
    return pd.Series(np.where(hit, r, np.nan), index=d.index)


def short_breakdown(d, level):
    hit = d["low"] <= level
    entry = np.minimum(level, d["open"])
    r = 1 - d["close"] / entry
    return pd.Series(np.where(hit, r, np.nan), index=d.index)


# --------------------------------------------------------------------------
# Strategies: (name, side, mask_fn, return_fn, rank_fn)
# rank_fn picks the MAX_PER_DAY strongest signals each day (higher = first).
# --------------------------------------------------------------------------

def strategies():
    S = []

    for g_ in (0.02, 0.03, 0.05):
        S.append((f"Gap-down fade: buy gap <= -{g_:.0%}", "long",
                  lambda d, g_=g_: (d.gap <= -g_) & (d.gap > -0.09),
                  lambda d: long_oc(d), lambda d: -d.gap))
        S.append((f"Gap-up fade: short gap >= +{g_:.0%}", "short",
                  lambda d, g_=g_: (d.gap >= g_) & (d.gap < 0.09),
                  lambda d: short_oc(d), lambda d: d.gap))

    for g_ in (0.02, 0.03):
        S.append((f"Gap-up go: buy gap >= +{g_:.0%}", "long",
                  lambda d, g_=g_: (d.gap >= g_) & (d.gap < 0.09),
                  lambda d: long_oc(d), lambda d: d.gap))
        S.append((f"Gap-down go: short gap <= -{g_:.0%}", "short",
                  lambda d, g_=g_: (d.gap <= -g_) & (d.gap > -0.09),
                  lambda d: short_oc(d), lambda d: -d.gap))

    # Gap fade only in the stock's trend direction (buy dips in uptrends).
    S.append(("Uptrend dip: gap <= -2%, prior close > SMA50", "long",
              lambda d: (d.gap <= -0.02) & (d.gap > -0.09) & (d.prev_close > d.sma50),
              lambda d: long_oc(d), lambda d: -d.gap))
    S.append(("Downtrend pop: short gap >= +2%, prior close < SMA50", "short",
              lambda d: (d.gap >= 0.02) & (d.gap < 0.09) & (d.prev_close < d.sma50),
              lambda d: short_oc(d), lambda d: d.gap))

    # Idiosyncratic gap: stock gaps against a flat/positive market.
    S.append(("Lone gap-down fade: gap <= -3% while market gap >= 0", "long",
              lambda d: (d.gap <= -0.03) & (d.gap > -0.09) & (d.mkt_gap >= 0),
              lambda d: long_oc(d), lambda d: -d.gap))
    S.append(("Lone gap-up fade: short gap >= +3% while market gap <= 0", "short",
              lambda d: (d.gap >= 0.03) & (d.gap < 0.09) & (d.mkt_gap <= 0),
              lambda d: short_oc(d), lambda d: d.gap))

    # Day-after big move.
    for m in (0.05, 0.08):
        S.append((f"Day-after loser bounce: buy after r1 <= -{m:.0%}", "long",
                  lambda d, m=m: d.r1 <= -m, lambda d: long_oc(d), lambda d: -d.r1))
        S.append((f"Day-after winner fade: short after r1 >= +{m:.0%}", "short",
                  lambda d, m=m: d.r1 >= m, lambda d: short_oc(d), lambda d: d.r1))
    S.append(("Winner follow-through: buy after r1 >= +5% closing at high, vol 2x", "long",
              lambda d: (d.r1 >= 0.05) & (d.prev_close_loc >= 0.85) & (d.prev_vol_x >= 2),
              lambda d: long_oc(d), lambda d: d.r1))
    S.append(("Weak-close winner fade: short after r1 >= +5% closing in bottom third", "short",
              lambda d: (d.r1 >= 0.05) & (d.prev_close_loc <= 0.33),
              lambda d: short_oc(d), lambda d: d.r1))

    # Breakouts of prior-day high/low and opening-range proxies.
    S.append(("Prior-high breakout: buy-stop at prev high (opened below it)", "long",
              lambda d: d.open < d.prev_high,
              lambda d: long_breakout(d, d.prev_high), lambda d: d.prev_vol_x))
    S.append(("NR7 breakout: buy-stop at prev high after narrowest range in 7", "long",
              lambda d: d.nr7.astype(bool) & (d.open < d.prev_high),
              lambda d: long_breakout(d, d.prev_high), lambda d: -d.atr))
    S.append(("NR7 breakdown: sell-stop at prev low after narrowest range in 7", "short",
              lambda d: d.nr7.astype(bool) & (d.open > d.prev_low),
              lambda d: short_breakdown(d, d.prev_low), lambda d: -d.atr))
    for k in (0.5, 1.0):
        S.append((f"Open+{k}xATR breakout: buy-stop above open, near 20d high", "long",
                  lambda d: (d.prev_close >= 0.97 * d.hi20) & (d.prev_close > d.sma20),
                  lambda d, k=k: long_breakout(d, d.open * (1 + k * d.atr)),
                  lambda d: d.r5))
        S.append((f"Open-{k}xATR breakdown: sell-stop below open, below SMA20", "short",
                  lambda d: (d.prev_close < d.sma20) & (d.r5 < 0),
                  lambda d, k=k: short_breakdown(d, d.open * (1 - k * d.atr)),
                  lambda d: -d.r5))

    # Gap fades with a stop, to see if a stop helps or hurts.
    for st in (0.02, 0.03):
        S.append((f"Gap-down fade -3% with {st:.0%} stop", "long",
                  lambda d: (d.gap <= -0.03) & (d.gap > -0.09),
                  lambda d, st=st: long_oc(d, stop=st), lambda d: -d.gap))
        S.append((f"Gap-up fade +3% with {st:.0%} stop", "short",
                  lambda d: (d.gap >= 0.03) & (d.gap < 0.09),
                  lambda d, st=st: short_oc(d, stop=st), lambda d: d.gap))
    return S


def run(df: pd.DataFrame, start, end, universe_mask):
    period = df[(df.date >= start) & (df.date < end) & universe_mask]
    days = period["date"].drop_duplicates().sort_values()
    out = []
    for name, side, mask_fn, ret_fn, rank_fn in strategies():
        sel = period[mask_fn(period).fillna(False)].copy()
        if sel.empty:
            continue
        sel["ret"] = ret_fn(sel) - COST
        sel = sel[sel["ret"].notna()]
        sel["rank"] = rank_fn(sel)
        sel = (sel.sort_values(["date", "rank"], ascending=[True, False])
                  .groupby("date").head(MAX_PER_DAY))
        if len(sel) < 20:
            continue
        daily = sel.groupby("date")["ret"].mean().reindex(days, fill_value=0.0)
        equity = (1 + daily).cumprod()
        dd = (equity / equity.cummax() - 1).min()
        monthly = daily.groupby(daily.index.to_period("M")).sum()
        active = daily[daily != 0]
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        out.append({
            "strategy": name, "side": side,
            "trades": len(sel), "days_active": len(active),
            "win%": (sel["ret"] > 0).mean() * 100,
            "avg_trade%": sel["ret"].mean() * 100,
            "total%": (equity.iloc[-1] - 1) * 100,
            "sharpe": sharpe, "maxDD%": dd * 100,
            "pos_months": f"{(monthly > 0).sum()}/{len(monthly)}",
        })
    return pd.DataFrame(out)


def main():
    df = features(load())
    fno = fno_symbols()
    liquid = (df.med_to >= MIN_TURNOVER_CR) & (df.prev_close >= MIN_PRICE) & ~df.frozen
    non_fno = liquid & ~df.symbol.isin(fno)

    end = df.date.max() + pd.Timedelta(days=1)
    pd.set_option("display.width", 220, "display.max_colwidth", 70,
                  "display.max_rows", 200)
    fmt = {c: "{:.2f}".format for c in ("win%", "avg_trade%", "total%", "sharpe", "maxDD%")}

    print(f"Data {df.date.min().date()} -> {df.date.max().date()}, "
          f"F&O names known: {len(fno)}, cost {COST*1e4:.0f} bps round-trip, "
          f"max {MAX_PER_DAY} trades/day equal-weight\n")

    for label, mask in (("ALL LIQUID CASH STOCKS", liquid), ("NON-F&O LIQUID STOCKS ONLY", non_fno)):
        last6 = run(df, TEST_START, end, mask)
        prior6 = run(df, PRIOR_START, TEST_START, mask)
        m = last6.merge(prior6[["strategy", "avg_trade%", "sharpe", "total%"]],
                        on="strategy", how="left", suffixes=("", "_prior6m"))
        m = m.sort_values("sharpe", ascending=False)
        print(f"=== {label}: last 6 months ({TEST_START.date()} -> {df.date.max().date()}) ===")
        print(m.to_string(index=False, formatters={**fmt,
              "avg_trade%_prior6m": "{:.2f}".format, "sharpe_prior6m": "{:.2f}".format,
              "total%_prior6m": "{:.1f}".format}))
        print()


if __name__ == "__main__":
    main()
