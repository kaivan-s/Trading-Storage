"""
Per-stock pattern mining, with the checks that tell a real stock-specific
edge from a lucky one.

Data: Yahoo daily candles since 2005. Yahoo's OHLC is split-adjusted already;
adjust() also catches any split/bonus it missed from impossible overnight jumps.

Protocol
  1. Discovery (up to DISCOVERY_END): for every stock x pattern x horizon,
     measure the return vs that stock's own average return over the same
     window ("beat just holding it"). A pattern is "found" if n >= MIN_N and
     |t| >= T_FOUND; its sign decides buy or sell side.
  2. Holdout (after DISCOVERY_END): the found patterns, same side, untouched.
  3. Null: FAKE_PER_STOCK random-date "patterns" per stock with the same
     signal frequencies, run through the identical steps.
  4. Pooled: the same pattern across all the other stocks in discovery,
     to see whether a "Dixon pattern" is really a general pattern.

Signals are known at the close (entry next open) except the gap patterns,
which are known at the open (entry that open, exit that close).

    Run:  ~/anaconda3/bin/python attic/stock_patterns.py 2>/dev/null
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

CACHE = os.path.join(HERE, "..", "data", "cache", "groww_daily.parquet")
STOCKS = ["DIXON", "RELIANCE", "HDFCBANK", "INFY", "TATASTEEL", "MARUTI", "BSE", "TRENT"]
START = pd.Timestamp("2005-01-01")
DISCOVERY_END = pd.Timestamp("2019-12-31")
COST = 0.001
MIN_N, T_FOUND = 30, 2.0
FAKE_PER_STOCK = 300
SPLIT_FACTORS = (1.5, 2, 2.5, 3, 4, 5, 10, 20)
rng = np.random.default_rng(3)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def download() -> pd.DataFrame:
    """Yahoo daily history (Groww's daily candles only start in 2020)."""
    import yfinance as yf
    have = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame(columns=["symbol"])
    out = [have]
    for sym in STOCKS:
        if sym in set(have["symbol"]):
            continue
        h = yf.Ticker(f"{sym}.NS").history(period="max", auto_adjust=False, actions=False)
        h = h.reset_index().rename(columns=str.lower)
        h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None).dt.normalize()
        d = h[["date", "open", "high", "low", "close", "volume"]].copy()
        d = d[(d.date >= START) & (d.volume > 0)]
        d["symbol"] = sym
        out.append(d)
        print(f"  {sym}: {len(d)} days from {d.date.min().date()}")
        time.sleep(0.5)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(CACHE)
    return df


def adjust(d: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Back-adjust splits/bonuses: an overnight ratio matching a standard factor."""
    d = d.sort_values("date").reset_index(drop=True).copy()
    events = []
    for i in range(1, len(d)):
        ratio = d.at[i - 1, "close"] / d.at[i, "open"]
        if 0.7 < ratio < 1.4:
            continue
        f = min(SPLIT_FACTORS, key=lambda x: abs(ratio / x - 1))
        if abs(ratio / f - 1) < 0.15:
            d.loc[:i - 1, ["open", "high", "low", "close"]] /= f
            d.loc[:i - 1, "volume"] *= f
            events.append((d.at[i, "date"].date(), f))
    return d, events


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

def rsi(c: pd.Series, n: int) -> pd.Series:
    ch = c.diff()
    up = ch.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-ch.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def patterns(d: pd.DataFrame) -> tuple[dict, dict]:
    """Close-known signals (entry next open) and open-known gap signals (same day)."""
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    pc = c.shift(1)
    ret = c / pc - 1
    rng_ = h - l
    atr = rng_.rolling(14).mean()
    sma10, sma20, sma50, sma200 = (c.rolling(n).mean() for n in (10, 20, 50, 200))
    loc = (c - l) / rng_.replace(0, np.nan)
    vx = v / v.shift(1).rolling(20).mean()
    r2 = rsi(c, 2)
    gap = o / pc - 1
    dow = d["date"].dt.dayofweek
    month_end = d["date"].dt.month != d["date"].shift(-1).dt.month

    close_sig = {
        "up day >= 3%": ret >= 0.03,
        "down day >= 3%": ret <= -0.03,
        "3 up closes in a row": (ret > 0) & (ret.shift(1) > 0) & (ret.shift(2) > 0),
        "3 down closes in a row": (ret < 0) & (ret.shift(1) < 0) & (ret.shift(2) < 0),
        "NR7 (narrowest range of 7)": rng_ == rng_.rolling(7).min(),
        "inside day": (h < h.shift(1)) & (l > l.shift(1)),
        "close in top 10% of range": loc >= 0.9,
        "close in bottom 10% of range": loc <= 0.1,
        "20-day closing high": c >= c.rolling(20).max(),
        "20-day closing low": c <= c.rolling(20).min(),
        "52-week closing high": c >= c.rolling(250).max(),
        "RSI(2) < 10": r2 < 10,
        "RSI(2) > 90": r2 > 90,
        "RSI(2) < 10 above 200-SMA": (r2 < 10) & (c > sma200),
        "crossed above 20-SMA": (c > sma20) & (pc <= sma20.shift(1)),
        "crossed below 20-SMA": (c < sma20) & (pc >= sma20.shift(1)),
        "pullback below 10-SMA in uptrend": (c < sma10) & (c > sma50) & (sma50 > sma200),
        "volume >= 2x, up day": (vx >= 2) & (ret > 0),
        "volume >= 2x, down day": (vx >= 2) & (ret < 0),
        "bullish outside day": (h > h.shift(1)) & (l < l.shift(1)) & (c > o) & (c > h.shift(1)),
        "bearish outside day": (h > h.shift(1)) & (l < l.shift(1)) & (c < o) & (c < l.shift(1)),
        "range >= 2x ATR, up close": (rng_ >= 2 * atr.shift(1)) & (ret > 0),
        "range >= 2x ATR, down close": (rng_ >= 2 * atr.shift(1)) & (ret < 0),
        "stretched 2 ATR below 20-SMA": c < sma20 - 2 * atr,
        "stretched 2 ATR above 20-SMA": c > sma20 + 2 * atr,
        "gap down >= 1%, closed green": (gap <= -0.01) & (c > o),
        "gap up >= 1%, closed red": (gap >= 0.01) & (c < o),
        "Monday close": dow == 0,
        "Friday close": dow == 4,
        "month-end close": month_end,
    }
    open_sig = {
        "gap up >= 2% (same day)": gap >= 0.02,
        "gap down >= 2% (same day)": gap <= -0.02,
        "gap up 0.5-2% (same day)": (gap >= 0.005) & (gap < 0.02),
        "gap down 0.5-2% (same day)": (gap <= -0.005) & (gap > -0.02),
    }
    return ({k: v.fillna(False).to_numpy() for k, v in close_sig.items()},
            {k: v.fillna(False).to_numpy() for k, v in open_sig.items()})


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------

def outcomes(d: pd.DataFrame) -> dict[str, np.ndarray]:
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    n = len(c)
    nxt_o = np.r_[o[1:], np.nan]
    out = {
        "1d": np.r_[c[1:], np.nan] / nxt_o - 1,                       # next open -> next close
        "5d": np.r_[c[5:], [np.nan] * 5][:n] / nxt_o - 1,             # next open -> close 5 days on
        "same": c / o - 1,                                             # gap patterns
    }
    return out


def stat(sig: np.ndarray, r: np.ndarray, mask: np.ndarray, hold: int) -> tuple[int, float, float]:
    """n, mean excess return vs the window average, t (non-overlapping events)."""
    ok = mask & np.isfinite(r)
    base = np.nanmean(r[ok])
    idx = np.where(sig & ok)[0]
    if hold > 1 and len(idx):
        keep, last = [], -10**9
        for i in idx:
            if i - last >= hold:
                keep.append(i)
                last = i
        idx = np.array(keep)
    x = r[idx] - base
    if len(x) < 3:
        return len(x), np.nan, np.nan
    return len(x), float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def evaluate(sig, outs, disc, hold_mask, hz):
    r = outs[hz]
    hold = 5 if hz == "5d" else 1
    n1, m1, t1 = stat(sig, r, disc, hold)
    n2, m2, t2 = stat(sig, r, hold_mask, hold)
    return n1, m1, t1, n2, m2, t2


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    df = download()
    df["date"] = pd.to_datetime(df["date"])
    found, fake_found, all_tests = [], [], []
    per_stock = {}
    for sym in STOCKS:
        d, splits = adjust(df[df.symbol == sym])
        d = d[(d.close > 0)].reset_index(drop=True)
        disc = (d["date"] <= DISCOVERY_END).to_numpy()
        hold = ~disc
        outs = outcomes(d)
        close_sig, open_sig = patterns(d)
        per_stock[sym] = dict(d=d, outs=outs, close_sig=close_sig, open_sig=open_sig, disc=disc, hold=hold)
        print(f"{sym}: {d.date.min().date()} -> {d.date.max().date()}, discovery {disc.sum()} days, "
              f"holdout {hold.sum()} days, splits/bonuses adjusted: {splits}")

        tests = [(name, s, hz) for name, s in close_sig.items() for hz in ("1d", "5d")]
        tests += [(name, s, "same") for name, s in open_sig.items()]
        for name, s, hz in tests:
            n1, m1, t1, n2, m2, t2 = evaluate(s, outs, disc, hold, hz)
            row = dict(stock=sym, pattern=name, hz=hz, n_disc=n1, ex_disc=m1, t_disc=t1,
                       n_hold=n2, ex_hold=m2, t_hold=t2)
            all_tests.append(row)
            if n1 >= MIN_N and abs(t1) >= T_FOUND:
                found.append(row)
            # fake patterns: same frequency, random dates
        freqs = [s.mean() for _, s, _ in tests]
        for k in range(FAKE_PER_STOCK):
            p = freqs[k % len(freqs)]
            s = rng.random(len(d)) < p
            hz = ("1d", "5d", "same")[k % 3]
            n1, m1, t1, n2, m2, t2 = evaluate(s, outs, disc, hold, hz)
            if n1 >= MIN_N and abs(t1) >= T_FOUND:
                fake_found.append(dict(stock=sym, hz=hz, ex_disc=m1, t_disc=t1, n_hold=n2, ex_hold=m2, t_hold=t2))

    tests = pd.DataFrame(all_tests)
    f = pd.DataFrame(found)
    ff = pd.DataFrame(fake_found)

    # pooled check: the same pattern across the other stocks, discovery period
    pooled = []
    for r in f.itertuples():
        xs = []
        for sym, P in per_stock.items():
            if sym == r.stock:
                continue
            sig = (P["close_sig"].get(r.pattern) if r.hz != "same" else P["open_sig"].get(r.pattern))
            rr = P["outs"][r.hz]
            ok = P["disc"] & np.isfinite(rr)
            base = np.nanmean(rr[ok])
            xs.append(rr[sig & ok] - base)
        x = np.concatenate(xs)
        pooled.append(np.sign(x.mean()) == np.sign(r.ex_disc) and abs(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) >= 2)
    f["general"] = pooled

    def side_adj(frame):
        s = np.sign(frame["ex_disc"])
        frame = frame.copy()
        frame["hold_signed"] = s * frame["ex_hold"]
        frame["hold_t_signed"] = s * frame["t_hold"]
        return frame

    f, ff = side_adj(f), side_adj(ff)
    total_tests = len(tests)
    print(f"\n{total_tests} real tests ({len(STOCKS)} stocks x {total_tests // len(STOCKS)}), "
          f"found in discovery: {len(f)} ({100 * len(f) / total_tests:.1f}%)")
    print(f"fake random patterns: {len(ff)}/{FAKE_PER_STOCK * len(STOCKS)} found "
          f"({100 * len(ff) / (FAKE_PER_STOCK * len(STOCKS)):.1f}%)  <- what pure chance produces")

    def summary(frame, label):
        fr = frame.dropna(subset=["hold_signed"])
        print(f"  {label:<38} n={len(fr):>3}  same side in holdout: {100 * (fr.hold_signed > 0).mean():>3.0f}%   "
              f"still |t|>=2 same side: {100 * (fr.hold_t_signed >= 2).mean():>3.0f}%   "
              f"avg holdout edge: {100 * fr.hold_signed.mean():+.3f}%/trade (before {100 * COST:.1f}% cost)")

    print(f"\nDid the 'found' patterns keep working after {DISCOVERY_END.date()}?")
    summary(f, "real patterns, all")
    summary(f[f.general], "  ...also works on other stocks")
    summary(f[~f.general], "  ...only this stock (the idea)")
    summary(ff, "fake random patterns (chance)")

    print("\nPer stock: patterns found in discovery -> holdout")
    pd.set_option("display.width", 250)
    show = f.assign(
        side=np.where(f.ex_disc > 0, "BUY", "SELL"),
        disc=lambda x: (100 * x.ex_disc).round(2).astype(str) + "% (t " + x.t_disc.round(1).astype(str) + ", n " + x.n_disc.astype(str) + ")",
        holdout=lambda x: (100 * x.hold_signed).round(2).astype(str) + "% (t " + x.hold_t_signed.round(1).astype(str) + ", n " + x.n_hold.astype(str) + ")",
    )[["stock", "side", "pattern", "hz", "disc", "holdout", "general"]]
    print(show.sort_values(["stock", "hz"]).to_string(index=False))

    out = os.path.join(HERE, "stock_patterns_all_tests.csv")
    tests.to_csv(out, index=False)
    print(f"\nall tests written to {out}")


if __name__ == "__main__":
    main()
