"""
Re-price the daily-bar gap strategies with 5-minute bars (Yahoo Finance).

The daily backtest fills at NSE's official open, which is the pre-open auction
price, and exits at the official close. Here the same signals are traded the
way a person could actually trade them:

  * entry at the open of the 09:20 bar (first five minutes of trading skipped)
    or at the 09:25 bar for a slower fill
  * exit at the close of the 15:10 bar (MIS square-off time, ~15:15)
  * stops checked bar by bar after entry; a bar that opens beyond the stop
    fills at that bar's open

Yahoo keeps only ~60 days of 5-minute history, so this covers late Jul 2026
onward.

Run:  ~/anaconda3/bin/python attic/intraday_5m_check.py 2>/dev/null
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yfinance as yf

import intraday_backtest as bt
import intraday_robustness as rb

CACHE = os.path.join(bt.HERE, "..", "data", "cache", "intraday_5m.parquet")
COST = 0.0025
FLOOR_CR = 10


def signals(df, start, end, fno):
    u = ((df.date >= start) & (df.date < end) & (df.prev_close >= bt.MIN_PRICE)
         & ~df.frozen & (df.med_to >= FLOOR_CR) & ~df.symbol.isin(fno))
    p = df[u]
    out = {
        "Downtrend pop short": (rb.trades(p, rb.downtrend_pop, 0, 5), "short"),
        "Gap-up fade short": (rb.trades(p, rb.gap_up_fade, 0, 5), "short"),
        "Gap-down fade long": (rb.trades(p, rb.gap_down_fade, 0, 5), "long"),
        "Book: 3 gap-down longs": (rb.trades(p, rb.gap_down_fade, 0, 3), "long"),
        "Book: 3 gap-up shorts": (rb.trades(p, rb.gap_up_fade, 0, 3), "short"),
    }
    return out


def download(symbols):
    have = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
    done = set(have["symbol"].unique()) if len(have) else set()
    need = [s for s in symbols if s not in done]
    frames = [have] if len(have) else []
    for i in range(0, len(need), 40):
        chunk = need[i:i + 40]
        raw = yf.download([f"{s}.NS" for s in chunk], period="60d", interval="5m",
                          group_by="ticker", progress=False, auto_adjust=False,
                          threads=True)
        for s in chunk:
            t = f"{s}.NS"
            if t not in raw.columns.get_level_values(0):
                continue
            d = raw[t].dropna(subset=["Open"]).reset_index()
            if d.empty:
                continue
            d = d.rename(columns={"Datetime": "ts", "Open": "o", "High": "h",
                                  "Low": "l", "Close": "c", "Volume": "v"})
            d["ts"] = d["ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            d["symbol"] = s
            frames.append(d[["symbol", "ts", "o", "h", "l", "c", "v"]])
        print(f"downloaded {min(i + 40, len(need))}/{len(need)}")
    bars = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "ts"])
    bars.to_parquet(CACHE)
    return bars


def reprice(trades, side, bars_by, entry_time, stop=None):
    rows = []
    for t in trades.itertuples():
        b = bars_by.get((t.symbol, t.date.normalize()))
        if b is None:
            continue
        hhmm = b["ts"].dt.strftime("%H:%M")
        start = b[hhmm >= entry_time]
        end_bar = b[hhmm <= "15:10"]
        if start.empty or end_bar.empty:
            continue
        entry = start["o"].iloc[0]
        path = b[(hhmm >= entry_time) & (hhmm <= "15:10")]
        exit_px = end_bar["c"].iloc[-1]
        if stop:
            if side == "long":
                lvl = entry * (1 - stop)
                hit = path[path["l"] <= lvl]
                if len(hit):
                    exit_px = min(lvl, hit["o"].iloc[0])
            else:
                lvl = entry * (1 + stop)
                hit = path[path["h"] >= lvl]
                if len(hit):
                    exit_px = max(lvl, hit["o"].iloc[0])
        r = exit_px / entry - 1 if side == "long" else 1 - exit_px / entry
        auction = t.close / t.open - 1 if side == "long" else 1 - t.close / t.open
        rows.append({"date": t.date, "symbol": t.symbol, "ret": r - COST,
                     "daily_bar_ret": auction - COST,
                     "slip_vs_auction%": (entry / t.open - 1) * 100})
    return pd.DataFrame(rows)


def summarise(r, days):
    daily = r.groupby("date")["ret"].mean().reindex(days, fill_value=0.0)
    eq = (1 + daily).cumprod()
    return {
        "trades": len(r),
        "win%": round((r.ret > 0).mean() * 100, 1),
        "avg_trade%": round(r.ret.mean() * 100, 2),
        "median%": round(r.ret.median() * 100, 2),
        "total%": round((eq.iloc[-1] - 1) * 100, 1),
        "sharpe": round(daily.mean() / daily.std() * np.sqrt(252), 2) if daily.std() else 0,
        "maxDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
    }


def main():
    df = bt.features(bt.load())
    fno = bt.fno_symbols()
    start = pd.Timestamp("2026-07-29")
    end = df.date.max() + pd.Timedelta(days=1)
    sigs = signals(df, start, end, fno)

    symbols = sorted({s for t, _ in sigs.values() for s in t.symbol})
    print(f"{len(symbols)} symbols to fetch")
    bars = download(symbols)
    bars["day"] = bars["ts"].dt.normalize()
    bars_by = {k: g.sort_values("ts") for k, g in bars.groupby(["symbol", "day"])}
    days = df.loc[(df.date >= start) & (df.date < end), "date"].drop_duplicates().sort_values()

    pd.set_option("display.width", 220)
    for name, (trades, side) in sigs.items():
        print(f"\n=== {name}  ({start.date()} -> {df.date.max().date()}, cost {COST*1e4:.0f} bps) ===")
        base = reprice(trades, side, bars_by, "09:20")
        cover = len(base) / max(len(trades), 1) * 100
        print(f"signals {len(trades)}, priced with 5m bars {len(base)} ({cover:.0f}%)")
        if base.empty:
            continue
        b = base.assign(ret=base["daily_bar_ret"])
        print("  daily-bar assumption (auction open -> official close):", summarise(b, days))
        for et in ("09:20", "09:25", "09:30"):
            for stop in (None, 0.015, 0.02):
                if stop and et != "09:20":
                    continue
                r = reprice(trades, side, bars_by, et, stop)
                print(f"  entry {et}, exit 15:15, stop={stop}:".ljust(42), summarise(r, days))
        print(f"  median price move from auction open to 09:20 entry: "
              f"{base['slip_vs_auction%'].median():+.2f}% "
              f"({'against you' if (base['slip_vs_auction%'].median() < 0) == (side == 'short') else 'in your favour'})")

    # Combined book from the two halves.
    l = reprice(sigs["Book: 3 gap-down longs"][0], "long", bars_by, "09:20")
    s = reprice(sigs["Book: 3 gap-up shorts"][0], "short", bars_by, "09:20")
    both = pd.concat([l, s])
    print("\n=== Combined book (3 longs + 3 shorts), entry 09:20, exit 15:15 ===")
    print("  daily-bar assumption:", summarise(both.assign(ret=both.daily_bar_ret), days))
    print("  5-minute reality:   ", summarise(both, days))


if __name__ == "__main__":
    main()
