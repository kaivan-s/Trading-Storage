"""
Gap-Ups on 5-minute bars: downtrend vs uptrend vs all, and news vs no news.

Signals: the tab_audit.py gap rule (open 2-9% above the previous close, non-F&O,
price >= 50, 20-day median turnover >= 10 cr), split by the previous close vs
its 50-day average. News = a medium/high-importance filing after the previous
close and before 09:15 (gap_news.py); sessions after the filings cache ends
are "news unknown" and left out of the news split only.

Trade: short at the open of the 09:20 bar (also 09:25), cover at the close of
the 15:10 bar (~15:15 MIS square-off), 0.25% round-trip cost. "Auction" is the
same trade from NSE's official open to the official close, for comparison.

Bars come from Groww, one request per signal, cached in
data/cache/gap_5m_bars.parquet so reruns only fetch what's new. Groww serves
some corrupt days (e.g. 12 May 2025, prices ~100x), so a day is dropped unless
its bars agree with the official bhavcopy high, low, open and close.

Run: ~/anaconda3/bin/python attic/gap_5m.py 2>/dev/null
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fetch  # noqa: E402
from gap_news import news_flags  # noqa: E402
from tab_audit import CACHE, COST, load  # noqa: E402

BARS = os.path.join(CACHE, "gap_5m_bars.parquet")
WORKERS = 4


def fetch_day(client, sym: str, day: pd.Timestamp) -> pd.DataFrame | None:
    for attempt in range(3):
        try:
            c = client.get_historical_candles(
                exchange="NSE", segment="CASH", groww_symbol=f"NSE-{sym}",
                start_time=f"{day:%Y-%m-%d} 09:15:00", end_time=f"{day:%Y-%m-%d} 15:30:00",
                candle_interval="5minute")
            rows = (c.get("candles") if isinstance(c, dict) else c) or []
            if not rows:
                return None
            b = pd.DataFrame([r[:6] for r in rows], columns=["ts", "o", "h", "l", "c", "v"])
            b["ts"] = pd.to_datetime(b.ts)
            b["symbol"], b["date"] = sym, day
            return b
        except Exception as exc:
            if "not found" in str(exc).lower() or "invalid" in str(exc).lower():
                return None
            time.sleep(1 + 2 * attempt)
    return None


def get_bars(sig: pd.DataFrame) -> pd.DataFrame:
    have = pd.read_parquet(BARS) if os.path.exists(BARS) else pd.DataFrame(columns=["symbol", "date"])
    done = set(zip(have.symbol, pd.to_datetime(have.date)))
    need = [(s, d) for s, d in zip(sig.symbol, sig.date) if (s, d) not in done]
    print(f"{len(sig)} signals, {len(need)} to fetch")
    client = fetch._get_groww_client()
    frames = [have] if len(have) else []
    for i in range(0, len(need), 400):
        chunk = need[i:i + 400]
        with ThreadPoolExecutor(WORKERS) as ex:
            got = [b for b in ex.map(lambda k: fetch_day(client, *k), chunk) if b is not None]
        frames += got
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(BARS)
        frames = [out]
        print(f"  fetched {min(i + 400, len(need))}/{len(need)}", flush=True)
    return frames[0] if frames else have


def price(sig: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.assign(hhmm=bars.ts.dt.strftime("%H:%M"))
    by = {k: g.sort_values("ts") for k, g in bars.groupby(["symbol", "date"])}
    rows = []
    for t in sig.itertuples():
        b = by.get((t.symbol, t.date))
        if b is None:
            continue
        end = b[b.hhmm <= "15:10"]
        e20, e25 = b[b.hhmm >= "09:20"], b[b.hhmm >= "09:25"]
        if end.empty or e20.empty or e25.empty or end.hhmm.iloc[-1] < "15:00":
            continue
        x = end.c.iloc[-1]
        day_hi, day_lo = b.h.max(), b.l.min()
        if not (0.8 < x / t.close < 1.25 and 0.85 < e20.o.iloc[0] / t.open < 1.15
                and day_hi <= t.high * 1.02 and day_lo >= t.low * 0.98):
            continue
        rows.append({
            "date": t.date, "symbol": t.symbol, "trend": t.trend, "news": t.news,
            "auction": 1 - t.close / t.open - COST,
            "e0920": 1 - x / e20.o.iloc[0] - COST,
            "e0925": 1 - x / e25.o.iloc[0] - COST,
            "drift_0920": e20.o.iloc[0] / t.open - 1,
        })
    return pd.DataFrame(rows)


def show(name: str, r: pd.DataFrame) -> None:
    if len(r) < 20:
        print(f"  {name:<22} n={len(r):>5}  (too few)")
        return
    mid = r.date.min() + (r.date.max() - r.date.min()) / 2
    a, b = r[r.date < mid].e0920, r[r.date >= mid].e0920
    print(f"  {name:<22} n={len(r):>5}  "
          f"auction med {100*r.auction.median():+5.2f}%  |  09:20 med {100*r.e0920.median():+5.2f}% "
          f"avg {100*r.e0920.mean():+5.2f}% win {100*(r.e0920 > 0).mean():3.0f}%  |  "
          f"09:25 med {100*r.e0925.median():+5.2f}%  |  09:20 halves {100*a.median():+.2f}% / {100*b.median():+.2f}%")


def main() -> None:
    d = load()
    flags, have = news_flags(sorted(d.date.unique()))
    m = (d.gap.between(0.02, 0.09, inclusive="left") & ~d.fno & (d.prev_close >= 50)
         & (d.to20 >= 1000) & d.oc.notna() & d.sma50_prev.notna()).fillna(False)
    sig = d.loc[m, ["symbol", "date", "open", "high", "low", "close", "A_prev", "sma50_prev"]].copy()
    sig["trend"] = np.where(sig.A_prev < sig.sma50_prev, "down", "up")
    sig["news"] = [("yes" if (dt, s) in flags else "no") if dt in have else "unknown"
                   for dt, s in zip(sig.date, sig.symbol)]

    bars = get_bars(sig)
    r = price(sig, bars)
    print(f"\npriced {len(r)} of {len(sig)} signals, {r.date.min().date()} -> {r.date.max().date()}, "
          f"after {COST*100:.2f}% cost; positive = the short made money")
    print(f"median move from official open to 09:20 entry: {100*r.drift_0920.median():+.2f}% "
          f"(negative = price already fell before you could enter)\n")
    for trend, g in [("Downtrend (the tab)", r[r.trend == "down"]), ("Uptrend", r[r.trend == "up"]), ("All gap-ups", r)]:
        print(trend)
        show("all", g)
        show("no news", g[g.news == "no"])
        show("news before open", g[g.news == "yes"])
        print()


if __name__ == "__main__":
    main()
