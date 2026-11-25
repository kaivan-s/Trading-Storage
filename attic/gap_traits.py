"""
Gap-Ups in Downtrends: which traits known at 09:20 separate the shorts that
failed from the ones that worked? Reuses gap_5m.py's signals and cached bars.

Traits (all known by the 09:20 entry):
  early_vol   volume of the 09:15 bar / prior 20-day average daily volume
  drift       09:20 open vs the official open (positive = gap still extending)
  ret20       prior 20-session return up to yesterday's close
  depth       yesterday's close vs its 50-day average
  news        medium/high filing after the previous close (gap_news.py)
  mkt_gap     median opening gap of all liquid stocks that day (broad-market open)

Run: ~/anaconda3/bin/python attic/gap_traits.py 2>/dev/null
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gap_5m import BARS  # noqa: E402
from gap_news import news_flags  # noqa: E402
from tab_audit import COST, load  # noqa: E402


def build() -> pd.DataFrame:
    d = load()
    g = d.groupby("symbol", sort=False)
    d["ret20"] = d.A_prev / g.A.shift(21) - 1
    liq = (d.to20 >= 1000) & (d.prev_close >= 50)
    d["mkt_gap"] = d.date.map(d[liq].groupby("date").gap.median())

    flags, have = news_flags(sorted(d.date.unique()))
    m = (d.gap.between(0.02, 0.09, inclusive="left") & ~d.fno & (d.prev_close >= 50)
         & (d.to20 >= 1000) & d.sma50_prev.notna() & (d.A_prev < d.sma50_prev)).fillna(False)
    sig = d.loc[m, ["symbol", "date", "open", "high", "low", "close", "v20",
                    "A_prev", "sma50_prev", "ret20", "mkt_gap", "gap"]].copy()
    sig["news"] = [("yes" if (dt, s) in flags else "no") if dt in have else "unknown"
                   for dt, s in zip(sig.date, sig.symbol)]

    bars = pd.read_parquet(BARS)
    bars["hhmm"] = bars.ts.dt.strftime("%H:%M")
    by = {k: b.sort_values("ts") for k, b in bars.groupby(["symbol", "date"])}
    rows = []
    for t in sig.itertuples():
        b = by.get((t.symbol, t.date))
        if b is None:
            continue
        end, e20, first = b[b.hhmm <= "15:10"], b[b.hhmm >= "09:20"], b[b.hhmm == "09:15"]
        if end.empty or e20.empty or first.empty or end.hhmm.iloc[-1] < "15:00":
            continue
        x, entry = end.c.iloc[-1], e20.o.iloc[0]
        if not (0.8 < x / t.close < 1.25 and 0.85 < entry / t.open < 1.15
                and b.h.max() <= t.high * 1.02 and b.l.min() >= t.low * 0.98):
            continue
        rows.append({
            "date": t.date, "symbol": t.symbol, "news": t.news, "gap": t.gap,
            "ret": 1 - x / entry - COST,
            "early_vol": first.v.iloc[0] / t.v20 if t.v20 else np.nan,
            "drift": entry / t.open - 1,
            "ret20": t.ret20, "depth": t.A_prev / t.sma50_prev - 1, "mkt_gap": t.mkt_gap,
        })
    return pd.DataFrame(rows)


def main() -> None:
    r = build()
    mid = r.date.min() + (r.date.max() - r.date.min()) / 2

    def line(name, s):
        if len(s) < 30:
            print(f"  {name:<30} n={len(s):>5}  (too few)")
            return
        a, b = s[s.date < mid].ret, s[s.date >= mid].ret
        print(f"  {name:<30} n={len(s):>5}  median {100*s.ret.median():+5.2f}%  avg {100*s.ret.mean():+5.2f}%  "
              f"win {100*(s.ret > 0).mean():3.0f}%  blow-ups(<-5%) {100*(s.ret < -0.05).mean():4.1f}%  "
              f"halves {100*a.median():+.2f} / {100*b.median():+.2f}")

    print(f"{len(r)} downtrend gap-ups, {r.date.min().date()} -> {r.date.max().date()}, "
          f"09:20 short to 15:15, after {COST*100:.2f}% cost; positive = short made money\n")
    line("all", r)
    splits = {
        "early_vol (09:15 bar / avg day)": ("early_vol", [0, 0.05, 0.10, 0.20, 0.40, np.inf]),
        "drift (09:20 vs open)": ("drift", [-1, -0.02, -0.01, 0, 0.01, 0.02, 1]),
        "ret20 (prior 20 sessions)": ("ret20", [-1, -0.10, -0.05, 0, 0.05, 0.10, 5]),
        "depth (vs 50-day avg)": ("depth", [-1, -0.20, -0.10, -0.05, -0.03, 0]),
        "mkt_gap (median open gap)": ("mkt_gap", [-1, -0.005, 0, 0.005, 0.01, 1]),
        "gap": ("gap", [0.02, 0.03, 0.04, 0.05, 0.07, 0.09]),
    }
    for title, (col, bins) in splits.items():
        print(f"\n{title}")
        for k, s in r.groupby(pd.cut(r[col], bins), observed=True):
            line(str(k), s)
    print("\nnews")
    for k, s in r.groupby("news"):
        line(k, s)

    print("\ncombined filters (exclude a trade if any listed trait is present)")
    hot = r.early_vol >= 0.20
    ext = r.drift > 0.01
    rec = r.ret20 > 0
    shallow = r.depth > -0.03
    strong_mkt = r.mkt_gap >= 0.005
    for name, bad in [("early_vol>=20%", hot), ("drift>+1%", ext), ("ret20>0", rec),
                      ("depth>-3%", shallow), ("mkt_gap>=+0.5%", strong_mkt),
                      ("early_vol>=20% or drift>+1%", hot | ext),
                      ("early_vol>=20% or ret20>0", hot | rec),
                      ("hot|ext|rec", hot | ext | rec)]:
        line(f"keep: not {name}", r[~bad])
        line(f"  dropped: {name}", r[bad])

    print("\nthe two failure shapes from 25 Nov 2026")
    line("news-led: early_vol>=40% & drift>+2%", r[(r.early_vol >= 0.4) & (r.drift > 0.02)])
    line("early_vol>=40% & drift<=0", r[(r.early_vol >= 0.4) & (r.drift <= 0)])
    line("quiet: early_vol<5% & depth>-3%", r[(r.early_vol < 0.05) & (r.depth > -0.03)])
    line("strong open: mkt_gap>=1%", r[r.mkt_gap >= 0.01])
    line("strong open & quiet stock", r[(r.mkt_gap >= 0.01) & (r.early_vol < 0.10)])
    line("flat open (mkt_gap<0.5%)", r[r.mkt_gap < 0.005])
    line("flat open & early_vol>=20%", r[(r.mkt_gap < 0.005) & (r.early_vol >= 0.2)])

    # The tab publishes the five largest gaps, so the gate is judged on those.
    r = r.sort_values(["date", "gap"], ascending=[True, False])
    top = r[r.groupby("date").cumcount() < 5].copy()
    print("\ntop 5 by gap only, by market opening gap")
    for lo, hi in [(-1, 0), (0, 0.0025), (0.0025, 0.005), (0.005, 0.0075), (0.0075, 1)]:
        line(f"{lo*100:+.2f}%..{hi*100:+.2f}% ({top[(top.mkt_gap >= lo) & (top.mkt_gap < hi)].date.nunique()} days)",
             top[(top.mkt_gap >= lo) & (top.mkt_gap < hi)])
    top["open_kind"] = np.where(top.mkt_gap >= 0.0075, "broad gap-up open", "other opens")
    for kind, s in top.groupby("open_kind"):
        line(f"{kind} ({s.date.nunique()} days)", s)
        for yr, g in s.groupby(s.date.dt.year):
            line(f"  {yr}", g)
    print("\ntop 5 by news")
    for k, s in top.groupby("news"):
        line(k, s)


if __name__ == "__main__":
    main()
