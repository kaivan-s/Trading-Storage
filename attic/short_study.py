"""
Why do stocks fall 5%+ in a day, and can any of it be seen in advance?

Universe: every cached NSE bhavcopy (EQ + BE), price >= 50, 20-day median
turnover >= 5 cr, at least 60 sessions of history, ETFs excluded.

Part A  Causes. Each 5%+ close-to-close fall is put in the first bucket it fits:
        market day (median stock <= -1.5%), sector day (sector median, others
        only, <= -2.5%), news (filing between the previous close and this
        close), after a run-up (+15% in 5 sessions or +8% the day before),
        or none of these. Also splits each fall into gap vs intraday part.
Part B  What was knowable at the previous close. For each feature, the rate of
        next-day 5%+ falls and the next-day open->close move by decile.
Part C  An intraday short ranking. Each feature's deciles are scored on the
        first half of the dates (median next-day open->close vs the typical
        stock); the summed score picks the top 5 per day in the second half.
        Short at the open, cover at the close, 0.25% cost. Then the reverse
        (train on the second half, test on the first).

A cash-market short has to be closed the same day, so only the open->close
part of a fall is capturable; the gap belongs to whoever already held a short.

Run: ~/anaconda3/bin/python attic/short_study.py 2>/dev/null
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from tab_audit import CACHE, COST, load

FILINGS = os.path.join(CACHE, "filings_hist")
TOP = 5
FEATURES = {
    "r1": "today's return",
    "r5": "5-session return",
    "r20": "20-session return",
    "gap": "today's gap",
    "oc": "today's open->close",
    "clv": "close location in today's range (0 = low, 1 = high)",
    "d50": "distance from 50-day average",
    "d200": "distance from 200-day EMA",
    "off_hi": "distance from 250-day high",
    "vr": "today's volume / 20-day average",
    "dlv": "delivery % minus its 20-day average",
    "atr": "20-day average true range, %",
    "upw": "upper wick as share of today's range",
    "streak": "consecutive up days",
}


def features(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("symbol", sort=False)
    d["r1"] = d.ret
    d["r5"] = d.A / g.A.shift(5) - 1
    d["r20"] = d.A / g.A.shift(20) - 1
    rng = (d.high - d.low).replace(0, np.nan)
    d["clv"] = (d.close - d.low) / rng
    d["upw"] = (d.high - d[["open", "close"]].max(axis=1)) / rng
    d["sma50_t"] = g.A.transform(lambda s: s.rolling(50).mean())
    d["d50"] = d.A / d.sma50_t - 1
    d["d200"] = d.A / d.ema200 - 1
    d["off_hi"] = d.A / d.hi250 - 1
    d["vr"] = d.volume / d.v20
    dp = pd.to_numeric(d.deliv_pct, errors="coerce")
    d["dlv"] = dp - dp.groupby(d.symbol).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    tr = pd.concat([d.ah - d.al, (d.ah - g.A.shift(1)).abs(), (d.al - g.A.shift(1)).abs()], axis=1).max(axis=1) / d.A
    d["atr"] = tr.groupby(d.symbol).transform(lambda s: s.rolling(20, min_periods=10).mean())
    d["n_open_ok"] = g.open.shift(-1) > 0
    return d


def news_by_day(days) -> tuple[dict, set]:
    out, have = {}, set()
    for day in days:
        p = os.path.join(FILINGS, f"{day:%Y-%m-%d}.json")
        if os.path.exists(p):
            have.add(day)
            for sym, rows in json.load(open(p)).items():
                out[(day, sym)] = rows[0].get("category") or "Other"
    return out, have


def part_a(d: pd.DataFrame, uni: pd.Series) -> None:
    u = d[uni].copy()
    u["mkt"] = u.groupby("date").ret.transform("median")
    s_sum = u.groupby(["date", "sector"]).ret.transform("sum")
    s_n = u.groupby(["date", "sector"]).ret.transform("count")
    u["sec"] = np.where(s_n >= 6, (s_sum - u.ret) / (s_n - 1), np.nan)
    u["r5_prev"] = u.groupby("symbol").r5.shift(1)
    u["r1_prev"] = u.groupby("symbol").r1.shift(1)
    news, have = news_by_day(sorted(u.date.unique()))
    f = u[(u.ret <= -0.05) & u.date.isin(have)].copy()
    f["news"] = [news.get(k) for k in zip(f.date, f.symbol)]

    def cause(r):
        if r.mkt <= -0.015:
            return "1 market-wide day"
        if r.sec <= -0.025:
            return "2 sector-wide day"
        if r.news:
            return "3 company filing"
        if r.r5_prev >= 0.15 or r.r1_prev >= 0.08:
            return "4 giving back a run-up"
        return "5 no visible reason"
    f["cause"] = f.apply(cause, axis=1)
    days = f.date.nunique()
    print(f"PART A  Why 5%+ falls happen  ({len(f)} falls on {u[u.date.isin(have)].date.nunique()} sessions "
          f"with filings data, {len(f)/max(days,1):.1f} per session that had one)")
    base = (u[u.date.isin(have)].ret <= -0.05).mean()
    print(f"  share of liquid stock-days that fell 5%+: {100*base:.2f}%\n")
    print(f"  {'cause':<26}{'share':>7}{'median fall':>13}{'gap part':>10}{'intraday part':>15}{'fell further after open':>25}")
    for c, g in f.groupby("cause"):
        print(f"  {c:<26}{100*len(g)/len(f):6.0f}%{100*g.ret.median():12.1f}%{100*g.gap.median():9.1f}%"
              f"{100*g.oc.median():14.1f}%{100*(g.oc <= -0.02).mean():22.0f}%")
    print(f"  {'all':<26}{100:6.0f}%{100*f.ret.median():12.1f}%{100*f.gap.median():9.1f}%"
          f"{100*f.oc.median():14.1f}%{100*(f.oc <= -0.02).mean():22.0f}%")
    print("\n  Filings behind company-news falls (top 8):")
    for cat, n in f[f.cause == "3 company filing"].news.value_counts().head(8).items():
        print(f"    {cat:<34}{n:>5}")
    nxt = d.loc[f.index, "n_oc"]
    print(f"\n  The day after a 5%+ fall, open->close median {100*nxt.median():+.2f}% "
          f"(typical stock {100*d.loc[uni, 'n_oc'].median():+.2f}%)\n")


def binned(train: pd.DataFrame, col: str, y: pd.Series):
    edges = np.unique(np.nanquantile(train[col], np.linspace(0, 1, 11)))
    b = pd.cut(train[col], edges, include_lowest=True, labels=False)
    return edges, y.groupby(b).median()


def part_b(d: pd.DataFrame, uni: pd.Series) -> None:
    u = d[uni & d.n_oc.notna() & d.n_open_ok].copy()
    bench = u.groupby("date").n_oc.transform("median")
    u["ex"] = -(u.n_oc - bench)
    base_f, base_oc = (u.n_ret <= -0.05).mean(), (u.n_oc <= -0.04).mean()
    print(f"PART B  What the previous close showed  (n={len(u):,} stock-days)")
    print(f"  base rates next day: fell 5%+ close-to-close {100*base_f:.2f}%, fell 4%+ after the open {100*base_oc:.2f}%")
    print("  per feature: decile 1 (lowest) .. 10 (highest). 'x' = how many times the base rate of a 4%+ fall")
    print("  after the open; 'short' = median open->close short vs the typical stock that day\n")
    for col, desc in FEATURES.items():
        x = u[[col, "n_oc", "ex"]].dropna()
        if x.empty:
            continue
        q = pd.qcut(x[col], 10, labels=False, duplicates="drop")
        grp = x.groupby(q)
        lift = grp.n_oc.apply(lambda s: (s <= -0.04).mean()) / base_oc
        sh = grp.ex.median() * 100
        print(f"  {col:<7}{desc}")
        print("    x     " + " ".join(f"{v:5.1f}" for v in lift))
        print("    short " + " ".join(f"{v:+5.2f}" for v in sh))
    print()


def score_model(train: pd.DataFrame):
    models = {}
    for col in FEATURES:
        x = train[[col, "ex"]].dropna()
        if len(x) < 1000:
            continue
        models[col] = binned(x, col, x.ex)
    return models


def apply_score(test: pd.DataFrame, models) -> pd.Series:
    s = pd.Series(0.0, index=test.index)
    for col, (edges, med) in models.items():
        b = pd.cut(test[col].clip(edges[0], edges[-1]), edges, include_lowest=True, labels=False)
        s += b.map(med).fillna(0).astype(float)
    return s


def part_c(d: pd.DataFrame, uni: pd.Series) -> pd.DataFrame:
    u = d[uni & d.n_oc.notna() & d.n_open_ok].copy()
    u["bench"] = u.groupby("date").n_oc.transform("median")
    u["ex"] = -(u.n_oc - u.bench)
    dates = np.array(sorted(u.date.unique()))
    mid = dates[len(dates) // 2]
    print(f"PART C  Intraday short ranking, top {TOP} per day, short at next open, cover at next close, "
          f"{COST*100:.2f}% cost\n")
    picks = []
    for name, tr, te in [("train 1st half, test 2nd", u.date < mid, u.date >= mid),
                         ("train 2nd half, test 1st", u.date >= mid, u.date < mid)]:
        models = score_model(u[tr])
        t = u[te].copy()
        t["score"] = apply_score(t, models)
        top = t.sort_values("score", ascending=False).groupby("date").head(TOP)
        sh = -top.n_oc - COST
        ex = top.ex
        rnd = (-t.n_oc - COST)
        daily = sh.groupby(top.date).mean()
        print(f"  {name}: {len(top)} trades, {top.date.min().date()} -> {top.date.max().date()}")
        print(f"    short won {100*(sh > 0).mean():.0f}%  median {100*sh.median():+.2f}%  avg {100*sh.mean():+.2f}%  "
              f"vs typical stock {100*ex.median():+.2f}%  |  any stock shorted: median {100*rnd.median():+.2f}%")
        print(f"    days the 5-stock book made money: {100*(daily > 0).mean():.0f}%, "
              f"avg day {100*daily.mean():+.2f}%, fell 4%+ after open {100*(top.n_oc <= -0.04).mean():.1f}% "
              f"(base {100*(t.n_oc <= -0.04).mean():.1f}%)")
        dec = pd.qcut(t.score, 10, labels=False, duplicates="drop")
        print("    score decile (1 = lowest): short median vs typical stock")
        print("      " + " ".join(f"{100*v:+5.2f}" for v in t.ex.groupby(dec).median()))
        top = top.assign(split=name)
        picks.append(top)
        # which features carry the score
        w = {c: (m[1].max() - m[1].min()) * 100 for c, m in models.items()}
        print("    biggest feature spreads: " + ", ".join(f"{c} {v:.2f}" for c, v in sorted(w.items(), key=lambda kv: -kv[1])[:6]))
        print()
    out = pd.concat(picks)
    out[["date", "symbol", "score", "open", "close", "n_oc", "split"]].to_parquet(
        os.path.join(CACHE, "short_study_picks.parquet"))
    return out


def main() -> None:
    d = features(load())
    uni = ((d.close >= 50) & (d.to20 >= 500) & (d.n_hist >= 60)).fillna(False)
    print(f"{d.date.min().date()} -> {d.date.max().date()}, {d[uni].symbol.nunique()} liquid symbols\n")
    part_a(d, uni)
    part_b(d, uni)
    part_c(d, uni)


if __name__ == "__main__":
    main()
