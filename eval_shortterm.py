#!/usr/bin/env python3
"""
Short-term horizon study: which names reach +4% intraday, and can we see it
coming the evening before?

EVENT  day D's HIGH >= 4% above day D-1's CLOSE (NSE's PREV_CLOSE, which is
       already adjusted on ex-dates, so splits cannot fake an event).

UNIVERSE  EQ series, close >= Rs 20, trailing 20-session median turnover
          >= Rs 1 crore (100 lakh), at least 60 sessions of history.

PART 1  Describe every day in the review window: how many names hit, whether
        the move was a gap or intraday, whether it held into the close, which
        sectors carried it, and the biggest movers.

PART 2  Every feature is measured at D-1's close -- known the evening before
        -- and tested against the D event over the whole cache:
          * raw lift       hit rate of the top quintile vs the day's base rate
          * ATR-controlled lift within the same volatility quintile. Reaching
                           4% is mostly a volatility question, so a feature
                           that only re-discovers "this stock is volatile"
                           adds nothing an ATR sort does not already give.
          * both halves    the effect must hold in each half of the sample
        t-stats are on the daily series (one observation per session), so
        2,000 names on the same day are not counted as 2,000 samples.

PART 3  Combine the features that survive into a score, fitted on the first
        60% of sessions only, and judge the top-N per day on the last 40%:
        hit rate, hit rate from the OPEN (a gap-up above 4% cannot be bought
        below 4%), and a simple buy-at-open / sell-at-+4%-or-close trade.

Usage:
    python eval_shortterm.py
    python eval_shortterm.py --start 2026-09-21 --end 2026-09-28 --top 40
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date

import numpy as np
import pandas as pd

import fetch

EVENT = 0.04
MIN_PRICE = 20.0
MIN_TURN_LACS = 100.0
MIN_HIST = 60
TRAIN_FRAC = 0.60


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load(end: date) -> pd.DataFrame:
    files = sorted(fetch.RAW_DIR.glob("bhav_*.*"))
    files = [p for p in files if p.name[5:13] <= f"{end:%Y%m%d}"]
    raw = pd.concat([fetch._read_cache(p) for p in files], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = fetch.drop_repeat_sessions(raw, verbose=False)
    df = raw[raw["series"].str.upper() == "EQ"].copy()
    df = df[(df["close"] > 0) & (df["prev_close"] > 0) & (df["volume"] > 0)]
    # ETFs trade under series EQ too; EQUITY_L lists companies only.
    eq_list = fetch.CACHE_DIR / "equity_list.csv"
    if eq_list.exists():
        df = df[df["symbol"].isin(set(pd.read_csv(eq_list)["symbol"].astype(str).str.strip()))]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    smap_path = fetch.CACHE_DIR / "sector_map.json"
    if smap_path.exists():
        smap = json.loads(smap_path.read_text())
        df["sector"] = df["symbol"].map(
            lambda s: (smap.get(s) or {}).get("basic_industry") or "Unknown")
    else:
        df["sector"] = "Unknown"
    return df


def rsi(ret: pd.Series, n: int = 14) -> pd.Series:
    up = ret.clip(lower=0)
    dn = (-ret).clip(lower=0)
    a = 1.0 / n
    ru = up.ewm(alpha=a, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=a, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything here is known at the close of the row's own date."""
    df = df.copy()
    g = df.groupby("symbol", sort=False)
    pc = df["prev_close"]

    ret = df["close"] / pc - 1.0
    df["ret1"] = ret.where(ret.abs() < 0.6)
    df["tri"] = g["ret1"].transform(lambda s: (1 + s.fillna(0)).cumprod())
    df["n_hist"] = g.cumcount() + 1

    df["reach"] = df["high"] / pc - 1.0
    df["gap"] = df["open"] / pc - 1.0
    df["reach_open"] = df["high"] / df["open"] - 1.0
    df["oc"] = df["close"] / df["open"] - 1.0
    df["hit"] = (df["reach"] >= EVENT).astype(float)

    tr = np.maximum(df["high"], pc) - np.minimum(df["low"], pc)
    df["rng1"] = tr / pc
    g = df.groupby("symbol", sort=False)
    df["atr14"] = g["rng1"].transform(lambda s: s.rolling(14, min_periods=10).mean())
    df["rng_vs_atr"] = df["rng1"] / df["atr14"]
    df["nr7"] = (df["rng1"] <= g["rng1"].transform(
        lambda s: s.rolling(7, min_periods=7).min())).astype(float)

    df["ret5"] = g["tri"].transform(lambda s: s / s.shift(5) - 1)
    df["ret20"] = g["tri"].transform(lambda s: s / s.shift(20) - 1)
    df["from_hi20"] = df["tri"] / g["tri"].transform(
        lambda s: s.rolling(20, min_periods=15).max()) - 1
    df["from_hi250"] = df["tri"] / g["tri"].transform(
        lambda s: s.rolling(250, min_periods=120).max()) - 1
    df["rsi14"] = g["ret1"].transform(lambda s: rsi(s.fillna(0)))

    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["clv"] = ((df["close"] - df["low"]) / hl).fillna(0.5)
    df["close_at_high"] = ((df["close"] >= df["high"] * 0.999)
                           & (df["ret1"] >= 0.045)).astype(float)
    # Closed on a price band (5/10/20%): an upper-circuit close has no sellers,
    # so a buy order at that close usually does not fill.
    near_band = sum((df["ret1"] - b).abs() < 0.003 for b in (0.05, 0.10, 0.20)) > 0
    df["at_band"] = (df["close_at_high"].astype(bool) & near_band).astype(float)
    df["locked"] = (df["high"] == df["low"]).astype(float)

    df["vol_ratio"] = df["volume"] / g["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    df["dq"] = df["deliv_pct"] / g["deliv_pct"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=15).mean())
    df["med_turn20"] = g["turnover"].transform(
        lambda s: s.rolling(20, min_periods=15).median())
    df["log_turn"] = np.log10(df["med_turn20"])
    df["log_price"] = np.log10(df["close"])
    df["hits10"] = g["hit"].transform(lambda s: s.rolling(10, min_periods=5).sum())

    adj = df["tri"]
    ema50 = g["tri"].transform(lambda s: s.ewm(span=50, adjust=False).mean())
    ema200 = g["tri"].transform(lambda s: s.ewm(span=200, adjust=False).mean())
    df["uptrend"] = ((adj > ema50) & (ema50 > ema200)).astype(float)

    df["uni"] = ((df["close"] >= MIN_PRICE)
                 & (df["med_turn20"] >= MIN_TURN_LACS)
                 & (df["n_hist"] >= MIN_HIST))

    # Cross-sectional context, computed on the universe only.
    u = df[df["uni"]]
    sec_rate = u.groupby(["date", "sector"])["hit"].mean().rename("sector_hit_rate")
    mkt = u.groupby("date").agg(mkt_adv=("ret1", lambda s: (s > 0).mean()),
                                mkt_hit_rate=("hit", "mean"))
    df = df.join(sec_rate, on=["date", "sector"]).join(mkt, on="date")
    df.loc[df["sector"] == "Unknown", "sector_hit_rate"] = np.nan

    # Next-session outcomes, aligned onto the feature row.
    g = df.groupby("symbol", sort=False)
    for c in ("hit", "reach", "gap", "reach_open", "oc", "ret1", "locked", "date"):
        df[f"nx_{c}"] = g[c].shift(-1)
    # A symbol that skipped sessions (suspension) must not pair with a later one.
    dates = np.sort(df["date"].unique())
    nxt = pd.Series(dates[1:], index=dates[:-1])
    df = df[df["nx_date"].notna() & (df["nx_date"] == df["date"].map(nxt))]
    return df


FEATURES = [
    "atr14", "rng1", "rng_vs_atr", "nr7", "ret1", "ret5", "ret20",
    "from_hi20", "from_hi250", "rsi14", "clv", "close_at_high",
    "vol_ratio", "dq", "log_turn", "log_price", "hits10",
    "uptrend", "sector_hit_rate", "mkt_adv", "mkt_hit_rate",
]
BINARY = {"nr7", "close_at_high", "uptrend"}


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------

def tstat(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 5 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def pval(t: float) -> float:
    if not np.isfinite(t):
        return np.nan
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def daily_quintile(s: pd.DataFrame, col: str) -> pd.Series:
    """Per-day quintile 1..5. Binary features map to 1 / 5."""
    if col in BINARY:
        return s[col].map({0.0: 1, 1.0: 5})
    return s.groupby("date")[col].transform(
        lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) + 1
        if v.notna().sum() >= 25 else pd.Series(np.nan, index=v.index))


# --------------------------------------------------------------------------
# Part 1
# --------------------------------------------------------------------------

def describe_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Rows are D-1 features with nx_* = day D. Report by day D."""
    w = df[df["uni"] & (df["nx_date"] >= start) & (df["nx_date"] <= end)].copy()
    print("\n" + "=" * 96)
    print(f"PART 1 — daily 4% movers, {start.date()} to {end.date()}")
    print(f"event: high >= prev close +{EVENT:.0%}; universe: EQ, close >= Rs {MIN_PRICE:.0f}, "
          f"20d median turnover >= Rs {MIN_TURN_LACS / 100:.0f} cr")
    print("=" * 96)
    print(f"{'day':<12}{'univ':>6}{'hits':>6}{'rate':>7}{'gap>=4%':>9}"
          f"{'held>=4%':>10}{'closed red':>12}{'med reach':>11}{'med close':>11}{'mkt adv':>9}")
    for d, grp in w.groupby("nx_date"):
        h = grp[grp["nx_hit"] == 1]
        print(f"{pd.Timestamp(d).date()!s:<12}{len(grp):>6}{len(h):>6}"
              f"{len(h) / len(grp):>7.1%}{(h['nx_gap'] >= EVENT).mean():>9.0%}"
              f"{(h['nx_ret1'] >= EVENT).mean():>10.0%}{(h['nx_ret1'] < 0).mean():>12.0%}"
              f"{h['nx_reach'].median():>11.1%}{h['nx_ret1'].median():>11.1%}"
              f"{(grp['nx_ret1'] > 0).mean():>9.0%}")

    hits = w[w["nx_hit"] == 1]
    print(f"\nTotal: {len(hits)} events from {hits['symbol'].nunique()} distinct names "
          f"over {w['nx_date'].nunique()} sessions.")
    rep = hits.groupby("symbol").size()
    print(f"Repeat movers (hit on 2+ days): {(rep >= 2).sum()} names — "
          + ", ".join(f"{s}×{n}" for s, n in rep[rep >= 2].sort_values(ascending=False)
                      .head(15).items()))

    sec = (hits.groupby("sector").size().rename("hits").to_frame()
           .join(w.groupby("sector").size().rename("names_days")))
    sec["rate"] = sec["hits"] / sec["names_days"]
    sec = sec[sec["hits"] >= 4].sort_values("rate", ascending=False)
    print("\nSectors with the most concentrated moves (>= 4 events):")
    print(f"  {'sector':<42}{'hits':>6}{'name-days':>11}{'rate':>8}")
    for s, r in sec.head(12).iterrows():
        print(f"  {str(s)[:40]:<42}{int(r['hits']):>6}{int(r['names_days']):>11}{r['rate']:>8.0%}")

    print("\nBiggest reach per day (top 8):")
    for d, grp in hits.groupby("nx_date"):
        top = grp.nlargest(8, "nx_reach")
        print(f"  {pd.Timestamp(d).date()}: " + ", ".join(
            f"{r.symbol} +{r.nx_reach:.1%}/{r.nx_ret1:+.1%}" for r in top.itertuples()))
    print("  (format: reach from prev close / where it closed)")

    print("\nDay-before profile, movers vs everyone else in this window (medians):")
    cols = ["atr14", "rng1", "ret1", "ret5", "ret20", "from_hi20", "rsi14",
            "vol_ratio", "dq", "clv", "hits10", "log_turn"]
    a, b = hits[cols].median(), w[w["nx_hit"] == 0][cols].median()
    print(f"  {'feature':<12}{'movers':>10}{'others':>10}")
    for c in cols:
        print(f"  {c:<12}{a[c]:>10.3f}{b[c]:>10.3f}")


# --------------------------------------------------------------------------
# Part 2
# --------------------------------------------------------------------------

def feature_tests(u: pd.DataFrame) -> pd.DataFrame:
    u = u.copy()
    u["atr_q"] = daily_quintile(u, "atr14")
    cell = u.groupby(["date", "atr_q"])["nx_hit"].transform("mean")
    u["resid"] = u["nx_hit"] - cell
    base_day = u.groupby("date")["nx_hit"].transform("mean")
    u["raw_resid"] = u["nx_hit"] - base_day

    dates = np.sort(u["date"].unique())
    half = dates[len(dates) // 2]
    out = []
    for f in FEATURES:
        if f in ("mkt_adv", "mkt_hit_rate"):
            continue
        q = daily_quintile(u, f)
        rates = u.groupby(q)["nx_hit"].mean()
        top, bot = (q == 5), (q == 1)
        raw = u[top].groupby("date")["raw_resid"].mean()
        ctl = u[top].groupby("date")["resid"].mean()
        ctl_bot = u[bot].groupby("date")["resid"].mean()
        h1 = ctl[ctl.index < half].mean()
        h2 = ctl[ctl.index >= half].mean()
        t = tstat(ctl)
        out.append({
            "feature": f,
            **{f"Q{i}": rates.get(i, np.nan) for i in range(1, 6)},
            "raw_lift": raw.mean(),
            "ctl_top": ctl.mean(), "ctl_bot": ctl_bot.mean(),
            "t_ctl": t, "h1": h1, "h2": h2,
        })
    return pd.DataFrame(out)


def market_context(u: pd.DataFrame) -> None:
    d = u.groupby("date").agg(mkt_adv=("mkt_adv", "first"),
                              mkt_hit=("mkt_hit_rate", "first"),
                              nx=("nx_hit", "mean"))
    print("\nMarket regime: does today's tape predict tomorrow's hit rate?")
    print(f"  corr(today's hit rate, tomorrow's) = {d['mkt_hit'].corr(d['nx']):+.2f}")
    print(f"  corr(today's % advancing, tomorrow's hit rate) = {d['mkt_adv'].corr(d['nx']):+.2f}")
    d["b"] = pd.qcut(d["mkt_hit"], 3, labels=["quiet", "normal", "hot"])
    print("  tomorrow's hit rate by today's tape: " + ", ".join(
        f"{k} {v:.1%}" for k, v in d.groupby("b", observed=True)["nx"].mean().items()))


# --------------------------------------------------------------------------
# Part 3
# --------------------------------------------------------------------------

def rank_score(day: pd.DataFrame, weights: dict) -> pd.Series:
    s = pd.Series(0.0, index=day.index)
    for f, w in weights.items():
        s += w * day[f].rank(pct=True).fillna(0.5)
    return s


def trade_open(r: pd.DataFrame) -> np.ndarray:
    """Buy at D's open; sell at +4% if the high reaches it, else at D's close."""
    return np.where(r["nx_reach_open"] >= EVENT, EVENT, r["nx_oc"])


def trade_btst(r: pd.DataFrame) -> np.ndarray:
    """Buy at D-1's close; sell at D's open if it gaps past +4%, at +4% if
    touched intraday, else at D's close. Returns are vs D-1's close."""
    return np.where(r["nx_gap"] >= EVENT, r["nx_gap"],
                    np.where(r["nx_reach"] >= EVENT, EVENT, r["nx_ret1"]))


def evaluate(sample: pd.DataFrame, legs: dict, title: str) -> None:
    """One row per list. Every mean is per session first, then across sessions."""
    dates = np.sort(sample["date"].unique())
    half = dates[len(dates) // 2]
    s = sample.assign(t_open=trade_open(sample), t_btst=trade_btst(sample))
    uni = s.groupby("date")[["t_open", "t_btst"]].mean()
    print(f"\n{title}: {len(dates)} sessions "
          f"({pd.Timestamp(dates[0]).date()} to {pd.Timestamp(s['nx_date'].max()).date()})")
    print(f"  {'list':<34}{'n/day':>6}{'hit%':>7}{'open%':>7}{'gap4%':>7}{'band%':>7}"
          f"{'c-c':>7}{'open tr':>8}{'btst':>7}{'t btst':>8}{'h1':>7}{'h2':>7}")
    for name, pick in legs.items():
        parts = [pick(d) for _, d in s.groupby("date")]
        rows = pd.concat([x for x in parts if len(x)])
        per = rows.groupby("date").agg(
            n=("symbol", "size"), hit=("nx_hit", "mean"),
            ho=("nx_reach_open", lambda v: (v >= EVENT).mean()),
            gap=("nx_gap", lambda v: (v >= EVENT).mean()),
            band=("at_band", "mean"), cc=("nx_ret1", "mean"),
            to=("t_open", "mean"), tb=("t_btst", "mean"))
        ex = per["tb"] - uni["t_btst"].reindex(per.index)
        t = tstat(ex)
        h1 = per.loc[per.index < half, "tb"].mean()
        h2 = per.loc[per.index >= half, "tb"].mean()
        ts = f"{t:>8.1f}" if np.isfinite(t) else f"{'—':>8}"
        print(f"  {name:<34}{per['n'].mean():>6.0f}{per['hit'].mean():>7.1%}"
              f"{per['ho'].mean():>7.1%}{per['gap'].mean():>7.1%}{per['band'].mean():>7.1%}"
              f"{per['cc'].mean():>7.2%}{per['to'].mean():>8.2%}{per['tb'].mean():>7.2%}"
              f"{ts}{h1:>7.2%}{h2:>7.2%}")


LEGEND = """  hit% = D high >= D-1 close +4%   open% = D high >= D open +4% (capturable by a buyer at the open)
  gap4% = opened >= +4% already      band% = closed D-1 on a 5/10/20% price band (likely unfillable)
  c-c = D close vs D-1 close         open tr = buy D open, sell +4% or close
  btst = buy D-1 close, sell gap / +4% / close    t btst = btst minus universe btst, daily t-stat
  h1/h2 = btst in each half. All returns gross; costs are ~0.1% intraday, ~0.25% delivery."""


def setups(top: int, weights: dict | None = None) -> dict:
    legs = {
        "universe (no selection)": lambda d: d,
        f"ATR only, top {top}": lambda d: d.nlargest(top, "atr14"),
        "strong close (at high, +4.5%)": lambda d: d[d["close_at_high"] == 1],
        "  ...on a price band": lambda d: d[d["at_band"] == 1],
        "  ...off the band (fillable)": lambda d: d[(d["close_at_high"] == 1) & (d["at_band"] == 0)],
        "volume breakout (2x vol, +3%, clv>.8)": lambda d: d[
            (d["vol_ratio"] >= 2) & (d["ret1"] >= 0.03) & (d["clv"] >= 0.8)],
        "  ...and not on a band": lambda d: d[
            (d["vol_ratio"] >= 2) & (d["ret1"] >= 0.03) & (d["clv"] >= 0.8) & (d["at_band"] == 0)],
        "repeat mover (3+ hits in 10d)": lambda d: d[d["hits10"] >= 3],
        "sector wave (sector >=30% hit)": lambda d: d[(d["sector_hit_rate"] >= 0.3)],
    }
    if weights:
        legs[f"combined score, top {top}"] = lambda d: d.loc[rank_score(d, weights).nlargest(top).index]
        legs[f"combined, off band, top {top}"] = lambda d: (
            lambda x: x.loc[rank_score(x, weights).nlargest(top).index])(d[d["at_band"] == 0])
    return legs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-09-21")
    ap.add_argument("--end", default="2026-09-28")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--dump", default=None, help="write the window's events to CSV")
    a = ap.parse_args()
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)

    print("Loading cached bhavcopy…")
    df = features(load(end.date()))
    describe_window(df, start, end)

    if a.dump:
        w = df[df["uni"] & (df["nx_date"] >= start) & (df["nx_date"] <= end)
               & (df["nx_hit"] == 1)]
        w.to_csv(a.dump, index=False)
        print(f"\nwrote {len(w)} events to {a.dump}")

    u = df[df["uni"] & df["atr14"].notna()].copy()
    print("\n" + "=" * 96)
    print(f"PART 2 — what predicts a 4% reach the next day? {u['date'].nunique()} sessions, "
          f"{len(u):,} name-days, base rate {u['nx_hit'].mean():.1%}")
    print("=" * 96)
    res = feature_tests(u)
    print("Q1..Q5 = next-day hit rate by feature quintile (binary: Q1=no, Q5=yes).")
    print("raw = top-quintile lift over the day's base rate; ctl = same, but vs names in the")
    print("same ATR quintile (what the feature adds beyond volatility); h1/h2 = ctl in each half.")
    print(f"{'feature':<16}{'Q1':>6}{'Q2':>6}{'Q3':>6}{'Q4':>6}{'Q5':>6}"
          f"{'raw':>8}{'ctl':>8}{'ctl Q1':>8}{'t':>7}{'h1':>7}{'h2':>7}")
    for r in res.sort_values("t_ctl", key=lambda s: -s.abs()).itertuples():
        print(f"{r.feature:<16}" + "".join(
            f"{getattr(r, f'Q{i}'):>6.1%}" if np.isfinite(getattr(r, f'Q{i}')) else f"{'—':>6}"
            for i in range(1, 6))
            + f"{r.raw_lift * 100:>+7.1f}p{r.ctl_top * 100:>+7.1f}p{r.ctl_bot * 100:>+7.1f}p"
            f"{r.t_ctl:>7.1f}{r.h1 * 100:>+6.1f}p{r.h2 * 100:>+6.1f}p")
    market_context(u)

    # Weights: features whose ATR-controlled top-vs-bottom spread is significant
    # and agrees in sign across both halves, fitted on the TRAIN window only.
    dates = np.sort(u["date"].unique())
    train = u[u["date"] < dates[int(len(dates) * TRAIN_FRAC)]]
    tr = feature_tests(train).set_index("feature")
    weights = {"atr14": 1.0}
    for f, r in tr.iterrows():
        if f == "atr14":
            continue
        spread = r["ctl_top"] - r["ctl_bot"]
        if abs(r["t_ctl"]) >= 3 and np.sign(r["h1"]) == np.sign(r["h2"]) == np.sign(r["ctl_top"]):
            weights[f] = float(np.sign(spread)) * 0.5
    print("\n" + "=" * 96)
    print("PART 3 — tradeable setups; the combined score is fitted on the first 60% only")
    print("=" * 96)
    print("weights (rank-percentile sum): " + ", ".join(f"{k} {v:+.1f}" for k, v in weights.items()))
    print(LEGEND)
    evaluate(u, setups(a.top), "Candidate setups, full sample")
    test = u[u["date"] >= dates[int(len(dates) * TRAIN_FRAC)]]
    evaluate(test, setups(a.top, weights), "Out-of-sample (last 40%), incl. combined score")


if __name__ == "__main__":
    main()
