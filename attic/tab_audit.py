"""
One test for every tab on the Setups page, on the same data and benchmark.

Data: every cached NSE bhavcopy (EQ + BE). Prices are chained through NSE's
own prev_close, which is adjusted on ex-dates, so splits and bonuses don't
create fake returns.

Benchmark: the same day's median liquid stock on the same measurement, so a
"+0.3%" is +0.3% better than the typical stock, not better than zero.
Medians are the headline: means of right-skewed returns flatter every list.

Each tab is tested on the horizon it is used for:
  Coils                  20 sessions (and 5) from the close
  Gap-Ups in Downtrends  same day, open -> close, as a fade (short)
  Circuit Carry          buy the close, sell next session at +4% / open gap / close
  Big Movers, Unusual Vol, 52W Breakouts, Streaks
                         next open, next day close->close, 5 and 20 sessions
  Sector Pulse           do today's top-3 sectors beat the market tomorrow / 5d?

The live tabs run intraday in the app; here they use the end-of-day version
of the same rule. Multi-day horizons use a 10-session cooldown per symbol so a
name sitting in a list for weeks isn't counted every day.

Run: ~/anaconda3/bin/python attic/tab_audit.py 2>/dev/null
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "data", "cache")
ETF = r"ETF|BEES|^SETF|GOLD|SILVER|LIQUID|CASE$|NIFTY|SENSEX|IETF|MODEFENCE"
COST = 0.0025
COOLDOWN = 10


def load() -> pd.DataFrame:
    cols = ["symbol", "series", "date", "prev_close", "open", "high", "low", "close", "volume", "turnover", "deliv_pct"]
    fs = sorted(glob.glob(os.path.join(HERE, "..", "data", "raw", "bhav_*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in fs])
    d = d[d.series.isin(["EQ", "BE"]) & ~d.symbol.str.contains(ETF)]
    d = d[(d.close > 0) & (d.prev_close > 0) & (d.open > 0)]
    d["date"] = pd.to_datetime(d.date)
    d = d.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"]).reset_index(drop=True)
    g = d.groupby("symbol", sort=False)

    d["ret"] = d.close / d.prev_close - 1
    d["gap"] = d.open / d.prev_close - 1
    d["oc"] = d.close / d.open - 1
    d["A"] = g.ret.transform(lambda s: (1 + s).cumprod())          # split-safe close index
    k = d.A / d.close
    d["ah"], d["al"] = d.high * k, d.low * k

    sh = lambda col, n: g[col].shift(n)
    roll = lambda col, n, fn, mp=None, s=1: g[col].transform(
        lambda x: getattr(x.shift(s).rolling(n, min_periods=mp or n), fn)())
    d["n_hist"] = g.cumcount() + 1
    d["to20"] = roll("turnover", 20, "median", 10)                  # lakhs, prior 20 sessions
    d["v20"] = roll("volume", 20, "mean", 10)
    d["sma50"] = roll("close", 50, "mean", 50, s=0).shift(0)
    d["sma50_prev"] = g.A.transform(lambda x: x.shift(1).rolling(50).mean())
    d["A_prev"] = sh("A", 1)
    d["hi250"] = g.ah.transform(lambda x: x.shift(1).rolling(250, min_periods=100).max())
    d["up"] = (d.ret > 0).astype(int)
    d["streak"] = g.up.transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumsum()) * d.up
    d["cum3"] = d.A / sh("A", 3) - 1

    # coil indicators, as in stocks.add_indicators
    for n in (20, 50, 200):
        d[f"ema{n}"] = g.A.transform(lambda s, n=n: s.ewm(span=n, adjust=False, min_periods=n).mean())
    diff = g.A.diff()
    upm = diff.clip(lower=0).groupby(d.symbol).transform(lambda s: s.ewm(alpha=1 / 14, adjust=False).mean())
    dnm = (-diff).clip(lower=0).groupby(d.symbol).transform(lambda s: s.ewm(alpha=1 / 14, adjust=False).mean())
    d["rsi"] = 100 - 100 / (1 + upm / dnm.replace(0, np.nan))
    d["pos_hi"] = d.A / g.ah.transform(lambda s: s.rolling(85, min_periods=51).max())
    v5 = g.volume.transform(lambda s: s.rolling(5, min_periods=3).mean())
    v20i = g.volume.transform(lambda s: s.rolling(20, min_periods=10).mean())
    d["vol_ratio"] = v5 / v20i.replace(0, np.nan)
    d["range20"] = g.ah.transform(lambda s: s.rolling(20, min_periods=10).max()) / \
        g.al.transform(lambda s: s.rolling(20, min_periods=10).min()) - 1
    d["liq60"] = g.turnover.transform(lambda s: s.rolling(60, min_periods=1).median())

    # forward outcomes from day t's close
    d["n_gap"] = sh("gap", -1)
    d["n_oc"] = sh("oc", -1)
    d["n_ret"] = sh("ret", -1)
    d["n_hi"] = sh("ah", -1) / d.A - 1
    for h in (5, 20):
        d[f"f{h}"] = sh("A", -h) / d.A - 1

    try:
        smap = pd.read_json(os.path.join(CACHE, "sector_map.json"), orient="index")["basic_industry"]
        d["sector"] = d.symbol.map(smap)
    except Exception:
        d["sector"] = np.nan
    try:
        fno = set(json.load(open(os.path.join(CACHE, "fno_symbols.json"))))
    except Exception:
        fno = set()
    if not fno:
        fo = sorted(glob.glob(os.path.join(HERE, "..", "data", "raw", "fo", "fo_*.csv.gz")))
        fno = set(pd.read_csv(fo[-1])["symbol"].astype(str).str.strip()) if fo else set()
    d["fno"] = d.symbol.isin(fno)
    return d


def cooldown(mask: pd.Series, d: pd.DataFrame) -> pd.Series:
    out = np.zeros(len(d), bool)
    pos = np.flatnonzero(mask.to_numpy())
    sym = d.symbol.to_numpy()
    last_sym, last_i = None, -10**9
    rank = d.groupby("symbol").cumcount().to_numpy()
    for i in pos:
        if sym[i] != last_sym or rank[i] - last_i > COOLDOWN:
            out[i] = True
            last_sym, last_i = sym[i], rank[i]
    return pd.Series(out, index=d.index)


def excess(d: pd.DataFrame, mask: pd.Series, col: str, liq: pd.Series, sign: int = 1) -> pd.Series:
    bench = d[liq].groupby("date")[col].median()
    x = d.loc[mask, [col, "date"]].dropna()
    return sign * (x[col] - x.date.map(bench))


def line(label: str, ex: pd.Series, raw: pd.Series | None = None) -> dict:
    ex = ex.dropna()
    if len(ex) < 20:
        return dict(label=label, n=len(ex))
    half = ex.index.to_series().map(lambda i: i)  # placeholder to keep index
    return dict(label=label, n=len(ex), med=100 * ex.median(), avg=100 * ex.mean(), beat=100 * (ex > 0).mean(),
                raw_med=100 * raw.dropna().median() if raw is not None else np.nan)


def halves(d: pd.DataFrame, ex: pd.Series) -> str:
    ex = ex.dropna()
    if ex.empty:
        return ""
    dt = d.loc[ex.index, "date"]
    mid = d.date.min() + (d.date.max() - d.date.min()) / 2
    a, b = ex[dt < mid], ex[dt >= mid]
    return f"1st half {100 * a.median():+.2f}% / beat {100 * (a > 0).mean():.0f}%   2nd half {100 * b.median():+.2f}% / beat {100 * (b > 0).mean():.0f}%"


def main() -> None:
    d = load()
    liq = (d.to20 >= 100) & (d.close >= 10) & (d.n_hist >= 30)          # scanners.py floor
    print(f"{d.date.min().date()} -> {d.date.max().date()}, {d.symbol.nunique()} symbols, "
          f"{d.date.nunique()} sessions\n")

    rows = []

    def report(tab, label, mask, col, sign=1, cool=False):
        m = mask & d[col].notna()
        if cool:
            m = cooldown(m, d)
        ex = excess(d, m, col, liq, sign)
        r = line(f"{tab}: {label}", ex, sign * d.loc[m, col])
        r["halves"] = halves(d, ex)
        rows.append(r)

    # Coils: the seven gates + eligibility (CoilParams defaults)
    coil = ((d.close >= 10) & (d.A > d.ema50) & (d.ema50 > d.ema200) & d.pos_hi.between(0.85, 0.99)
            & d.rsi.between(45, 68) & ((d.A / d.ema20 - 1).abs() <= 0.06) & (d.vol_ratio <= 1.0)
            & (d.range20 <= 0.14) & (d.n_hist >= 200) & (d.liq60 >= 30)).fillna(False)
    report("Coils", "20 sessions", coil, "f20", cool=True)
    report("Coils", "5 sessions", coil, "f5", cool=True)

    # Gap-ups in downtrends: open 2-9% up, prior close below its 50-day average,
    # non-F&O, price >= 50, turnover >= 10 cr. Profit = fade (short open -> close).
    gf = (d.gap.between(0.02, 0.09, inclusive="left") & (d.A_prev < d.sma50_prev) & ~d.fno
          & (d.prev_close >= 50) & (d.to20 >= 1000)).fillna(False)
    report("Gap-Ups in Downtrends", "same-day fade (short open->close)", gf, "oc", sign=-1)
    gf_up = (d.gap.between(0.02, 0.09, inclusive="left") & (d.A_prev >= d.sma50_prev) & ~d.fno
             & (d.prev_close >= 50) & (d.to20 >= 1000)).fillna(False)
    report("  (control)", "same gap-ups in UPtrends, fade", gf_up, "oc", sign=-1)

    # Circuit carry: close at the high on a 5/10/20% band move
    band = (d.ret.between(0.049, 0.051) | d.ret.between(0.098, 0.102) | d.ret.between(0.195, 0.205))
    cc = (band & (d.close >= d.high * 0.999) & (d.close >= 10) & (d.to20 >= 100)).fillna(False)
    d["carry"] = np.where(d.n_gap >= 0.04, d.n_gap, np.where(d.n_hi >= 0.04, 0.04, d.n_ret)) - COST
    report("Circuit Carry", "buy close, sell +4%/gap/close next day (after cost)", cc, "carry")
    d["hit4"] = (d.n_hi >= 0.04).astype(float).where(d.n_hi.notna())
    hit = d.loc[cc, "hit4"].mean()
    base_hit = d.loc[liq & (d.ret > 0.03), "hit4"].mean()

    # Big movers, unusual vol, streaks, 52w breakouts
    lists = {
        "Big Movers": liq & (d.ret >= 0.05),
        "Unusual Vol": liq & (d.volume >= 2 * d.v20) & (d.ret > 0),
        "Streaks": liq & (d.streak >= 3) & (d.cum3 >= 0.05),
        "52W Breakouts": liq & (d.A / d.hi250 - 1 >= -0.02) & (d.ret > 0) & (d.volume >= d.v20),
    }
    for tab, m in lists.items():
        m = m.fillna(False)
        report(tab, "next open (overnight)", m, "n_gap")
        report(tab, "next day close->close", m, "n_ret")
        report(tab, "5 sessions", m, "f5", cool=True)
        report(tab, "20 sessions", m, "f20", cool=True)

    # Sector pulse: top-3 sectors by today's average move (>= 5 liquid names)
    s = d[liq & d.sector.notna()]
    sec = s.groupby(["date", "sector"]).agg(n=("ret", "size"), avg=("ret", "median"),
                                            nxt=("n_ret", "median"), f5=("f5", "median")).reset_index()
    sec = sec[sec.n >= 5]
    mkt = d[liq].groupby("date").agg(nxt=("n_ret", "median"), f5=("f5", "median"))
    sec["rk"] = sec.groupby("date").avg.rank(ascending=False)
    sec["rk_lo"] = sec.groupby("date").avg.rank()
    for lab, sel in (("top-3 sectors", sec.rk <= 3), ("bottom-3 sectors", sec.rk_lo <= 3)):
        t = sec[sel]
        for col, h in (("nxt", "next day"), ("f5", "5 sessions")):
            ex = (t[col] - t.date.map(mkt[col])).dropna()
            rows.append(dict(label=f"Sector Pulse: {lab}, {h}", n=len(ex), med=100 * ex.median(),
                             avg=100 * ex.mean(), beat=100 * (ex > 0).mean(), raw_med=np.nan, halves=""))

    print(f"{'tab / measurement':66s} {'n':>7s} {'median vs typical':>18s} {'avg':>7s} {'beat':>6s}")
    for r in rows:
        if "med" not in r:
            print(f"{r['label']:66s} {r['n']:>7d}   too few")
            continue
        print(f"{r['label']:66s} {r['n']:>7,d} {r['med']:>+17.2f}% {r['avg']:>+6.2f}% {r['beat']:>5.0f}%")
        if r.get("halves"):
            print(f"{'':66s}   {r['halves']}")
    for lab, m in (("downtrend gap-ups", gf), ("uptrend gap-ups", gf_up)):
        x = -d.loc[m, "oc"].dropna()
        print(f"Gap fade, absolute short open->close, {lab}: median {100 * x.median():+.2f}%  avg {100 * x.mean():+.2f}%  "
              f"fell {100 * (x > 0).mean():.0f}%  (before costs; intraday cost ~0.05-0.1%)")
    print(f"\nCircuit Carry: next session reached +4% on {100 * hit:.0f}% of signals "
          f"(any liquid stock up 3%+ that day: {100 * base_hit:.0f}%). Fill at the circuit is not modelled.")


if __name__ == "__main__":
    main()
