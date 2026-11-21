"""
RSI + EMA trend + support/resistance, combined. Do the combinations call
direction better than the pieces on their own?

Eight setups fixed before looking at results (four up, four down), plus each
component alone as a reference.

  Up 1  Trend pullback   close > EMA200 and EMA50 > EMA200; RSI(14) crosses back
                         above 40; support held or the low touched EMA50 within 3 days
  Up 2  EMA50 bounce     EMA20 > EMA50 > EMA200; low within 1% of EMA50, close above; RSI 40-55
  Up 3  Breakout         close 1%+ above an unbroken swing high; RSI 55-70; close > EMA20 > EMA50
  Up 4  Oversold support RSI < 30; support held today; close > EMA200
  Down 1-4               mirror images (RSI 60 / 45-60 / 30-45 / 70; resistance; EMA below)

Support/resistance = unbroken ±5-session swing lows/highs from the last 250
sessions (price_action.per_symbol). Trades enter at the next open; return to
the close 5 / 10 / 20 sessions later vs the average stock in the same universe
over the same window, 0.25% cost, one signal per symbol per horizon.
PASS = beats the typical stock on average after cost in both halves.

Run: ~/anaconda3/bin/python attic/rsi_ema_sr.py [fo|cash] 2>/dev/null
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

from ema_cross_breakout import load_fo
from falling_patterns import add_forward, cooldown
from price_action import load_bhav, per_symbol
from tab_audit import CACHE, COST

H = (5, 10, 20)


def benchmark(d: pd.DataFrame, uni: pd.Series) -> pd.DataFrame:
    """Same-day cross-sectional MEAN, so a random stock scores zero on average.
    (A median benchmark against average trade returns flatters every long and
    penalises every short, because returns are right-skewed.)"""
    for h in H:
        x = d.loc[uni, f"fwd{h}"].clip(-0.9, 3)
        d[f"bench{h}"] = d.date.map(x.groupby(d.loc[uni, "date"]).mean())
    return d


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    parts = [per_symbol(g) for _, g in df.groupby("symbol", sort=False)]
    sr = pd.concat([p for p in parts if p is not None], ignore_index=True)[
        ["symbol", "date", "Support hold", "Resistance reject", "Support break", "Resistance break"]]
    df = df.merge(sr, on=["symbol", "date"], how="left")
    for k in ("Support hold", "Resistance reject", "Support break", "Resistance break"):
        df[k] = df[k].fillna(False).astype(bool)
    g = df.groupby("symbol", sort=False)
    for n in (20, 50, 200):
        df[f"e{n}"] = g.close.transform(lambda s, n=n: s.ewm(span=n, adjust=False, min_periods=n).mean())
    diff = g.close.diff()
    up = diff.clip(lower=0).groupby(df.symbol).transform(lambda s: s.ewm(alpha=1 / 14, adjust=False).mean())
    dn = (-diff).clip(lower=0).groupby(df.symbol).transform(lambda s: s.ewm(alpha=1 / 14, adjust=False).mean())
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    df["rsi_p"] = g.rsi.shift(1)
    any3 = lambda s: s.groupby(df.symbol).transform(lambda x: x.rolling(3, min_periods=1).max()).astype(bool)
    df["touch50_up"] = (df.low <= df.e50 * 1.01) & (df.close > df.e50)
    df["touch50_dn"] = (df.high >= df.e50 * 0.99) & (df.close < df.e50)
    df["sup3"] = any3(df["Support hold"] | df.touch50_up)
    df["res3"] = any3(df["Resistance reject"] | df.touch50_dn)
    med20 = g.volume.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    df["vol_x"] = df.volume / med20                              # today vs usual
    df["vol_5d"] = g.volume.transform(lambda s: s.rolling(5, min_periods=5).mean()) / med20   # last 5 days vs usual
    return df


VOL_BINS = [(0, 0.7, "quiet <0.7x"), (0.7, 1.5, "normal 0.7-1.5x"), (1.5, 3, "heavy 1.5-3x"), (3, np.inf, "very heavy 3x+")]


def volume_setups(d: pd.DataFrame) -> dict[str, tuple[pd.Series, int]]:
    s = setups(d)
    v, v5 = d.vol_x, d.vol_5d
    m = lambda k: s[k][0]
    return {
        "Up 1v pullback, volume drying up": (m("Up 1 trend pullback") & (v5 < 0.8), +1),
        "Up 2v EMA50 bounce, volume drying up": (m("Up 2 EMA50 bounce") & (v5 < 0.8), +1),
        "Up 3v breakout on 2x+ volume": (m("Up 3 breakout") & (v >= 2), +1),
        "Up 4v oversold at support, 2x+ volume": (m("Up 4 oversold at support") & (v >= 2), +1),
        "Dn 1v rally fails, rally on light volume": (m("Dn 1 downtrend rally fails") & (v5 < 0.8), -1),
        "Dn 1h rally fails, 1.5x+ volume on the turn": (m("Dn 1 downtrend rally fails") & (v >= 1.5), -1),
        "Dn 2v EMA50 rejection, 1.5x+ volume": (m("Dn 2 EMA50 rejection") & (v >= 1.5), -1),
        "Dn 3v breakdown on 2x+ volume": (m("Dn 3 breakdown") & (v >= 2), -1),
        "Dn 4v overbought at resistance, 2x+ volume": (m("Dn 4 overbought at resistance") & (v >= 2), -1),
    }


def volume_study(u: pd.DataFrame) -> None:
    h = 10
    print(f"\n  Each setup by the signal day's volume ({h}-session horizon: avg edge, % right, n)")
    print(f"  {'':<34}" + "".join(f"{lab:>24}" for _, _, lab in VOL_BINS))
    for name, (m, sign) in setups(u).items():
        cells = []
        for lo, hi, _ in VOL_BINS:
            e = cooldown(u[m.fillna(False) & (u.vol_x >= lo) & (u.vol_x < hi)].dropna(subset=[f"fwd{h}"]), h)
            if len(e) < 50:
                cells.append(f"{'n=' + str(len(e)):>24}")
                continue
            ex = sign * (e[f"fwd{h}"] - e[f"bench{h}"])
            cells.append(f"{100*ex.mean():+6.2f}% {100*(ex > 0).mean():3.0f}% n={len(e):<6}")
        print(f"  {name:<34}" + "".join(f"{c:>24}" for c in cells))
    print("\n  Volume-confirmed versions, fixed in advance:\n")
    for name, (m, sign) in volume_setups(u).items():
        report(name, u[m.fillna(False)], sign)


def setups(d: pd.DataFrame) -> dict[str, tuple[pd.Series, int]]:
    c, r, rp = d.close, d.rsi, d.rsi_p
    up = (c > d.e200) & (d.e50 > d.e200)
    dn = (c < d.e200) & (d.e50 < d.e200)
    x40 = (r > 40) & (rp <= 40)
    x60 = (r < 60) & (rp >= 60)
    return {
        # reference: components alone
        "ref  uptrend (EMA)": (up, +1),
        "ref  RSI crosses up through 40": (x40, +1),
        "ref  support held": (d["Support hold"], +1),
        "ref  resistance broken": (d["Resistance break"], +1),
        "Up 1 trend pullback": (up & x40 & d.sup3, +1),
        "Up 2 EMA50 bounce": ((d.e20 > d.e50) & (d.e50 > d.e200) & d.touch50_up & r.between(40, 55), +1),
        "Up 3 breakout": (d["Resistance break"] & r.between(55, 70) & (c > d.e20) & (d.e20 > d.e50), +1),
        "Up 4 oversold at support": ((r < 30) & d["Support hold"] & (c > d.e200), +1),
        "ref  downtrend (EMA)": (dn, -1),
        "ref  RSI crosses down through 60": (x60, -1),
        "ref  resistance rejected": (d["Resistance reject"], -1),
        "ref  support broken": (d["Support break"], -1),
        "Dn 1 downtrend rally fails": (dn & x60 & d.res3, -1),
        "Dn 2 EMA50 rejection": ((d.e20 < d.e50) & (d.e50 < d.e200) & d.touch50_dn & r.between(45, 60), -1),
        "Dn 3 breakdown": (d["Support break"] & r.between(30, 45) & (c < d.e20) & (d.e20 < d.e50), -1),
        "Dn 4 overbought at resistance": ((r > 70) & d["Resistance reject"] & (c < d.e200), -1),
    }


def report(name: str, ev: pd.DataFrame, sign: int) -> list[str]:
    cells, passes = [], 0
    for h in H:
        e = cooldown(ev.dropna(subset=[f"fwd{h}", f"bench{h}"]), h)
        if len(e) < 50:
            cells.append(f"{h}d n={len(e)} too few")
            continue
        ex = sign * (e[f"fwd{h}"] - e[f"bench{h}"])
        mid = e.date.min() + (e.date.max() - e.date.min()) / 2
        a, b = ex[e.date < mid].mean(), ex[e.date >= mid].mean()
        ok = ex.mean() > COST and a > COST and b > COST
        passes += ok
        cells.append(f"{h}d n={len(e):>5} {100*ex.mean():+5.2f}% right {100*(ex > 0).mean():3.0f}% "
                     f"({100*a:+.2f}/{100*b:+.2f}){' PASS' if ok else '     '}")
    print(f"  {name:<34}" + "  |  ".join(cells))
    return cells


def run(df: pd.DataFrame, label: str, liq_cr: float) -> None:
    d = indicators(df)
    d = add_forward(d, H)
    to20 = d.groupby("symbol").turnover_cr.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    price = d.raw_close if "raw_close" in d else d.close
    uni = ((price >= 50) & (to20 >= liq_cr) & d.e200.notna()).fillna(False)
    d = benchmark(d, uni)
    u = d[uni].copy()
    for h in H:
        u[f"fwd{h}"] = u[f"fwd{h}"].clip(-0.9, 3)
    print(f"\n{label}: {u.date.min().date()} -> {u.date.max().date()}, {u.symbol.nunique()} symbols")
    print("  per horizon: avg edge vs the average stock in the setup's direction, % of trades right, (1st/2nd half)\n")
    for name, (m, sign) in setups(u).items():
        report(name, u[m.fillna(False)], sign)
        if name.startswith("Up 4"):
            print()
    volume_study(u)


def main() -> None:
    which = sys.argv[1:] or ["fo", "cash"]
    if "fo" in which:
        fo = load_fo()
        run(fo[fo.date >= "2009-01-01"], "F&O stocks (Yahoo, adjusted; today's list)", 10)
    if "cash" in which:
        b = load_bhav()
        vol = pd.concat([pd.read_parquet(f, columns=["symbol", "series", "date", "volume"])
                         for f in sorted(glob.glob(os.path.join(CACHE, "..", "raw", "bhav_*.parquet")))])
        vol = vol[vol.series == "EQ"].drop(columns="series").drop_duplicates(["symbol", "date"])
        b = b.merge(vol, on=["symbol", "date"])
        run(b, "All liquid cash stocks (bhavcopy)", 5)


if __name__ == "__main__":
    main()
