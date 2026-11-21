"""
"Crossed pivot" screen: an uptrending stock that closes above its daily R1
floor pivot (from the previous session's high, low and close) on an up day.

  python attic/pivot_cross.py profile   -> today's list vs a sample list
  python attic/pivot_cross.py           -> short-term backtest (1-5 sessions)

Setup on day t: close above the 10/20/50/200 EMAs, close above daily R1,
up at least 1% on the day, volume at least the 50-day median.
Variants add "within 5% of the 52-week high" and "volume 2x+".
Returns are measured from the next session's open, plus the overnight move
if bought at the signal day's close.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ema_cross_breakout import COST, bracket, load_all, load_fo  # noqa: E402

SAMPLE = "ALICON GCSL ESAFSFB NAZARA INDUSTOWER COSMOFIRST MAXESTATES SBC DHOOTTRANS".split()


def features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c, h, l, v = g.close, g.high, g.low, g.volume
    ema = {n: c.ewm(span=n, adjust=False, min_periods=n).mean() for n in (10, 20, 50, 200)}
    pp = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
    r1, r2 = 2 * pp - l.shift(1), pp + (h.shift(1) - l.shift(1))
    g["d1"] = c / c.shift(1) - 1
    g["vol_x"] = v / v.shift(1).rolling(50, min_periods=30).median()
    g["from_hi"] = c / h.rolling(250, min_periods=200).max() - 1
    g["above_all"] = (c > ema[10]) & (c > ema[20]) & (c > ema[50]) & (c > ema[200]) & ema[200].notna()
    g["piv"] = np.select([c > r2, c > r1, c > pp], ["R2+", "R1", "PP-R1"], "<PP")
    g["from20"] = c / ema[20] - 1
    g["liq"] = (g.turnover_cr.rolling(20).median() >= 2) & (c >= 50)
    g["setup"] = g.above_all & (c > r1) & (g.d1 >= 0.01) & (g.vol_x >= 1)
    g["setup_hi"] = g.setup & (g.from_hi >= -0.05)
    g["setup_vol2"] = g.setup & (g.vol_x >= 2)
    return g


def profile() -> None:
    df = load_all()
    df = df[df.date >= df.date.max() - pd.Timedelta(days=500)]
    f = pd.concat([features(g).tail(1) for _, g in df.groupby("symbol", sort=False)]).set_index("symbol")
    f = f[f.date == f.date.max()]
    s = f.reindex(SAMPLE)
    pd.set_option("display.width", 220)
    print("as of", f.date.max().date())
    print(s[["close", "d1", "vol_x", "from_hi", "from20", "piv", "above_all", "setup", "setup_hi"]].round(3).to_string())
    liq = f[f.liq]
    for k in ("setup", "setup_hi", "setup_vol2"):
        print(f"{k}: {int(liq[k].sum())} liquid stocks pass today, covers {s[k].mean():.0%} of the sample")


def backtest(df: pd.DataFrame, start: str) -> None:
    cols = ("setup", "setup_hi", "setup_vol2", "above_all")
    res = {k: [] for k in cols + ("all",)}
    rng = np.random.default_rng(0)
    for _, g in df.groupby("symbol", sort=False):
        g = features(g)
        o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        dts = g.date.to_numpy()
        ok = g.liq.to_numpy() & (dts >= np.datetime64(start))
        ok[-6:] = False
        sets = {k: np.flatnonzero(ok & g[k].to_numpy()) for k in cols}
        pool = np.flatnonzero(ok)
        sets["all"] = rng.choice(pool, size=min(len(pool), 40), replace=False) if len(pool) else pool
        # one random above-all day per 10 so the baseline isn't dominated by long trends
        sets["above_all"] = sets["above_all"][::10]
        for k, idx in sets.items():
            for i in idx:
                e = o[i + 1]
                res[k].append(dict(ovn=e / c[i] - 1, r1=c[i + 1] / e - 1, r3=c[i + 3] / e - 1, r5=c[i + 5] / e - 1,
                                   cc1=c[i + 1] / c[i] - 1,
                                   b32=bracket(o, h, l, c, i, 0.03, 0.02) - COST,
                                   b53=bracket(o, h, l, c, i, 0.05, 0.03) - COST))
    names = {"setup": "Crossed R1, up 1%+, vol 1x+, above all EMAs", "setup_hi": "...and within 5% of 52w high",
             "setup_vol2": "...or with 2x+ volume", "above_all": "Any day above all 4 EMAs", "all": "Random liquid stock-day"}
    print(f"{'':50s} {'overnight':>9s} {'close->close 1d':>15s} {'1d':>6s} {'3d':>6s} {'5d':>6s}  {'+3/-2':>15s}  {'+5/-3':>15s}")
    for k in cols + ("all",):
        t = pd.DataFrame(res[k])
        m = lambda s: f"{100 * t[s].mean():+5.2f}%"
        print(f"{names[k] + f' (n={len(t)})':50s} {m('ovn'):>9s} {m('cc1'):>15s} {m('r1'):>6s} {m('r3'):>6s} {m('r5'):>6s}"
              f"  {m('b32')} w{100 * (t.b32 > 0).mean():.0f}%  {m('b53')} w{100 * (t.b53 > 0).mean():.0f}%")
        print(f"{'':50s} {'up':>9s} {100 * (t.ovn > 0).mean():>14.0f}% {100 * (t.r1 > 0).mean():>5.0f}% "
              f"{100 * (t.r3 > 0).mean():>5.0f}% {100 * (t.r5 > 0).mean():>5.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        profile()
    else:
        print("=" * 20, "F&O stocks, 2010 onward (averages)", "=" * 20)
        backtest(load_fo(), "2010-01-01")
        print("\n" + "=" * 20, "All liquid NSE stocks, Sep 2024 onward", "=" * 20)
        backtest(load_all(), "2024-09-01")
