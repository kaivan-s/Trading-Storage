"""
Breakout -> retest of the breakout level -> resume. Does the retest identify trends?

Rules fixed before looking at results (long side; short side is the mirror):
  Breakout   close 1%+ above an unbroken swing high (±5 sessions, last 250); that
             swing high is the level L (the highest one broken that day)
  Away       price then trades 3%+ above L, so a later dip is a real pullback
  Retest     within 20 sessions of the breakout, the low comes back within 2% of L
             with no close more than 3% below L (that would be a failed breakout)
  Entries    "retest held"  the retest day closes at or above L
             "confirmed"    within 5 sessions of the retest, a close above the
                            retest day's high
  Volume     breakout on 1.5x+ the usual volume, retest day below 1x
  Compare    entering on the breakout day itself; failed breakouts (close 3%+
             below L within 20 sessions), traded in the opposite direction

Trades enter at the next open; return to the close 5 / 10 / 20 sessions later
vs the average stock (same universe and window), 0.25% cost, one signal per
symbol per horizon. PASS = beats the average stock after cost in both halves.

Run: ~/anaconda3/bin/python attic/retest.py [fo|cash] 2>/dev/null
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

from ema_cross_breakout import load_fo
from falling_patterns import add_forward
from price_action import load_bhav
from rsi_ema_sr import H, benchmark, report
from tab_audit import CACHE

K, MAX_AGE, BREAK, AWAY, ZONE, FAIL, WINDOW, CONFIRM = 5, 250, 0.01, 0.03, 0.02, 0.03, 20, 5
EVENTS = ("breakout", "retest_held", "confirmed", "failed")


def scan(h: np.ndarray, l: np.ndarray, c: np.ndarray, vx: np.ndarray, side: int) -> dict[str, np.ndarray]:
    """side=+1: resistance breakouts and their retests; side=-1: support breakdowns.
    Prices are flipped for the short side so one routine serves both."""
    n = len(c)
    if side < 0:
        h, l, c = -l, -h, -c
    out = {k: np.zeros(n, bool) for k in EVENTS}
    vol_ok = {k: np.zeros(n, bool) for k in ("breakout", "retest_held", "confirmed")}
    piv = np.zeros(n, bool)
    for i in range(K, n - K):
        if h[i] == h[i - K:i + K + 1].max():
            piv[i] = True
    levels: list[list] = []      # [price, formed_i]
    active: list[dict] = []      # breakouts being followed
    for t in range(n):
        i = t - K
        if i >= K and piv[i]:
            levels.append([h[i], i])
        levels = [x for x in levels if t - x[1] <= MAX_AGE]
        margin = lambda lvl: abs(lvl) * BREAK
        broken = [x for x in levels if c[t] > x[0] + margin(x[0])]
        keep_active = []
        for b in active:
            L = b["L"]
            if t - b["t"] > WINDOW and b["touch"] is None:
                continue
            if c[t] < L - abs(L) * FAIL:
                out["failed"][t] = True
                continue
            b["away"] = b["away"] or h[t] >= L + abs(L) * AWAY
            if b["touch"] is None:
                if b["away"] and t - b["t"] >= 2 and l[t] <= L + abs(L) * ZONE:
                    b["touch"], b["touch_hi"] = t, h[t]
                    if c[t] >= L:
                        out["retest_held"][t] = True
                        vol_ok["retest_held"][t] = b["vol"] and vx[t] < 1
                keep_active.append(b)
                continue
            if c[t] > b["touch_hi"]:
                out["confirmed"][t] = True
                vol_ok["confirmed"][t] = b["vol"] and vx[b["touch"]] < 1
                continue
            if t - b["touch"] < CONFIRM:
                keep_active.append(b)
        active = keep_active
        if broken:
            L = max(x[0] for x in broken)
            out["breakout"][t] = True
            vol_ok["breakout"][t] = vx[t] >= 1.5
            active.append({"L": L, "t": t, "away": False, "touch": None, "vol": bool(vx[t] >= 1.5)})
            levels = [x for x in levels if x not in broken]
    for k, v in vol_ok.items():
        out[f"{k}_vol"] = v
    return out


def events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["vol_x"] = df.volume / g.volume.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    cols = {}
    for side, tag in ((1, "up"), (-1, "dn")):
        parts = {}
        for sym, x in g:
            r = scan(x.high.to_numpy(float), x.low.to_numpy(float), x.close.to_numpy(float),
                     x.vol_x.fillna(0).to_numpy(float), side)
            for k, v in r.items():
                parts.setdefault(k, []).append(pd.Series(v, index=x.index))
        for k, v in parts.items():
            cols[f"{tag}_{k}"] = pd.concat(v)
    return pd.concat([df, pd.DataFrame(cols)], axis=1)


def run(df: pd.DataFrame, label: str, liq_cr: float) -> None:
    d = events(df)
    d = add_forward(d, H)
    to20 = d.groupby("symbol").turnover_cr.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    price = d.raw_close if "raw_close" in d else d.close
    uni = ((price >= 50) & (to20 >= liq_cr) & (d["rank"] >= 260)).fillna(False)
    d = benchmark(d, uni)
    u = d[uni].copy()
    for h in H:
        u[f"fwd{h}"] = u[f"fwd{h}"].clip(-0.9, 3)
    print(f"\n{label}: {u.date.min().date()} -> {u.date.max().date()}, {u.symbol.nunique()} symbols")
    print("  per horizon: avg edge vs the average stock in the trade's direction, % right, (1st/2nd half)\n")
    for tag, sign, word in (("up", 1, "Long"), ("dn", -1, "Short")):
        print(f"  {word} side")
        report(f"{word}: on the breakout day" if sign > 0 else f"{word}: on the breakdown day", u[u[f"{tag}_breakout"]], sign)
        report(f"{word}: retest held", u[u[f"{tag}_retest_held"]], sign)
        report(f"{word}: retest confirmed", u[u[f"{tag}_confirmed"]], sign)
        report(f"{word}: breakout on volume", u[u[f"{tag}_breakout_vol"]], sign)
        report(f"{word}: retest held, volume pattern", u[u[f"{tag}_retest_held_vol"]], sign)
        report(f"{word}: confirmed, volume pattern", u[u[f"{tag}_confirmed_vol"]], sign)
        report(f"{word}: FAILED, trade reversed", u[u[f"{tag}_failed"]], -sign)
        print()


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
        run(b.merge(vol, on=["symbol", "date"]), "All liquid cash stocks (bhavcopy)", 5)


if __name__ == "__main__":
    main()
