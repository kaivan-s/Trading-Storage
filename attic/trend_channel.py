"""
Trend channels: do the channel lines act as support/resistance?

Channel: least-squares line through log closes of the previous W sessions
(W = 60 and 120), extended one day forward, with bands at +/- 2 standard
deviations of the residuals. Only clean channels count: R^2 >= 0.6.
Fitting on the window that ENDS YESTERDAY keeps the touch day itself out
of its own channel.

Signals (each with its textbook expectation)
  Up channel,   lower-line bounce   low reaches the lower line, close back above   -> up
  Up channel,   lower-line break    close 1% below the lower line                  -> down
  Up channel,   upper-line reject   high reaches the upper line, close below       -> down
  Up channel,   upper-line break    close above the upper line                     -> up
  Down channel, upper-line reject   high reaches the upper line, close below       -> down
  Down channel, upper-line break    close 1% above the upper line                  -> up
  Down channel, lower-line bounce   low reaches the lower line, close back above   -> up

Outcome and control are exactly those of price_action.py: +1 ATR before
-1 ATR within 10 sessions from the next open, compared with days that had the
same 5-day move and the same distance from the 50-day average.

Run: ~/anaconda3/bin/python attic/trend_channel.py 2>/dev/null
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ema_cross_breakout import load_fo  # noqa: E402
from price_action import load_bhav, per_symbol  # noqa: E402

R2_MIN = 0.6
DOWN = {"lower-line break", "upper-line reject"}


def channel(y: np.ndarray, w: int):
    """Rolling OLS of y on 0..w-1 over the window ending at t-1, projected to t."""
    n = len(y)
    x = np.arange(w, dtype=float)
    xb, sxx = x.mean(), ((x - x.mean()) ** 2).sum()
    yy = np.nan_to_num(y)
    sy = np.convolve(yy, np.ones(w), "valid")
    sxy = np.convolve(yy, x[::-1], "valid")
    syy = np.convolve(yy ** 2, np.ones(w), "valid")
    yb = sy / w
    b = (sxy - w * xb * yb) / sxx
    sst = syy - w * yb ** 2
    sse = np.maximum(sst - b ** 2 * sxx, 0)
    r2 = 1 - sse / np.where(sst > 0, sst, np.nan)
    s = np.sqrt(sse / (w - 2))
    pred = yb + b * (w - xb)
    out = [np.full(n, np.nan) for _ in range(4)]
    # window k covers y[k:k+w]; its projection applies to day k+w
    for arr, v in zip(out, (pred, s, b, r2)):
        arr[w:] = v[:n - w]
    return out


def signals(g: pd.DataFrame, base: pd.DataFrame, w: int) -> pd.DataFrame:
    h, l, c = (g[k].to_numpy(float) for k in ("high", "low", "close"))
    pred, s, b, r2 = channel(np.log(c), w)
    up_line, lo_line = np.exp(pred + 2 * s), np.exp(pred - 2 * s)
    clean = r2 >= R2_MIN
    upc, dnc = clean & (b > 0), clean & (b < 0)
    prev_c = np.r_[np.nan, c[:-1]]
    sig = {
        "Up channel: lower-line bounce": upc & (l <= lo_line) & (c > lo_line) & (prev_c > lo_line),
        "Up channel: lower-line break": upc & (c < lo_line * 0.99),
        "Up channel: upper-line reject": upc & (h >= up_line) & (c < up_line) & (prev_c < up_line),
        "Up channel: upper-line break": upc & (c > up_line),
        "Down channel: upper-line reject": dnc & (h >= up_line) & (c < up_line) & (prev_c < up_line),
        "Down channel: upper-line break": dnc & (c > up_line * 1.01),
        "Down channel: lower-line bounce": dnc & (l <= lo_line) & (c > lo_line) & (prev_c > lo_line),
    }
    for k, v in sig.items():
        base[f"{w}|{k}"] = v
    return base


def run(df: pd.DataFrame, label: str) -> None:
    parts = []
    for _, g in df.groupby("symbol", sort=False):
        base = per_symbol(g)
        if base is None:
            continue
        for w in (60, 120):
            base = signals(g.reset_index(drop=True), base, w)
        parts.append(base)
    x = pd.concat(parts, ignore_index=True)
    x = x[x.liq & x.out.notna()].copy()
    x["b_r5"] = pd.qcut(x.r5, 10, labels=False, duplicates="drop")
    x["b_pos"] = pd.qcut(x.pos50, 5, labels=False, duplicates="drop")
    x = x.join(x.groupby(["b_r5", "b_pos"]).out.mean().rename("ctrl"), on=["b_r5", "b_pos"])
    x["f5_ex"] = x.f5 - x.groupby("date").f5.transform("median")
    mid = x.date.min() + (x.date.max() - x.date.min()) / 2

    print(f"\n{'=' * 20} {label}: {x.date.min().date()} -> {x.date.max().date()}, {x.symbol.nunique()} stocks {'=' * 20}")
    print(f"{'window | signal':46s} {'expects':>7s} {'n':>7s} {'right way':>10s} {'similar days':>13s} {'edge':>6s}"
          f" {'1st / 2nd half':>16s} {'5d vs typical':>14s}")
    for col in [c for c in x.columns if "|" in c]:
        e = x[x[col]]
        if len(e) < 30:
            print(f"{col:46s} too few ({len(e)})")
            continue
        down = any(k in col for k in DOWN)
        p = 1 - e.out if down else e.out
        q = 1 - e.ctrl if down else e.ctrl
        edge = p - q
        fx = (-e.f5_ex if down else e.f5_ex).median()
        print(f"{col:46s} {'down' if down else 'up':>7s} {len(e):>7,d} {100 * p.mean():>9.1f}% {100 * q.mean():>12.1f}%"
              f" {100 * edge.mean():>+5.1f}  {100 * edge[e.date < mid].mean():>+6.1f} / {100 * edge[e.date >= mid].mean():>+5.1f}"
              f" {100 * fx:>+13.2f}%")


if __name__ == "__main__":
    run(load_bhav(), "All liquid NSE stocks")
    run(load_fo(), "F&O stocks (Yahoo, adjusted)")
