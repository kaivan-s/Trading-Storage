"""
Support/resistance with volume confirmation.

Same signals and outcome as price_action.py (+1 ATR before -1 ATR within 10
sessions from the next open), split by the signal day's volume against the
stock's own prior 20-session median, and (bhavcopy only) by delivery %
against the stock's own prior 20-session median.

Two controls per row:
  similar days       same 5-day move decile x distance-from-50-day quintile
  + same volume      also the same volume bucket, so "support held on 3x
                     volume" is compared with other 3x-volume days rather
                     than with all days. Heavy-volume days have their own
                     behaviour; this isolates what the LEVEL adds.

Run: ~/anaconda3/bin/python attic/sr_volume.py 2>/dev/null
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ema_cross_breakout import load_fo  # noqa: E402
from price_action import DOWN_CONCEPTS, ETF, HERE, load_bhav, per_symbol  # noqa: E402

CONCEPTS = ["Support hold", "Support, 2+ tests", "Resistance reject", "Support break", "Resistance break",
            "Role reversal", "Double bottom"]
VOL_BINS = [0, 0.7, 1.3, 2, 3, np.inf]
VOL_LABELS = ["quiet <0.7x", "normal 0.7-1.3x", "1.3-2x", "2-3x", "heavy 3x+"]


def bhav_volume() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(HERE, "..", "data", "raw", "bhav_*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=["symbol", "series", "date", "volume", "deliv_pct"]) for f in fs])
    d = d[(d.series == "EQ") & ~d.symbol.str.contains(ETF)].drop(columns="series")
    d["date"] = pd.to_datetime(d.date)
    return d.drop_duplicates(["symbol", "date"])


def add_vol(x: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    v = vol.sort_values(["symbol", "date"]).copy()
    g = v.groupby("symbol", sort=False)
    v["vol_x"] = v.volume / g.volume.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    if "deliv_pct" in v:
        v["deliv_rel"] = v.deliv_pct - g.deliv_pct.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    keep = ["symbol", "date", "vol_x"] + (["deliv_rel"] if "deliv_rel" in v else [])
    return x.merge(v[keep], on=["symbol", "date"], how="left")


def run(df: pd.DataFrame, vol: pd.DataFrame, label: str) -> None:
    parts = [per_symbol(g) for _, g in df.groupby("symbol", sort=False)]
    x = pd.concat([p for p in parts if p is not None], ignore_index=True)
    x = add_vol(x, vol)
    x = x[x.liq & x.out.notna() & x.vol_x.notna()].copy()
    x["b_r5"] = pd.qcut(x.r5, 10, labels=False, duplicates="drop")
    x["b_pos"] = pd.qcut(x.pos50, 5, labels=False, duplicates="drop")
    x["b_vol"] = pd.cut(x.vol_x, VOL_BINS, labels=False, right=False)
    x = x.join(x.groupby(["b_r5", "b_pos"]).out.mean().rename("ctrl"), on=["b_r5", "b_pos"])
    x = x.join(x.groupby(["b_r5", "b_pos", "b_vol"]).out.mean().rename("ctrl_v"), on=["b_r5", "b_pos", "b_vol"])
    has_deliv = "deliv_rel" in x and x.deliv_rel.notna().mean() > 0.5
    if has_deliv:
        x["b_del"] = pd.cut(x.deliv_rel, [-np.inf, -10, 10, np.inf], labels=False)
        x = x.join(x.groupby(["b_r5", "b_pos", "b_vol", "b_del"]).out.mean().rename("ctrl_vd"),
                   on=["b_r5", "b_pos", "b_vol", "b_del"])
    mid = x.date.min() + (x.date.max() - x.date.min()) / 2

    print(f"\n{'=' * 22} {label}: {x.date.min().date()} -> {x.date.max().date()} {'=' * 22}")
    print("Volume days on their own (no level), chance of +1 ATR first vs similar days:")
    for b, lab in enumerate(VOL_LABELS):
        e = x[x.b_vol == b]
        print(f"   {lab:16s} n={len(e):>8,d}   {100 * e.out.mean():5.1f}% vs {100 * e.ctrl.mean():5.1f}%  "
              f"edge {100 * (e.out - e.ctrl).mean():+.1f}")
    print(f"\n{'signal':22s} {'volume':16s} {'n':>7s} {'right way':>10s} {'edge vs similar':>16s}"
          f" {'edge vs same-volume':>20s} {'1st / 2nd half':>15s}")
    for k in CONCEPTS:
        down = k in DOWN_CONCEPTS
        for b, lab in enumerate(VOL_LABELS):
            e = x[x[k] & (x.b_vol == b)]
            if len(e) < 100:
                continue
            p = 1 - e.out if down else e.out
            q = 1 - e.ctrl if down else e.ctrl
            qv = 1 - e.ctrl_v if down else e.ctrl_v
            ev = p - qv
            print(f"{k + (' (dn)' if down else ''):22s} {lab:16s} {len(e):>7,d} {100 * p.mean():>9.1f}%"
                  f" {100 * (p - q).mean():>+15.1f} {100 * ev.mean():>+19.1f}"
                  f"   {100 * ev[e.date < mid].mean():>+5.1f} / {100 * ev[e.date >= mid].mean():>+5.1f}")
        print()
    if has_deliv:
        print("Delivery % vs the stock's own 20-day median (high = 10+ points above), heavy volume 2x+ only,"
              " edge vs same volume AND delivery bucket:")
        for k in CONCEPTS:
            down = k in DOWN_CONCEPTS
            for db, lab in ((0, "low delivery"), (2, "high delivery")):
                e = x[x[k] & (x.vol_x >= 2) & (x.b_del == db)]
                if len(e) < 100:
                    continue
                p = 1 - e.out if down else e.out
                qv = 1 - e.ctrl_v if down else e.ctrl_v
                qvd = 1 - e.ctrl_vd if down else e.ctrl_vd
                print(f"   {k:20s} {lab:14s} n={len(e):>6,d}  right way {100 * p.mean():5.1f}%  "
                      f"edge vs same volume {100 * (p - qv).mean():+5.1f}  vs same volume+delivery {100 * (p - qvd).mean():+5.1f}")


if __name__ == "__main__":
    run(load_bhav(), bhav_volume(), "All liquid NSE stocks")
    fo = load_fo()
    run(fo, fo[["symbol", "date", "volume"]], "F&O stocks (Yahoo, adjusted)")
