#!/usr/bin/env python3
"""
Not "where did it close on day 7" but "did it get there at any point inside
the window", with counts.

eval_meanrev.py measured close-to-close at a fixed horizon, which asks
whether you got paid for sitting still. This asks the swing-trader question
instead: over the next 5 or 7 sessions, did the name ever trade far enough
above the entry to take a profit, and how many names out of how many
signalled actually did.

MFE (max favourable excursion) = highest intraday high in the window / entry
MAE (max adverse excursion)    = lowest intraday low in the window / entry

Two things make a raw MFE hit rate meaningless on its own, so both are here:

  CONTROL   Any liquid stock touches +3% inside 7 sessions surprisingly
            often. The number to look at is the signal's rate MINUS the
            all-liquid-names rate on the same days, not the rate itself.
  MAE       MFE alone assumes you sold at the high and never got shaken out.
            A name that touches +5% after first trading down 9% is not a
            trade you would have held. "clean" counts only the names whose
            favourable move came BEFORE any -5% excursion.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import eval_reversal as ev
import reversal as rv

WINDOWS = (5, 7)
TARGETS = (0.02, 0.03, 0.05)
SHAKEOUT = -0.05     # an excursion this deep first, and you are out
MIN_NAMES = 5


def build():
    s = ev.load_all()
    s = s.copy()
    s["_uni"] = ((s["adj"] >= rv.REV_MIN_PRICE)
                 & (s["med_turn60"].fillna(0) >= rv.REV_MIN_TURNOVER)
                 & s["ema200"].notna() & s["from_52w_high"].notna()).fillna(False)

    def piv(c, like=None):
        w = s.pivot_table(index="date", columns="symbol", values=c).sort_index()
        return w if like is None else w.reindex(index=like.index, columns=like.columns)

    P = piv("adj")
    HI = piv("adj_high", P)
    LO = piv("adj_low", P)
    U = piv("_uni", P).fillna(0) > 0
    M20 = piv("mom20", P)
    return s, P, HI, LO, U, M20


def excursions(P, HI, LO, k):
    """
    MFE and MAE over sessions t+1..t+k, plus whether the favourable move
    came first.

    `first_up` walks the window session by session and records, for each
    cell, whether a target was reached before the price ever dipped
    SHAKEOUT below entry. Without it the hit counts flatter names that only
    recovered after a move you would not have sat through.
    """
    p = P.values
    hi = np.stack([HI.shift(-j).values for j in range(1, k + 1)])
    lo = np.stack([LO.shift(-j).values for j in range(1, k + 1)])
    mfe = np.nanmax(hi, axis=0) / p - 1
    mae = np.nanmin(lo, axis=0) / p - 1

    # Running extremes, so "which came first" is answerable per target.
    run_hi = np.fmax.accumulate(hi, axis=0) / p - 1
    run_lo = np.fmin.accumulate(lo, axis=0) / p - 1
    clean = {}
    for tgt in TARGETS:
        hit_j = np.where((run_hi >= tgt).any(axis=0),
                         np.argmax(run_hi >= tgt, axis=0), 10 ** 6)
        shake_j = np.where((run_lo <= SHAKEOUT).any(axis=0),
                           np.argmax(run_lo <= SHAKEOUT, axis=0), 10 ** 6)
        clean[tgt] = (hit_j < 10 ** 6) & (hit_j <= shake_j)
    return mfe, mae, clean


def rate(sel, mfe, mae, clean, dates, U, k):
    """Pooled counts plus a day-paired lift over the universe on the same days."""
    ok = np.isfinite(mfe)
    n = int((sel & ok).sum())
    if n == 0:
        return None
    out = {"n": n,
           "mfe_med": 100 * np.nanmedian(mfe[sel & ok]),
           "mae_med": 100 * np.nanmedian(mae[sel & ok])}
    for tgt in TARGETS:
        hit = (mfe >= tgt) & sel & ok
        out[f"hit{tgt}"] = int(hit.sum())
        out[f"rate{tgt}"] = 100 * hit.sum() / n
        out[f"clean{tgt}"] = 100 * (clean[tgt] & sel & ok).sum() / n
        # Day-paired lift: signal rate minus universe rate, per session.
        diffs = []
        for i in range(len(dates)):
            srow, urow = sel[i] & ok[i], U.values[i] & ok[i]
            if srow.sum() < MIN_NAMES or urow.sum() < 20:
                continue
            diffs.append((mfe[i][srow] >= tgt).mean() - (mfe[i][urow] >= tgt).mean())
        if len(diffs) >= 15:
            d = pd.Series(diffs) * 100
            # lag = k, not 1: a k-session MFE window sampled daily overlaps
            # its neighbours by k-1 sessions.
            t, _ = ev.nw_t(d, k)
            out[f"lift{tgt}"] = d.mean()
            out[f"t{tgt}"] = t
        else:
            out[f"lift{tgt}"], out[f"t{tgt}"] = np.nan, np.nan
    return out


def main() -> None:
    s, P, HI, LO, U, M20 = build()
    dates = P.index.to_numpy()
    Uv = U.values
    rank20 = M20.where(U).rank(axis=1, pct=True).values
    L = "=" * 88

    cells = [
        ("ALL liquid (control)", Uv),
        ("worst 2%  (extreme)", Uv & (rank20 <= 0.02)),
        ("worst 5%", Uv & (rank20 <= 0.05)),
        ("worst decile", Uv & (rank20 <= 0.10)),
        ("worst 30%  (mild)", Uv & (rank20 <= 0.30)),
    ]

    print(f"\n{L}\nDID IT GET THERE *WITHIN* THE WINDOW? — counts\n{L}")
    print(f"window    {pd.Timestamp(dates[0]):%d %b %Y} -> "
          f"{pd.Timestamp(dates[-1]):%d %b %Y}   ({len(dates)} sessions)")
    print(f"universe  {Uv.sum(axis=1).mean():.0f} liquid names/session")
    print("MFE uses intraday highs, so 'reached' means a limit order would")
    print("have filled. 'lift' is the rate minus the control's on the same")
    print("days -- that is the only part attributable to the signal.")

    for k in WINDOWS:
        mfe, mae, clean = excursions(P, HI, LO, k)
        print(f"\n{L}\nWITHIN {k} SESSIONS\n{L}")
        for lab, sel in cells:
            r = rate(sel, mfe, mae, clean, dates, U, k)
            if r is None:
                continue
            print(f"\n  {lab}   {r['n']:,} signals   "
                  f"(median MFE {r['mfe_med']:+.1f}%, median MAE {r['mae_med']:+.1f}%)")
            print(f"    {'target':<10}{'reached':>16}{'rate':>8}{'lift':>9}"
                  f"{'t':>7}{'clean*':>9}")
            for tgt in TARGETS:
                lift = r[f"lift{tgt}"]
                ls = f"{lift:+.1f}pp" if np.isfinite(lift) else "n/a"
                ts = f"{r[f't{tgt}']:.1f}" if np.isfinite(r[f"t{tgt}"]) else ""
                print(f"    {'+' + str(int(tgt * 100)) + '%':<10}"
                      f"{r[f'hit{tgt}']:>9,} / {r['n']:<6,}"
                      f"{r[f'rate{tgt}']:>7.1f}%{ls:>9}{ts:>7}"
                      f"{r[f'clean{tgt}']:>8.1f}%")
        print(f"\n  * clean = reached the target BEFORE ever trading "
              f"{SHAKEOUT:.0%} below entry")


if __name__ == "__main__":
    main()
