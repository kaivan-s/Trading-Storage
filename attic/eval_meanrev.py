#!/usr/bin/env python3
"""
Mean reversion tested the way it is actually traded.

eval_reversal.py tested the wrong thing three ways, and the 1-session
headline was the least of it:

  FORMATION   It used a 5-day drop. The documented short-term reversal
              factor is one-month formation, and mom20's in-pool IC (-0.020,
              t=-1.72) was the stronger read anyway. Swept by decile here.
  EXIT        It held a fixed clock. Mean reversion is defined by its exit --
              you sell when the name has reverted, not when the calendar
              says so. Fixed 5 and 7 are compared against three reversion
              rules, each capped at 10 sessions.
  SEVERITY    It gated on a 2% drop and RSI <= 55, which is a mild pullback.
              Swept down to the genuinely stretched.

BENCHMARK is the liquid universe's own cumulative median return over the SAME
dates as each trade, so variable-length holds stay comparable to each other
and to the fixed ones.

STATS are median excess per entry day with Newey-West at lag = max hold,
because overlapping windows are what made the last round look significant.
n_ind reports how many independent periods 15 months actually contains.

Two traps this script exists to avoid, both of which produced a fake +0.6%
to +0.9% before the control row was added:

  BENCHMARK  Compounding daily cross-sectional medians is NOT the universe's
             median compounded return -- median(product) != product(median),
             and the gap widens with horizon. It handed every leg a free
             +0.6% at 5 sessions whether it had selected anything or not.
             The benchmark is now the universe's median k-session forward
             return measured on the entry date, which makes the control read
             exactly 0.00% on fixed holds, as it must.
  EXIT RULE  "Sell on the first profitable close" has a positive median
             almost by construction: winners leave in a day with a small
             gain while losers run to the cap. It scores +0.88% on the
             control with NO selection. Same for "RSI back above 50" (+0.37%,
             and its hold collapses to 1 session on names already above 50,
             so it is not even the same trade). Fixed holds are the only
             honest leg here; the reversion exits are reported to show they
             are artifacts, not to be used.

VERDICT: there IS an edge, but only in the extreme tail.
Worst 2% of 20-day performers, ~8 names/session, held 7 sessions: +0.58%
median excess, NW t=2.94, p=0.003, +0.28% net of a 0.30% round trip. At 10
sessions +0.96%, p=0.004, +0.66% net. It is monotonic in severity (worst 2%
> worst 5% > decile > 2nd decile ~ 0), both halves agree, and unlike the
illiquidity artifact in eval_reversal.py it is LIQUID -- median turnover Rs
1,744 lakh/day against the universe's Rs 1,323, only 22% under Rs 500 lakh.

eval_meanrev_mfe.py asks the same question the other way -- did the name
ever trade far enough above entry INSIDE the window, rather than where it
closed at the end -- and agrees: 1,829 of 2,693 extreme-tail signals (67.9%)
reached +3% within 7 sessions against a 62.7% control, a +5.0pp lift at t=3.0,
with the same monotonic decay through the milder buckets.

Caveats before sizing anything: 8 names/day is a thin list, 7-10 session
horizons leave only 11-16 independent periods in the cache, the mean runs
far above the median (+1.45% vs +0.58% at 7d) so outcomes are skewed and
variable, and many cells were swept here -- monotonicity across severity
and the halves agreeing are the defence against that, not a single p-value.
The diluted versions (worst decile and looser) do NOT work.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import eval_reversal as ev
import reversal as rv

MAXH = 10          # cap on any reversion exit
COST = 0.30        # % round trip
MIN_NAMES = 5      # entry days with fewer selected names are dropped
RSI_EXIT = 50.0
STOP = -0.05       # close-based stop for the stop variant


def nw_p(x: pd.Series, lag: int) -> tuple[float, float, float]:
    t, _ = ev.nw_t(x, lag)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if np.isfinite(t) else np.nan
    return x.median(), t, p


def build():
    s = ev.load_all()
    uni = ((s["adj"] >= rv.REV_MIN_PRICE)
           & (s["med_turn60"].fillna(0) >= rv.REV_MIN_TURNOVER)
           & s["ema200"].notna() & s["from_52w_high"].notna())
    s = s.copy()
    s["_uni"] = uni.fillna(False)

    # pivot_table drops all-NaN rows and columns, so indicator grids come out
    # smaller than the price grid. Everything is reindexed onto price's axes.
    def piv(c, like=None):
        w = s.pivot_table(index="date", columns="symbol", values=c).sort_index()
        return w if like is None else w.reindex(index=like.index, columns=like.columns)

    P = piv("adj")
    R = piv("rsi", P)
    U = piv("_uni", P).fillna(0) > 0
    M20 = piv("mom20", P)
    M5 = piv("mom5", P)
    F52 = piv("from_52w_high", P)

    # Forward matrices, k = 1..MAXH.
    fwd = np.stack([(P.shift(-k) / P - 1).values for k in range(1, MAXH + 1)])
    rsi_f = np.stack([R.shift(-k).values for k in range(1, MAXH + 1)])

    # Benchmark: the universe's own median k-session forward return, measured
    # on the entry date. Compounding daily cross-sectional medians instead
    # looks equivalent and is not -- median(product) != product(median), and
    # the gap grows with horizon, which hands every leg a fake +0.6% at 5
    # sessions whether it selected anything or not.
    bench = np.stack([
        np.nanmedian(np.where(U.values, fwd[k], np.nan), axis=1, keepdims=True)
        for k in range(MAXH)
    ])
    bench = np.repeat(bench, P.shape[1], axis=2)
    return s, P, R, U, M20, M5, F52, fwd, rsi_f, bench


def first_true(cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the first True along axis 0, and whether any was found."""
    any_ = cond.any(axis=0)
    idx = np.argmax(cond, axis=0)
    return np.where(any_, idx, MAXH - 1), any_


def exits(fwd, rsi_f):
    """
    {name -> (excess-before-benchmark return, holding length)} per cell.

    Every rule is capped at MAXH. `fwd[k]` is the return from entry to k+1
    sessions later, so an exit index of k means a hold of k+1 sessions.
    """
    out = {}
    # fixed 10d is the matched control for the RSI exit: same selection, same
    # cap. The all-universe control cannot price that rule, because a name
    # already above RSI 50 exits on day 1 while an oversold one runs the full
    # 10 sessions -- that comparison is two different trades, not two signals.
    for k in (5, 7, 10):
        out[f"fixed {k}d"] = (fwd[k - 1], np.full(fwd.shape[1:], float(k)))

    prof_i, _ = first_true(fwd > 0)
    out["first green close"] = (np.take_along_axis(fwd, prof_i[None], 0)[0],
                                prof_i.astype(float) + 1)

    rsi_i, _ = first_true(rsi_f > RSI_EXIT)
    out[f"RSI back > {RSI_EXIT:.0f}"] = (np.take_along_axis(fwd, rsi_i[None], 0)[0],
                                          rsi_i.astype(float) + 1)

    # First green close, but bail if the close-based stop trips first.
    stop_i, stop_any = first_true(fwd <= STOP)
    use = np.where(stop_any & (stop_i < prof_i), stop_i, prof_i)
    out[f"first green, {STOP:.0%} stop"] = (np.take_along_axis(fwd, use[None], 0)[0],
                                             use.astype(float) + 1)
    return out


def run(label, sel, EX, bench, dates, control=None):
    """
    sel is a (T, S) boolean matrix of entry signals.

    Mean is reported next to median because the reversion exits are
    path-dependent: "sell on the first green close" lets winners out in a day
    with a small gain and holds losers to the 10-session cap, so its median
    is positive almost by construction while the mean carries the losses.
    `control` is the same exit rule run on the WHOLE universe -- any number
    that does not beat its control is the exit rule talking, not the signal.
    """
    print(f"\n  {label}   ({sel.sum(axis=1).mean():.0f} names/day)")
    print(f"    {'exit rule':<24}{'med':>8}{'mean':>8}{'vs ctrl':>9}"
          f"{'NW t':>7}{'p':>7}{'hold':>6}{'net':>8}{'n_ind':>6}")
    out = {}
    for name, (ret, hold) in EX.items():
        exc = (ret - bench_at(bench, hold)) * 100
        per_day, mean_day, keep_hold = [], [], []
        for i, d in enumerate(dates):
            row = sel[i]
            if row.sum() < MIN_NAMES:
                continue
            v = exc[i][row]
            h = hold[i][row]
            ok = np.isfinite(v)
            v, h = v[ok], h[ok]
            if v.size < MIN_NAMES:
                continue
            per_day.append((d, np.median(v)))
            mean_day.append(np.mean(v))
            keep_hold.append(np.nanmedian(h))
        if len(per_day) < 15:
            print(f"    {name:<24}{'too few entry days':>8}")
            continue
        ser = pd.Series([v for _, v in per_day],
                        index=[d for d, _ in per_day]).sort_index()
        mh = float(np.nanmedian(keep_hold))
        med, t, p = nw_p(ser, max(int(round(mh)), 1))
        mean = float(np.mean(mean_day))
        out[name] = med
        vs = "" if control is None else f"{med - control.get(name, np.nan):>+8.2f}%"
        net = med - COST
        print(f"    {name:<24}{med:>+7.2f}%{mean:>+7.2f}%{vs:>9}"
              f"{t:>7.2f}{p:>7.3f}{mh:>6.1f}{net:>+7.2f}%"
              f"{max(len(ser) // max(int(round(mh)), 1), 1):>6}")
    return out


def bench_at(bench, hold):
    """Benchmark return over each cell's own holding length."""
    idx = np.clip(hold.astype(int) - 1, 0, MAXH - 1)
    return np.take_along_axis(bench, idx[None], 0)[0]


def main() -> None:
    s, P, R, U, M20, M5, F52, fwd, rsi_f, bench = build()
    dates = P.index.to_numpy()
    EX = exits(fwd, rsi_f)
    Uv = U.values
    L = "=" * 86

    print(f"\n{L}\nMEAN REVERSION, 20-DAY FORMATION, REVERSION EXITS\n{L}")
    print(f"window   {pd.Timestamp(dates[0]):%d %b %Y} -> "
          f"{pd.Timestamp(dates[-1]):%d %b %Y}   ({len(dates)} sessions)")
    print(f"universe {Uv.sum(axis=1).mean():.0f} liquid names/session")
    print(f"benchmark: universe median cumulative return over the same dates")
    print(f"cost {COST:.2f}% round trip, exits capped at {MAXH} sessions")

    # --- the control that decides whether any of this means anything ----
    print(f"\n{L}\n0. CONTROL — no selection at all, every liquid name\n{L}")
    print("  Each exit rule applied to the whole universe. This is the number")
    print("  every leg below has to BEAT. A rule that scores here is an")
    print("  artifact of the exit, not a signal.")
    ctrl = run("ALL liquid names", Uv, EX, bench, dates)

    # --- formation decile sweep, 20-day ---------------------------------
    rank20 = M20.where(U).rank(axis=1, pct=True).values
    print(f"\n{L}\n1. HOW OVERSOLD? (20-day formation, decile of past return)\n{L}")
    for lo, hi, lab in [(0.00, 0.02, "worst 2%  (extreme)"),
                        (0.00, 0.05, "worst 5%"),
                        (0.00, 0.10, "worst decile"),
                        (0.10, 0.20, "2nd decile"),
                        (0.00, 0.30, "worst 30%  (mild)")]:
        sel = Uv & (rank20 > lo) & (rank20 <= hi)
        run(lab, sel, EX, bench, dates, ctrl)

    # --- 5-day formation, for contrast ----------------------------------
    rank5 = M5.where(U).rank(axis=1, pct=True).values
    print(f"\n{L}\n2. SAME, BUT 5-DAY FORMATION (what I built last time)\n{L}")
    run("worst decile, 5d", Uv & (rank5 <= 0.10), EX, bench, dates, ctrl)

    # --- does 'structurally intact' help? -------------------------------
    print(f"\n{L}\n3. DOES THE 'STILL NEAR 52W HIGH' FILTER ADD ANYTHING?\n{L}")
    base = Uv & (rank20 <= 0.10)
    intact = base & (F52.values >= -0.35)
    broken = base & (F52.values < -0.35)
    run("worst decile, intact", intact, EX, bench, dates, ctrl)
    run("worst decile, broken", broken, EX, bench, dates, ctrl)

    # --- RSI severity ----------------------------------------------------
    print(f"\n{L}\n4. RSI SEVERITY (worst 20-day decile, plus an RSI gate)\n{L}")
    for cap in (30, 40, 50):
        run(f"RSI <= {cap}", base & (R.values <= cap), EX, bench, dates, ctrl)

    # --- stability, on the cells that actually showed something ---------
    def series(sel, rule):
        ret, hold = EX[rule]
        exc = (ret - bench_at(bench, hold)) * 100
        rows = []
        for i, d in enumerate(dates):
            row = sel[i]
            if row.sum() < MIN_NAMES:
                continue
            v = exc[i][row]
            v = v[np.isfinite(v)]
            if v.size >= MIN_NAMES:
                rows.append((d, np.median(v)))
        return pd.Series([v for _, v in rows],
                         index=[d for d, _ in rows]).sort_index()

    print(f"\n{L}\n5. REGIME CHECK — only the extreme tail, fixed holds\n{L}")
    print("  Fixed holds only: the reversion exits score on the control too,")
    print("  so there is nothing of theirs to check.")
    cells = {"worst 2%": Uv & (rank20 <= 0.02), "worst 5%": Uv & (rank20 <= 0.05),
             "worst decile": Uv & (rank20 <= 0.10)}
    print(f"\n    {'cell':<16}{'rule':<12}{'1st half':>10}{'2nd half':>10}{'agree':>8}")
    for lab, sel in cells.items():
        for rule in ("fixed 7d", "fixed 10d"):
            ser = series(sel, rule)
            if len(ser) < 30:
                continue
            cut = ser.index[len(ser) // 2]
            a, b = ser[ser.index < cut], ser[ser.index >= cut]
            ag = "yes" if np.sign(a.median()) == np.sign(b.median()) else "NO"
            print(f"    {lab:<16}{rule:<12}{a.median():>+9.2f}%"
                  f"{b.median():>+9.2f}%{ag:>8}")

    # --- is the extreme tail just microcaps again? ----------------------
    print(f"\n{L}\n6. WHAT THE EXTREME TAIL IS MADE OF\n{L}")
    print("  The last round's only 'edge' was the illiquidity premium in")
    print("  disguise. If this tail is the same microcaps, it is untradable")
    print("  for the same reason.")
    TURN = s.pivot_table(index="date", columns="symbol",
                         values="med_turn60").reindex(
                             index=P.index, columns=P.columns)
    print(f"\n    {'':<26}{'worst 2%':>11}{'worst 5%':>11}{'universe':>11}")
    for grid, lab, mult in [(TURN, "turnover (Rs lakh)", 1),
                            (P, "price (Rs)", 1),
                            (F52, "off 52w high %", 100)]:
        w2 = np.nanmedian(grid.values[Uv & (rank20 <= 0.02)]) * mult
        w5 = np.nanmedian(grid.values[Uv & (rank20 <= 0.05)]) * mult
        un = np.nanmedian(grid.values[Uv]) * mult
        print(f"    {lab:<26}{w2:>11.1f}{w5:>11.1f}{un:>11.1f}")
    thin = 100 * np.nanmean(TURN.values[Uv & (rank20 <= 0.02)] < 500)
    print(f"\n    share of worst-2% picks under Rs 500 lakh/day: {thin:.0f}%")


if __name__ == "__main__":
    main()
