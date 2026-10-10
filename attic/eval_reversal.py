#!/usr/bin/env python3
"""
Does the reversal scan predict returns, where Expected Movers did not?

The catch study found 75% of next-session winners were not in an uptrend, so
Expected Movers cannot reach them. reversal.py targets that population. This
decides whether it is worth building, before any UI work.

The bar is set by what killed the momentum score, so the same tests apply:

  1. MEDIAN, NOT MEAN   — mean excess is dominated by a few huge winners.
                          Every headline here is a median.
  2. DAY-PAIRED         — a scan's median is compared against the SAME
                          session's universe median, then t-tested across
                          days. Pooling rows treats one hot session as many
                          independent wins, which is how the momentum score
                          first looked significant.
  3. VS ITS OWN POOL    — the score must beat the gate set it draws from.
                          The momentum score failed exactly here: its top 40
                          sat BELOW its own gated pool at every horizon.
  4. STABILITY          — first half vs second half. An edge in one half only
                          is a regime, not a signal.
  5. COSTS              — a 1-session round trip costs roughly 0.20-0.40% in
                          India. Gross edge under that is not tradable.

  6. OVERLAP            — a 20-day forward return sampled daily shares 19/20
                          of its window with the next one, so 125 scan days
                          are not 125 observations. Newey-West widens the
                          error bar accordingly (it inflated the movers leg
                          2.8x), and `n_ind` reports how many genuinely
                          independent periods the cache actually holds.

SUPERSEDED IN PART -- read eval_meanrev.py first.
This script tested a 5-day formation with a 2% drop threshold on a fixed
holding clock. All three were the wrong choice. eval_meanrev.py redoes it
with 20-day formation, a severity sweep and proper exits, and DOES find an
edge in the extreme tail (worst 2% of 20-day performers, +0.58% at a
7-session hold, p=0.003, net of costs, and liquid). The verdict below stands
only for the mild-threshold construction tested here.

VERDICT (324 sessions, Jun 2025 - Sep 2026): DO NOT BUILD *THIS* VERSION.
The reversal premise has no edge -- the gated pool is flat at every horizon
(1-session +0.08%, t=0.61) and rev_score ranks BELOW its own pool at 5 and
20 sessions, the identical failure to the momentum score. Inside the pool
mom5's IC is -0.010 (t=-0.88): among already-oversold names, more oversold
is not better.

The one variant that did work -- rank by high ATR and low turnover, which
the in-pool ICs pointed at -- is not a reversal signal at all. It carries no
reversal term, works about as well with the gates removed (14% name
overlap), only appears at 10-20 sessions, and buys microcaps: median
turnover Rs 186 lakh/day, 88% of picks under Rs 500 lakh. It is the size and
illiquidity premium, and at 20 sessions the cache holds 6 independent
periods (non-overlapping t=1.14). Not enough to build on.

Usage:
    python eval_reversal.py
    python eval_reversal.py --top 40 --dump rev.csv
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import fetch
import panel as pnl
import reversal as rv
import stocks as st
import tom as tomscan

HORIZONS = [1, 5, 10, 20]
ROUND_TRIP_COST = 0.30  # %, one buy + one sell, brokerage + impact + STT

# Scan-day features ranked against the outcome inside the reversal pool.
COMPONENTS = ["mom5", "mom20", "from_52w_high", "rsi", "atr_pct",
              "deliv20", "med_turn60", "down_streak", "rev_score"]


def load_all() -> pd.DataFrame:
    paths = sorted(fetch.RAW_DIR.glob("bhav_*"))
    print(f"Loading {len(paths)} cached sessions "
          f"({paths[0].stem[5:]} -> {paths[-1].stem[5:]})...")
    raw = pd.concat([fetch._read_cache(p) for p in paths], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["symbol", "date"])
    smap = (pd.read_json(fetch.CACHE_DIR / "sector_map.json", orient="index")
              .rename_axis("symbol").reset_index())
    print("Building panel...")
    stocks, _ = pnl.build(raw, smap)
    print("Computing indicators...")
    stocks = st.add_indicators(stocks)
    print("Computing reversal features...")
    return rv.add_reversal_features(stocks)


def add_forwards(s: pd.DataFrame) -> pd.DataFrame:
    """Forward excess return at each horizon, over the same session's median."""
    s = s.sort_values(["symbol", "date"]).copy()
    g = s.groupby("symbol", sort=False)
    for h in HORIZONS:
        fwd = (g["adj"].shift(-h) / s["adj"] - 1.0) * 100
        s[f"x{h}"] = fwd - fwd.groupby(s["date"]).transform("median")
    return s


def nw_t(x: pd.Series, lag: int) -> tuple[float, float]:
    """
    Newey-West t-stat for the mean of a series of overlapping windows.

    Bartlett weights out to `lag` scale the variance up by the autocorrelation
    the overlap induces. Also returns how much wider that makes the error bar
    than the naive one, which is the honest measure of how much the daily
    sampling was flattering the result.
    """
    x = x.dropna().astype(float)
    n = len(x)
    if n < 10:
        return float("nan"), float("nan")
    d = x - x.mean()
    var = float((d @ d) / n)
    for k in range(1, min(lag, n - 1) + 1):
        gk = float((d.iloc[k:] @ d.iloc[:-k].values) / n)
        var += 2 * (1 - k / (lag + 1)) * gk
    se_nw = math.sqrt(max(var, 1e-12) / n)
    se_naive = x.std(ddof=1) / math.sqrt(n)
    return x.mean() / se_nw, se_nw / se_naive


def paired(sel: pd.DataFrame, uni: pd.DataFrame, h: int) -> dict:
    """
    Per-day median of the selection MINUS the same day's liquid universe,
    t-tested across days.

    The x{h} columns are excess over the all-stock median, which includes
    illiquid names the scans can never buy — liquid stock beats that median
    on its own, so every leg would be flattered by the same free lift.
    Differencing against the universe day by day removes it and makes the
    baseline exactly zero. The t-stat is over days, which is the unit that
    repeats; pooling rows would treat one hot session as many wins.
    """
    col = f"x{h}"
    base = uni.dropna(subset=[col]).groupby("date")[col].median()
    by_day = sel.dropna(subset=[col]).groupby("date")[col].median()
    by_day = (by_day - base).dropna()
    by_day = by_day.sort_index()
    n = len(by_day)
    blank = {"n_days": n, "med": np.nan, "t": np.nan, "p": np.nan,
             "win": np.nan, "infl": np.nan, "n_ind": 0}
    if n < 10:
        return blank
    t, infl = nw_t(by_day, h)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if np.isfinite(t) else np.nan
    return {"n_days": n, "med": by_day.median(), "t": t, "p": p,
            "win": 100 * (by_day > 0).mean(), "infl": infl,
            "n_ind": max(n // max(h, 1), 1)}


def report(label: str, sel: pd.DataFrame, uni: pd.DataFrame, rows_per_day: float) -> None:
    print(f"\n{label}   ({rows_per_day:.0f} names/session)")
    print(f"  {'horizon':<10}{'med excess':>12}{'NW t':>7}{'p':>8}{'infl':>7}"
          f"{'n_ind':>7}{'net of cost':>13}")
    for h in HORIZONS:
        r = paired(sel, uni, h)
        if not np.isfinite(r["med"]):
            print(f"  {str(h) + ' sess':<10}{'baseline':>12}")
            continue
        net = r["med"] - ROUND_TRIP_COST
        flag = "" if net > 0 else "  <- eaten by costs"
        print(f"  {str(h) + ' sess':<10}{r['med']:>+11.2f}%{r['t']:>7.2f}"
              f"{r['p']:>8.3f}{r['infl']:>6.1f}x{r['n_ind']:>7}{net:>+12.2f}%{flag}")


def halves(label: str, sel: pd.DataFrame, uni: pd.DataFrame, h: int) -> None:
    days = np.sort(sel["date"].unique())
    if len(days) < 40:
        print(f"  {label:<24}too few days to split")
        return
    cut = days[len(days) // 2]
    a, b = sel[sel["date"] < cut], sel[sel["date"] >= cut]
    ra, rb = paired(a, uni, h), paired(b, uni, h)
    agree = "yes" if (np.isfinite(ra["med"]) and np.isfinite(rb["med"])
                      and np.sign(ra["med"]) == np.sign(rb["med"])) else "NO"
    print(f"  {label:<24}{ra['med']:>+9.2f}%{rb['med']:>+9.2f}%{agree:>10}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40,
                    help="size of the shortlist cut to test")
    ap.add_argument("--dump", help="write the reconstructed pool to CSV")
    args = ap.parse_args()

    s = add_forwards(load_all())
    L = "=" * 82

    # Universe: the tradable baseline both scans are judged against.
    uni_mask = ((s["adj"] >= rv.REV_MIN_PRICE)
                & (s["med_turn60"].fillna(0) >= rv.REV_MIN_TURNOVER))
    uni = s[uni_mask.fillna(False)].copy()

    # Warmup: 52-week high needs 120+ sessions, ema200 needs 200.
    warm = s["ema200"].notna() & s["from_52w_high"].notna()
    uni = uni[warm.reindex(uni.index, fill_value=False)]
    days = np.sort(uni["date"].unique())

    print(f"\n{L}\nREVERSAL SCAN — does it predict returns?\n{L}")
    print(f"window        {pd.Timestamp(days[0]):%d %b %Y} -> "
          f"{pd.Timestamp(days[-1]):%d %b %Y}   ({len(days)} sessions)")
    print(f"universe      {len(uni) / len(days):.0f} liquid names/session")
    print("excess        vs the same session's all-stock median return")
    print(f"cost assumed  {ROUND_TRIP_COST:.2f}% round trip")

    # --- the reversal scan ----------------------------------------------
    rev_pool = uni[rv.gate_mask(uni).fillna(False)].copy()
    rev_pool["rev_score"] = rv.score(rev_pool)
    rev_top = (rev_pool.sort_values("rev_score", ascending=False)
               .groupby("date").head(args.top))

    # --- Expected Movers, same ruler, for reference ----------------------
    mom_pool = uni[tomscan._gate_mask(uni).fillna(False)].copy()
    mom_pool["score"] = np.nan
    mom_top = pd.DataFrame(columns=mom_pool.columns)
    if not mom_pool.empty and "vol_ratio" in mom_pool.columns:
        # Rank on the shipped weights so the comparison is the real list.
        mom_pool["score"] = (
            0.30 * (mom_pool["vol_ratio"].clip(0, 3) / 3)
            + 0.20 * (mom_pool["rsi"].fillna(50) / 100)
            + 0.18 * (mom_pool["atr_pct"].fillna(0).clip(0, 0.08) / 0.08)
            + 0.14 * (1 - mom_pool["ext_ema20"].abs().fillna(0).clip(0, 0.1) / 0.1)
            + 0.18 * ((mom_pool["cmf"].fillna(0) + 1) / 2)
        )
        mom_top = (mom_pool.sort_values("score", ascending=False)
                   .groupby("date").head(args.top))

    print(f"\n{L}\nHEAD TO HEAD\n{L}")
    print(f"\nUNIVERSE is the baseline: every leg below is differenced against"
          f"\nthe same session's liquid-universe median, so 0.00% means no edge.")
    report("REVERSAL  gated pool", rev_pool, uni, len(rev_pool) / len(days))
    report(f"REVERSAL  top {args.top} by rev_score", rev_top, uni,
           len(rev_top) / len(days))
    report("MOVERS    gated pool", mom_pool, uni, len(mom_pool) / len(days))
    if not mom_top.empty:
        report(f"MOVERS    top {args.top} by score", mom_top, uni,
               len(mom_top) / len(days))

    # --- does the score add anything over its own gates? -----------------
    print(f"\n{L}\nDOES rev_score BEAT ITS OWN POOL?\n{L}")
    print("  The momentum score failed here: its top 40 ranked BELOW the pool")
    print("  it was drawn from. If these deltas are negative, rank on nothing.")
    print(f"\n  {'horizon':<10}{'pool':>10}{'top ' + str(args.top):>10}{'delta':>10}")
    for h in HORIZONS:
        a, b = paired(rev_pool, uni, h), paired(rev_top, uni, h)
        d = b["med"] - a["med"]
        print(f"  {str(h) + ' sess':<10}{a['med']:>+9.2f}%{b['med']:>+9.2f}%"
              f"{d:>+9.2f}%{'  <- score subtracts' if d < 0 else ''}")

    # --- stability --------------------------------------------------------
    print(f"\n{L}\nSTABILITY — first half vs second half (5-session excess)\n{L}")
    print(f"  {'':<24}{'1st half':>10}{'2nd half':>9}{'same sign':>10}")
    halves("reversal pool", rev_pool, uni, 5)
    halves(f"reversal top {args.top}", rev_top, uni, 5)
    halves("movers pool", mom_pool, uni, 5)

    # --- which gate carries it -------------------------------------------
    print(f"\n{L}\nGATE DECOMPOSITION (5-session excess, one gate dropped)\n{L}")
    print("  A gate that pays should LOWER the number when removed.")
    print(f"\n  {'dropped gate':<24}{'med excess':>12}{'names/day':>12}{'verdict':>12}")
    full = paired(rev_pool, uni, 5)["med"]
    print(f"  {'none (full gate set)':<24}{full:>+11.2f}%"
          f"{len(rev_pool) / len(days):>12.0f}{'':>12}")
    gates = {
        "rsi <= 55": lambda d: d["rsi"].fillna(50) <= rv.REV_RSI_MAX,
        "5d drop <= -2%": lambda d: d["mom5"] <= rv.REV_MIN_DROP_5D,
        "within 35% of 52w hi": lambda d: d["from_52w_high"] >= rv.REV_MAX_FROM_52W,
        "not at 52w high": lambda d: d["from_52w_high"] <= rv.REV_MIN_FROM_52W,
    }
    for name, fn in gates.items():
        keep = pd.Series(True, index=uni.index)
        for other, ofn in gates.items():
            if other != name:
                keep &= ofn(uni).fillna(False)
        sub = uni[keep]
        r = paired(sub, uni, 5)
        delta = r["med"] - full
        verdict = "pays" if delta < -0.05 else ("hurts" if delta > 0.05 else "flat")
        print(f"  {name:<24}{r['med']:>+11.2f}%{len(sub) / len(days):>12.0f}"
              f"{verdict:>12}")

    # --- component ranking inside the pool -------------------------------
    print(f"\n{L}\nCOMPONENT IC inside the reversal pool (5-session)\n{L}")
    print("  Spearman per day, averaged. t over days. Sets the weights, or")
    print("  shows there is nothing to weight.")
    print(f"\n  {'feature':<18}{'IC':>9}{'t':>8}{'stable':>9}")
    for c in COMPONENTS:
        if c not in rev_pool.columns:
            continue
        ics = (rev_pool.dropna(subset=[c, "x5"]).groupby("date")
               .apply(lambda d: d[c].corr(d["x5"], method="spearman"),
                      include_groups=False)
               .dropna())
        if len(ics) < 20:
            continue
        t = ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics)))
        print(f"  {c:<18}{ics.mean():>+9.4f}{t:>8.2f}"
              f"{('yes' if abs(t) > 2 else 'no'):>9}")

    if args.dump:
        rev_pool.to_csv(args.dump, index=False)
        print(f"\nPool written to {args.dump}")


if __name__ == "__main__":
    main()
