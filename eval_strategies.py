#!/usr/bin/env python3
"""
Every swing strategy worth testing, measured on one ruler.

The point of this file is comparability. Each strategy elsewhere in this repo
was validated against a slightly different universe, benchmark and statistic,
which makes "coils beat movers" an unanswerable claim. Here every candidate
is reduced to the same thing -- a boolean (dates x symbols) selection matrix
-- and run through identical machinery.

THE RULER
  universe    liquid (60-day median turnover >= Rs 100 lakh), price >= Rs 20,
              past its own 200-EMA and 252-day warmup
  benchmark   the universe's MEDIAN k-session forward return, measured on the
              entry date. Not the compounded daily median: median(product) !=
              product(median), and that mistake hands every strategy a free
              +0.6% at 5 sessions. The control row must read 0.00% or the
              benchmark is broken -- that is what it is there for.
  exits       fixed 5/7/10 session holds ONLY. Path-dependent exits such as
              "sell on the first green close" score +0.88% on a no-selection
              control, because winners leave in a day and losers run to the
              cap. They measure the rule, not the signal.
  statistic   median excess per entry day, then Newey-West at lag = horizon.
              A k-session window sampled daily overlaps its neighbour by k-1,
              so the naive t-stat overstates by 1.5-2.8x at these horizons.
  costs       0.30% round trip deducted (brokerage + STT + impact).
  honesty     mean is printed beside median because these distributions are
              badly right-skewed; halves must agree in sign; n_ind reports how
              many genuinely independent periods the cache holds, which at 10
              sessions is about 11.

Usage:
    python eval_strategies.py
    python eval_strategies.py --csv strategies.csv
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import eval_reversal as ev
import fetch
import panel as pnl
import reversal as rv
import stocks as st
import tom as tomscan

HORIZONS = (5, 7, 10)
COST = 0.30
MIN_NAMES = 5
MFE_TARGET = 0.03
MFE_WINDOW = 7


# ---------------------------------------------------------------- grids
def rsi_n(P: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder RSI at an arbitrary period, column-wise over a wide frame."""
    d = P.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    ad = (-d).clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))


def load() -> dict:
    paths = sorted(fetch.RAW_DIR.glob("bhav_*"))
    print(f"Loading {len(paths)} sessions "
          f"({paths[0].stem[5:]} -> {paths[-1].stem[5:]})...")
    raw = pd.concat([fetch._read_cache(p) for p in paths], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    smap = (pd.read_json(fetch.CACHE_DIR / "sector_map.json", orient="index")
              .rename_axis("symbol").reset_index())
    print("Building panel and indicators...")
    s, _ = pnl.build(raw.sort_values(["symbol", "date"]), smap)
    s = st.add_indicators(s)
    s = rv.add_reversal_features(s)          # mom5/mom20/from_52w_high/med_turn60
    s["adj_open"] = s["open"] * (s["adj"] / s["close"])

    g = s.groupby("symbol", sort=False)
    for n in (60, 120, 250):
        s[f"mom{n}"] = g["adj"].transform(lambda x, n=n: x.pct_change(n))
    # 12-1 momentum: the documented factor skips the most recent month.
    s["mom250_ex20"] = (1 + s["mom250"]) / (1 + s["mom20"]).replace(0, np.nan) - 1
    s["_uni"] = ((s["adj"] >= 20.0) & (s["med_turn60"].fillna(0) >= 100.0)
                 & s["ema200"].notna() & s["from_52w_high"].notna()).fillna(False)

    print("Pivoting to grids...")
    G = {}

    def piv(col, like=None):
        w = s.pivot_table(index="date", columns="symbol", values=col).sort_index()
        return w if like is None else w.reindex(index=like.index, columns=like.columns)

    G["P"] = piv("adj")
    like = G["P"]
    for c in ["adj_high", "adj_low", "adj_open", "rsi", "atr_pct", "vol_ratio",
              "ext_ema20", "pos_hi", "range20", "contraction", "cmf", "ema20",
              "ema50", "ema200", "turnover", "med_turn60", "deliv_pct",
              "mom5", "mom20", "mom60", "mom120", "mom250", "mom250_ex20",
              "from_52w_high", "trigger"]:
        if c in s.columns:
            G[c] = piv(c, like)
    G["U"] = piv("_uni", like).fillna(0) > 0

    # Derived grids the strategies need.
    G["rsi2"] = rsi_n(like, 2)
    rng = G["adj_high"] - G["adj_low"]
    G["nr7"] = rng.le(rng.rolling(7, min_periods=7).min())
    sd20 = like.rolling(20, min_periods=20).std()
    G["bbw"] = sd20 / like.rolling(20, min_periods=20).mean()
    G["deliv20"] = G["deliv_pct"].rolling(20, min_periods=10).mean() \
        if "deliv_pct" in G else None
    G["gap"] = G["adj_open"] / like.shift(1) - 1
    G["ret1"] = like.pct_change()

    # Sector-relative 60-day strength.
    sec = s.groupby("symbol")["sector"].last()
    sec = sec.reindex(like.columns)
    m60 = G["mom60"]
    sec_med = m60.T.groupby(sec).transform("median").T
    G["rs_sector"] = m60 - sec_med

    # Forward returns and the benchmark, per horizon.
    G["fwd"], G["bench"] = {}, {}
    Uv = G["U"].values
    for h in set(HORIZONS) | {MFE_WINDOW}:
        f = (like.shift(-h) / like - 1).values
        G["fwd"][h] = f
        G["bench"][h] = np.nanmedian(np.where(Uv, f, np.nan), axis=1, keepdims=True)

    # MFE over MFE_WINDOW sessions, for the "did it get there" column.
    hi = np.stack([G["adj_high"].shift(-j).values for j in range(1, MFE_WINDOW + 1)])
    G["mfe"] = np.nanmax(hi, axis=0) / like.values - 1
    return G, s, like


# ---------------------------------------------------------- strategies
def strategies(G, s, like) -> dict:
    """
    {name -> (selection matrix, one-line thesis)}.

    Every mask is ANDed with the universe. Ranks are cross-sectional within
    the universe on each date, so a "top decile" is a decile of tradable
    names rather than of the whole exchange.
    """
    U = G["U"]
    Uv = U.values

    def rank(col):
        return G[col].where(U).rank(axis=1, pct=True).values

    def r(col):
        return G[col].values

    up = (r("P") > r("ema50")) & (r("ema50") > r("ema200"))
    S = {}

    # --- mean reversion family -------------------------------------
    S["MR extreme (worst 2% 20d)"] = (
        Uv & (rank("mom20") <= 0.02),
        "Deeply oversold on a month. Our finding.")
    S["MR worst 5% 20d"] = (
        Uv & (rank("mom20") <= 0.05),
        "Same, one notch wider.")
    S["MR RSI(2) < 5"] = (
        Uv & (r("rsi2") < 5),
        "Connors short-term oversold.")
    S["MR 1-day crash > 6%"] = (
        Uv & (r("ret1") <= -0.06),
        "Single-session capitulation.")
    S["MR gap down > 4%"] = (
        Uv & (r("gap") <= -0.04),
        "Opens far below yesterday's close.")
    S["MR worst decile + uptrend"] = (
        Uv & (rank("mom20") <= 0.10) & up,
        "Oversold but structurally strong: buy the dip.")

    # --- momentum family --------------------------------------------
    S["Mom 12-1 top decile"] = (
        Uv & (rank("mom250_ex20") >= 0.90),
        "Classic 12-month momentum, skipping last month.")
    S["Mom 6m top decile"] = (
        Uv & (rank("mom120") >= 0.90),
        "Six-month winners.")
    S["Mom 12-1 + uptrend"] = (
        Uv & (rank("mom250_ex20") >= 0.90) & up,
        "Momentum confirmed by the moving averages.")
    # Severity sweep, same monotonicity test the MR finding had to pass: a
    # real factor should decay smoothly as the cut loosens.
    S["Mom 12-1 top 5% + uptrend"] = (
        Uv & (rank("mom250_ex20") >= 0.95) & up,
        "Stricter momentum cut.")
    S["Mom 12-1 top 30% + uptrend"] = (
        Uv & (rank("mom250_ex20") >= 0.70) & up,
        "Looser momentum cut.")
    S["Mom 12-1 + uptrend + pullback"] = (
        Uv & (rank("mom250_ex20") >= 0.90) & up
        & (np.abs(r("ext_ema20")) <= 0.03),
        "Momentum leader resting on its 20-EMA.")

    # --- breakout / high proximity ----------------------------------
    S["Near 52w high (<2%)"] = (
        Uv & (r("from_52w_high") >= -0.02),
        "Pressing the yearly high.")
    S["Breakout + volume"] = (
        Uv & (r("from_52w_high") >= -0.02) & (r("vol_ratio") >= 1.5),
        "New high on expanding volume.")
    S["Breakout + volume (within 5%)"] = (
        Uv & (r("from_52w_high") >= -0.05) & (r("vol_ratio") >= 1.5),
        "Same, loosened to get a usable list size.")

    # --- volatility contraction -------------------------------------
    S["Squeeze NR7 + uptrend"] = (
        Uv & G["nr7"].fillna(False).values & up,
        "Narrowest range in 7 sessions, in an uptrend.")
    S["Squeeze BB width low decile"] = (
        Uv & (rank("bbw") <= 0.10) & up,
        "Bollinger bands at their tightest.")

    # --- pullback ----------------------------------------------------
    S["Pullback to EMA20 + dry volume"] = (
        Uv & up & (np.abs(r("ext_ema20")) <= 0.02) & (r("vol_ratio") <= 0.8),
        "Uptrend resting on the 20-EMA on light volume.")

    # --- India-specific: delivery -----------------------------------
    if G.get("deliv20") is not None:
        S["Delivery accumulation"] = (
            Uv & (rank("deliv20") >= 0.90) & (np.abs(r("mom20")) <= 0.03),
            "Heavy delivery while price goes nowhere.")

    # --- relative strength, low vol ---------------------------------
    S["Sector rel-strength top decile"] = (
        Uv & (rank("rs_sector") >= 0.90),
        "Beating its own sector over 60 sessions.")
    S["Low volatility decile"] = (
        Uv & (rank("atr_pct") <= 0.10),
        "The low-volatility anomaly.")

    # --- our shipped scans, as baselines ----------------------------
    cp = st.CoilParams()
    coil = (Uv & up
            & G["pos_hi"].values.__ge__(cp.near_high_min)
            & G["pos_hi"].values.__le__(cp.near_high_max)
            & (r("rsi") >= cp.rsi_low) & (r("rsi") <= cp.rsi_high)
            & (np.abs(r("ext_ema20")) <= cp.max_ext_ema20)
            & (r("vol_ratio") <= cp.vol_dryup_max)
            & (r("range20") <= cp.range_max))
    S["OURS: Coiled Bases"] = (np.nan_to_num(coil, nan=0).astype(bool),
                               "Shipped coil gates.")

    mom_mask = tomscan._gate_mask(s).reindex(s.index).fillna(False)
    s2 = s.assign(_m=mom_mask.values)
    mg = s2.pivot_table(index="date", columns="symbol", values="_m") \
           .reindex(index=like.index, columns=like.columns).fillna(0) > 0
    S["OURS: Expected Movers"] = (Uv & mg.values, "Shipped momentum gates.")

    return S


# ------------------------------------------------------------- scoring
def measure(sel, G, dates, h) -> dict:
    fwd, bench = G["fwd"][h], G["bench"][h]
    exc = (fwd - bench) * 100
    ok = np.isfinite(exc)
    per_day, means = [], []
    for i in range(len(dates)):
        row = sel[i] & ok[i]
        if row.sum() < MIN_NAMES:
            continue
        v = exc[i][row]
        per_day.append((dates[i], np.median(v)))
        means.append(v.mean())
    if len(per_day) < 15:
        return {}
    ser = pd.Series([v for _, v in per_day],
                    index=[d for d, _ in per_day]).sort_index()
    t, infl = ev.nw_t(ser, h)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if np.isfinite(t) else np.nan
    cut = ser.index[len(ser) // 2]
    a, b = ser[ser.index < cut], ser[ser.index >= cut]
    return {"med": ser.median(), "mean": float(np.mean(means)), "t": t, "p": p,
            "net": ser.median() - COST, "days": len(ser), "infl": infl,
            "n_ind": max(len(ser) // h, 1),
            "h1": a.median(), "h2": b.median(),
            "agree": np.sign(a.median()) == np.sign(b.median())}


def mfe_lift(sel, G, dates) -> float:
    mfe, Uv = G["mfe"], G["U"].values
    ok = np.isfinite(mfe)
    diffs = []
    for i in range(len(dates)):
        srow, urow = sel[i] & ok[i], Uv[i] & ok[i]
        if srow.sum() < MIN_NAMES or urow.sum() < 20:
            continue
        diffs.append((mfe[i][srow] >= MFE_TARGET).mean()
                     - (mfe[i][urow] >= MFE_TARGET).mean())
    return 100 * float(np.mean(diffs)) if len(diffs) >= 15 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    args = ap.parse_args()

    G, s, like = load()
    dates = like.index.to_numpy()
    S = strategies(G, s, like)
    Uv = G["U"].values
    L = "=" * 110

    # A COMMON WINDOW is mandatory, not a nicety. Warmups differ enormously:
    # 12-1 momentum needs 250 sessions of history before it exists at all,
    # the coil gates need 200, mean reversion needs 120. Left alone, each
    # strategy gets scored on a different slice of market history and the
    # table compares periods rather than strategies. Everything below is
    # restricted to the sessions on which the hungriest strategy is defined.
    warm = (G["mom250_ex20"].where(G["U"]).notna().sum(axis=1) >= 20).values
    have_fwd = np.isfinite(G["fwd"][max(HORIZONS)]).any(axis=1)
    common = warm & have_fwd
    cdates = dates[common]
    print(f"\nCommon measurable window: {pd.Timestamp(cdates[0]):%d %b %Y} -> "
          f"{pd.Timestamp(cdates[-1]):%d %b %Y}  ({common.sum()} sessions)")
    print("Set by 12-1 momentum's 250-session warmup. Every strategy below is")
    print("scored on exactly these days so the comparison is like-for-like.")

    def restrict(sel):
        out = sel.copy()
        out[~common] = False
        return out

    S_full = S
    S = {k: (restrict(v[0]), v[1]) for k, v in S.items()}
    Uv_c = restrict(Uv)

    # Control first. If this is not 0.00% the benchmark is wrong and every
    # number below it is meaningless.
    print(f"\n{L}\nCONTROL — no selection, every liquid name\n{L}")
    for h in HORIZONS:
        c = measure(Uv_c, G, dates, h)
        print(f"  {h:>2} sessions   median excess {c['med']:+.2f}%   "
              f"(must be 0.00%)   mean {c['mean']:+.2f}%")

    rows = []
    for name, (sel, thesis) in S.items():
        # Averaged over the common window ONLY. Averaging across all 324
        # dates divides by the ~260 sessions inside the strategy's warmup
        # where it selects nothing, understating a live list by 5x for
        # anything needing 250 sessions of history.
        rec = {"strategy": name, "thesis": thesis,
               "names_day": float(sel.sum(axis=1)[common].mean()),
               "turnover": float(np.nanmedian(G["med_turn60"].values[sel]))
               if sel.any() else np.nan,
               "mfe_lift": mfe_lift(sel, G, dates)}
        for h in HORIZONS:
            m = measure(sel, G, dates, h)
            for k, v in m.items():
                rec[f"{k}_{h}"] = v
        # Same strategy on every session it is defined on, not just the
        # common window. Not like-for-like across strategies, but it is the
        # best available read on the ones that do not need 250 sessions of
        # warmup -- mean reversion gets 118 days here against 64.
        own = measure(S_full[name][0], G, dates, 7)
        rec["own_net_7"] = own.get("med", np.nan) - COST if own else np.nan
        rec["own_t_7"] = own.get("t", np.nan) if own else np.nan
        rec["own_days"] = own.get("days", np.nan) if own else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)

    print(f"\n{L}\nCOMPARISON — median excess return over the universe, "
          f"net of {COST:.2f}% costs\n{L}")
    print("'days' is how many of the %d common sessions had enough names to"
          % common.sum())
    print("measure. A thin strategy is only scored on the days it happens to")
    print("fire, and those days are not a random sample -- low coverage is a")
    print("warning, flagged with !\n")
    print(f"{'strategy':<34}{'n/day':>6}{'days':>5}{'turn':>6}"
          f"{'5d':>7}{'7d':>7}{'10d':>7}{'t':>6}{'p':>7}{'halves':>7}"
          f"{'MFE':>7}{'own 7d':>9}{'(days)':>8}")
    print("-" * 110)
    df = df[df["days_7"].notna()]
    for _, x in df.sort_values("net_7", ascending=False).iterrows():
        ag = "agree" if x.get("agree_7") else "NO"
        cov = x.get("days_7", np.nan)
        warn = "!" if np.isfinite(cov) and cov < 0.6 * common.sum() else " "
        print(f"{x['strategy']:<34}{x['names_day']:>6.0f}{cov:>5.0f}"
              f"{x['turnover']:>6.0f}"
              f"{x.get('net_5', np.nan):>+6.2f}%{x.get('net_7', np.nan):>+6.2f}%"
              f"{x.get('net_10', np.nan):>+6.2f}%"
              f"{x.get('t_7', np.nan):>6.2f}{x.get('p_7', np.nan):>7.3f}"
              f"{ag:>7}{x['mfe_lift']:>+6.1f}p"
              f"{x['own_net_7']:>+8.2f}%{x['own_days']:>7.0f}{warn:>2}")

    print(f"\n{L}\nSURVIVORS — net positive at 7d, p<0.05, halves agree\n{L}")
    keep = df[(df["net_7"] > 0) & (df["p_7"] < 0.05) & (df["agree_7"] == True)]
    if keep.empty:
        print("  none")
    else:
        for _, x in keep.sort_values("net_7", ascending=False).iterrows():
            print(f"\n  {x['strategy']}   —   {x['thesis']}")
            print(f"    {x['names_day']:.0f} names/session, median turnover "
                  f"Rs {x['turnover']:,.0f} lakh")
            print(f"    7d: median {x['med_7']:+.2f}% (mean {x['mean_7']:+.2f}%), "
                  f"net {x['net_7']:+.2f}%, t={x['t_7']:.2f}, p={x['p_7']:.3f}, "
                  f"{x['n_ind_7']:.0f} independent periods, "
                  f"{x['days_7']:.0f}/{len(dates)} days covered")
            print(f"    halves {x['h1_7']:+.2f}% / {x['h2_7']:+.2f}%, "
                  f"MFE +3% lift {x['mfe_lift']:+.1f}pp")

    # --- the test that decides usability -------------------------------
    print(f"\n{L}\nHOLDOUT — pick on the first half, judge on the second\n{L}")
    print("  Every number above is in-sample: the thresholds were chosen by")
    print("  sweeping this same data. Here the ranking is built on the first")
    print("  half only, then the winners are scored on the second half they")
    print("  never saw. A strategy that survives this is worth trading.")
    # Split the COMMON window in half, not the raw calendar -- the calendar's
    # first half is entirely inside the warmup and contains no signals at all.
    mid = cdates[len(cdates) // 2]
    first = common & (dates < mid)
    second = common & (dates >= mid)
    print(f"  split at {pd.Timestamp(mid):%d %b %Y}: "
          f"{first.sum()} sessions then {second.sum()}")

    def half_measure(sel, mask, h):
        g = {"fwd": {h: np.where(mask[:, None], G["fwd"][h], np.nan)},
             "bench": G["bench"], "U": G["U"], "mfe": G["mfe"],
             "med_turn60": G["med_turn60"]}
        return measure(sel, g, dates, h)

    ranked = []
    for name, (sel, _) in S.items():
        a = half_measure(sel, first, 7)
        b = half_measure(sel, second, 7)
        if a and b:
            ranked.append((name, a["med"] - COST, b["med"] - COST, b["t"], b["p"]))
    ranked.sort(key=lambda x: -x[1])
    print(f"\n  {'strategy':<34}{'1st half net':>14}{'2nd half net':>14}"
          f"{'2nd t':>8}{'2nd p':>8}{'held up':>9}")
    print("  " + "-" * 87)
    for name, a, b, t, p in ranked[:10]:
        held = "yes" if (a > 0 and b > 0) else "no"
        print(f"  {name:<34}{a:>+13.2f}%{b:>+13.2f}%{t:>8.2f}{p:>8.3f}{held:>9}")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nWritten to {args.csv}")


if __name__ == "__main__":
    main()
