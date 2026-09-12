#!/usr/bin/env python3
"""
Does the For-Tom momentum score actually predict anything?

backtest_tom.py reports hit rates for the shortlist but never compares them
against the base rate of the pool the shortlist is drawn from. A 25% hit rate
means nothing if 25% of every uptrend stock does the same thing. This script
measures lift instead:

  1. BASE RATE   — outcome for every uptrend/liquid stock (the pool).
  2. GATED       — outcome after the For-Tom gates (trend/zone/price/RSI).
  3. TOP N       — outcome for the rows the UI actually shows.
  4. DECILES     — outcome by score decile. If the score works, this is
                   monotonic. If it is flat, the weights are noise.
  5. COMPONENTS  — rank correlation of each raw feature with the outcome,
                   so weights can be set from evidence instead of taste.

Outcomes are measured several ways because "reached +3% intraday" assumes you
sold at the exact high. ret_cc (buy at today's close, sell at tomorrow's
close) is what the 15:20 workflow actually gets.

Usage:
    python eval_tom.py --days 120
    python eval_tom.py --days 120 --dump eval.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import fetch
import stocks as st
import tom as tomscan

# Raw scan-day features tested individually against the outcome.
COMPONENTS = [
    "vol_ratio", "rsi", "atr_pct", "ext_ema20", "cmf",
    "pos_hi", "to_trigger", "range20", "contraction", "base_days",
    "pchange", "turnover",
]

OUTCOMES = {
    "hi3": "reached +3% on next-day high (sell-at-high fiction)",
    "hi1": "reached +1% on next-day high",
    "ret_cc": "close -> next close  (the 15:20 workflow)",
    "ret_oc": "next open -> next close",
    "ret_lim": "close -> +2% limit next day, else next close",
    "ret_3d": "close -> close, 3 sessions",
    "ret_5d": "close -> close, 5 sessions",
    "ret_10d": "close -> close, 10 sessions",
}

# Columns reported in the summary tables, in order.
REPORT = ["hi3", "hi1", "ret_cc", "ret_oc", "ret_lim", "ret_3d", "ret_5d", "ret_10d"]
HORIZONS = {"ret_3d": 3, "ret_5d": 5, "ret_10d": 10}


def load(n_sessions: int) -> pd.DataFrame:
    """Read the newest `n_sessions` days straight from the bhavcopy cache.

    Deliberately does not use fetch.load_history: that opens an NSESession for
    any weekday missing from cache (every holiday), and NSE is blocked here.
    Evaluation must run offline.
    """
    paths = sorted(fetch.RAW_DIR.glob("bhav_*"))[-n_sessions:]
    if not paths:
        raise SystemExit(f"no cached bhavcopy in {fetch.RAW_DIR}")
    print(f"Loading {len(paths)} cached sessions "
          f"({paths[0].stem[5:]} -> {paths[-1].stem[5:]})...")
    raw = pd.concat([fetch._read_cache(p) for p in paths], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["symbol", "date"])

    smap_path = fetch.CACHE_DIR / "sector_map.json"
    if not smap_path.exists():
        raise SystemExit(f"no sector map at {smap_path}")
    smap = (pd.read_json(smap_path, orient="index")
              .rename_axis("symbol").reset_index())

    import panel as pnl
    print("Building panel...")
    stocks, _ = pnl.build(raw, smap)
    print("Computing indicators...")
    return st.add_indicators(stocks)


def eval_day(stocks: pd.DataFrame, all_dates: list, scan_date, liquid_min: float):
    """One scan day -> (gated rows with scores, base-pool rows). Both carry outcomes."""
    eod_hist = stocks[stocks["date"] <= scan_date]
    today = eod_hist[eod_hist["date"] == scan_date]
    if today.empty:
        return None, None

    nexts = [d for d in all_dates if d > scan_date]
    if not nexts:
        return None, None
    nxt = stocks[stocks["date"] == nexts[0]]
    if nxt.empty:
        return None, None
    nd = nxt.set_index("symbol")[["adj", "high", "open"]]

    # Forward closes for the multi-session horizons. Missing (too close to the
    # end of the data) becomes NaN and is skipped by the mean.
    fwd = {}
    for col, h in HORIZONS.items():
        if len(nexts) >= h:
            f = stocks[stocks["date"] == nexts[h - 1]]
            if not f.empty:
                fwd[col] = f.set_index("symbol")["adj"]

    # Median turnover over the trailing window, as the coil scan measures it.
    med_turn = (eod_hist.groupby("symbol")["turnover"].median()
                if "turnover" in eod_hist.columns else None)

    # `live` from the EOD close: what a 15:20 refresh approximately sees.
    live = today[["symbol"]].copy()
    live["ltp"] = today["adj"]
    live["volume"] = today["volume"]
    live["pchange"] = today["ret"] * 100 if "ret" in today.columns else 0.0
    live["high"] = today["high"] if "high" in today.columns else today["adj"]

    # top_n huge => every row that survives the gates, with its score.
    gated = tomscan.for_tomorrow_momentum(
        eod_hist, live, scan_rows=None, top_n=10 ** 9,
    )

    def outcomes(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        df = df[df["symbol"].isin(nd.index)].copy()
        if df.empty:
            return df
        px = pd.to_numeric(df[price_col], errors="coerce").to_numpy()
        idx = df["symbol"].to_numpy()
        nh = nd.loc[idx, "high"].to_numpy()
        nc = nd.loc[idx, "adj"].to_numpy()
        no = nd.loc[idx, "open"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            df["hi3"] = nh >= px * 1.03
            df["hi1"] = nh >= px * 1.01
            df["ret_cc"] = (nc / px - 1.0) * 100.0
            df["ret_oc"] = (nc / no - 1.0) * 100.0
            # Realistic exit: a +2% limit order fills if the high reaches it,
            # otherwise the position is closed at the next close.
            df["ret_lim"] = np.where(nh >= px * 1.02, 2.0, (nc / px - 1.0) * 100.0)
            for col, series in fwd.items():
                fc = series.reindex(idx).to_numpy()
                df[col] = (fc / px - 1.0) * 100.0
        df["scan_date"] = scan_date
        return df[np.isfinite(px) & (px > 0)]

    if not gated.empty and med_turn is not None:
        gated["turnover"] = gated["symbol"].map(med_turn)
    gated = outcomes(gated, "ltp") if not gated.empty else gated
    if not gated.empty:
        gated["rank"] = gated["score"].rank(ascending=False, method="first")

    # Base pool: the gates that define scope (uptrend, priced, liquid) but no
    # zone/RSI/score selection. This is what the shortlist must beat.
    pool = today.copy()
    if {"ema50", "ema200", "adj"}.issubset(pool.columns):
        pool = pool[(pool["adj"] > pool["ema50"]) & (pool["ema50"] > pool["ema200"])]
    pool = pool[pool["adj"] >= tomscan.MOM_MIN_PRICE]
    if med_turn is not None:
        pool = pool[pool["symbol"].map(med_turn).fillna(0) >= liquid_min]
    pool = outcomes(pool, "adj") if not pool.empty else pool

    return gated, pool


def pct(mask) -> float:
    m = pd.Series(mask).dropna()
    return 100.0 * m.mean() if len(m) else float("nan")


def summarize(gated: pd.DataFrame, pool: pd.DataFrame, top_n: int) -> None:
    line = "=" * 74
    print(f"\n{line}\nFOR-TOM SCORE EVALUATION\n{line}")

    top = gated[gated["rank"] <= top_n] if not gated.empty else gated
    print(f"\nscan days: {gated['scan_date'].nunique()}   "
          f"pool rows: {len(pool):,}   gated rows: {len(gated):,}   "
          f"top-{top_n} rows: {len(top):,}")

    def row(label: str, df: pd.DataFrame) -> None:
        cells = "".join(
            f"{(pct(df[c]) if c.startswith('hi') else df[c].mean()):>9.2f}"
            for c in REPORT if c in df.columns
        )
        print(f"{label:<22}{len(df):>8,}{cells}")

    head = "".join(f"{c:>9}" for c in REPORT)
    print(f"\n{'':<22}{'n':>8}{head}")
    print("-" * (30 + 9 * len(REPORT)))
    for label, df in (("BASE POOL (uptrend)", pool),
                      ("GATED (zone+RSI)", gated),
                      (f"TOP {top_n} by score", top)):
        if df is not None and not df.empty:
            row(label, df)

    if pool is not None and not pool.empty and not top.empty:
        print(f"\nLIFT of top-{top_n} over base pool:")
        for c in REPORT:
            if c not in top.columns or c not in pool.columns:
                continue
            a = pct(top[c]) if c.startswith("hi") else top[c].mean()
            b = pct(pool[c]) if c.startswith("hi") else pool[c].mean()
            print(f"    {c:<9}{a - b:+8.2f}{'pp' if c.startswith('hi') else '%'}")

    # Decile table: the real test of whether the score orders anything.
    if not gated.empty and gated["score"].notna().any():
        print(f"\n{'-' * 71}\nBy score decile (10 = highest score)\n{'-' * 71}")
        g = gated.dropna(subset=["score"]).copy()
        g["decile"] = pd.qcut(g["score"].rank(method="first"), 10,
                              labels=range(1, 11))
        print(f"{'decile':<9}{'n':>8}{'score':>9}{head}")
        for d, part in g.groupby("decile", observed=True):
            cells = "".join(
                f"{(pct(part[c]) if c.startswith('hi') else part[c].mean()):>9.2f}"
                for c in REPORT if c in part.columns
            )
            print(f"{int(d):<9}{len(part):>8,}{part['score'].mean():>9.3f}{cells}")

        hi, lo = g[g["decile"] == 10], g[g["decile"] == 1]
        print("\nD10 - D1 spread:")
        for c in REPORT:
            if c not in g.columns:
                continue
            a = pct(hi[c]) if c.startswith("hi") else hi[c].mean()
            b = pct(lo[c]) if c.startswith("hi") else lo[c].mean()
            print(f"    {c:<9}{a - b:+8.2f}{'pp' if c.startswith('hi') else '%'}")

    # Per-feature rank correlation: which inputs carry signal at all.
    if not gated.empty:
        cor_cols = [c for c in REPORT if c in gated.columns]
        print(f"\n{'-' * 71}\nFeature rank correlation with outcome (Spearman)"
              f"\n{'-' * 71}")
        print(f"{'feature':<14}{'weight':>8}"
              + "".join(f"{c:>9}" for c in cor_cols))
        wmap = {"vol_ratio": tomscan.MOM_W["vol"], "rsi": tomscan.MOM_W["rsi"],
                "atr_pct": tomscan.MOM_W["atr"], "ext_ema20": tomscan.MOM_W["ext"],
                "cmf": tomscan.MOM_W["cmf"]}
        for c in COMPONENTS + ["score"]:
            if c not in gated.columns:
                continue
            s = pd.to_numeric(gated[c], errors="coerce")
            if s.notna().sum() < 100:
                continue
            # Spearman == Pearson on ranks; avoids a scipy dependency.
            sr = s.rank()
            w = wmap.get(c)
            wtxt = f"{w:.2f}" if w is not None else ("1.00" if c == "score" else "-")
            cells = "".join(
                f"{sr.corr(pd.to_numeric(gated[oc], errors='coerce').astype(float).rank()):>9.3f}"
                for oc in cor_cols
            )
            print(f"{c:<14}{wtxt:>8}{cells}")

    print(f"\n{line}")
    for k, v in OUTCOMES.items():
        print(f"  {k:<7} {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the For-Tom score")
    ap.add_argument("--days", type=int, default=120,
                    help="trading days to evaluate")
    ap.add_argument("--top", type=int, default=40, help="shortlist size")
    ap.add_argument("--liquid", type=float, default=30.0,
                    help="min median turnover (lacs) for the base pool")
    ap.add_argument("--dump", type=str, help="save gated rows to CSV")
    args = ap.parse_args()

    stocks = load(250 + args.days + 20)
    all_dates = sorted(stocks["date"].unique())
    test_dates = all_dates[-args.days - 1:-1]  # need a next day for each
    print(f"Evaluating {len(test_dates)} scan days "
          f"({pd.Timestamp(test_dates[0]).date()} -> "
          f"{pd.Timestamp(test_dates[-1]).date()})\n")

    gated_all, pool_all = [], []
    for i, d in enumerate(test_dates, 1):
        g, p = eval_day(stocks, all_dates, d, args.liquid)
        if g is not None and not g.empty:
            gated_all.append(g)
        if p is not None and not p.empty:
            keep = ["symbol", "scan_date"] + [c for c in REPORT if c in p.columns]
            pool_all.append(p[keep])
        if i % 20 == 0:
            print(f"  {i}/{len(test_dates)} days")

    gated = pd.concat(gated_all, ignore_index=True) if gated_all else pd.DataFrame()
    pool = pd.concat(pool_all, ignore_index=True) if pool_all else pd.DataFrame()

    if args.dump and not gated.empty:
        gated.to_csv(args.dump, index=False)
        print(f"\ngated rows -> {args.dump}")

    summarize(gated, pool, args.top)


if __name__ == "__main__":
    main()
