#!/usr/bin/env python3
"""
Head-to-head backtest: CURRENT For-Tom vs MOMENTUM-tilted variant.

For each trading day:
  - CURRENT  = tom.for_tomorrow(...)              (coil / quietness logic)
  - MOMENTUM = tom.for_tomorrow_momentum(...)     (energy-scored top-N)

To keep the comparison fair, MOMENTUM is given the SAME per-day budget K as
CURRENT selected that day (top-K by score). Both are scored on the same
outcome: did the next day's HIGH reach +3% / +5% above the scan-day close.

Usage:
    python backtest_compare.py --days 30
    python backtest_compare.py --days 60 --gain 3
"""

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

import fetch
import stocks as st
import scan
import panel as pnl
import tom as tomscan


def load_data(start_date: date, end_date: date):
    test_days = (end_date - start_date).days
    n_sessions = 250 + test_days + 10
    print(f"Loading {n_sessions} sessions ending {end_date + timedelta(days=5)}…")
    raw = fetch.load_history(end=end_date + timedelta(days=5), n_sessions=n_sessions)
    if raw is None or raw.empty:
        print("No data available")
        return None, None
    smap_path = fetch.CACHE_DIR / "sector_map.json"
    if not smap_path.exists():
        print(f"No sector map at {smap_path}. Run: python run.py sectors")
        return None, None
    smap = pd.read_json(smap_path, orient="index").rename_axis("symbol").reset_index()
    print("Building panel…")
    stocks, panel = pnl.build(raw, smap)
    if stocks is None or stocks.empty:
        print("Failed to build panel")
        return None, None
    print("Computing indicators…")
    stocks = st.add_indicators(stocks)
    return stocks, panel


def evaluate(rows: pd.DataFrame, nd: pd.DataFrame, gain_pct: float, method: str,
             scan_date, out: list):
    """Score a selection frame against next-day highs."""
    thresh_hit3 = gain_pct
    for _, r in rows.iterrows():
        sym = r["symbol"]
        if sym not in nd.index:
            continue
        scan_price = float(r["ltp"])
        if scan_price <= 0:
            continue
        nhigh = float(nd.loc[sym, "high"])
        max_gain = (nhigh / scan_price - 1.0) * 100.0
        out.append({
            "scan_date": scan_date,
            "method": method,
            "symbol": sym,
            "max_gain": max_gain,
            "gain_3pct": max_gain >= 3.0,
            "gain_5pct": max_gain >= 5.0,
        })


def run(start_date: date, end_date: date, gain_pct: float = 3.0):
    stocks, panel = load_data(start_date, end_date)
    if stocks is None:
        return None

    all_dates = sorted(stocks["date"].unique())
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    test_dates = [d for d in all_dates if start_ts <= pd.Timestamp(d) <= end_ts]
    print(f"Comparing over {len(test_dates)} trading days\n")

    rows = []
    for i, scan_date in enumerate(test_dates):
        eod_stocks = stocks[stocks["date"] <= scan_date]
        last_day = eod_stocks[eod_stocks["date"] == scan_date].copy()
        if last_day.empty:
            continue
        nexts = [d for d in all_dates if d > scan_date]
        if not nexts:
            continue
        next_day = stocks[stocks["date"] == nexts[0]]
        if next_day.empty:
            continue
        nd = next_day.set_index("symbol")[["adj", "high", "open"]]

        # live snapshot from EOD
        live = last_day[["symbol"]].copy()
        live["ltp"] = last_day["adj"]
        live["volume"] = last_day["volume"]
        live["pchange"] = last_day["ret"] * 100 if "ret" in last_day.columns else 0
        live["high"] = last_day["high"] if "high" in last_day.columns else last_day["adj"]

        eod_panel = panel[panel["date"] <= scan_date]
        scan_rows = scan.classify(eod_panel, as_of=scan_date)
        ready = scan.recommend_sectors(scan_rows, eod_panel)
        coil_all = st.scan(eod_stocks, top=10_000)
        buys = (coil_all[coil_all["sector"].isin(ready)]
                if not coil_all.empty else coil_all)

        # CURRENT
        cur = tomscan.for_tomorrow(eod_stocks, live, buys, scan_rows=scan_rows)
        k = len(cur)
        if k == 0:
            continue

        # MOMENTUM — matched budget K
        mom = tomscan.for_tomorrow_momentum(eod_stocks, live, scan_rows=scan_rows, top_n=k)

        evaluate(cur, nd, gain_pct, "CURRENT", scan_date, rows)
        evaluate(mom, nd, gain_pct, "MOMENTUM", scan_date, rows)

        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(test_dates)} days…")

    return pd.DataFrame(rows)


def report(df: pd.DataFrame, gain_pct: float):
    print("\n" + "=" * 62)
    print("HEAD-TO-HEAD: Current (coil) vs Momentum-tilted")
    print("=" * 62)
    if df is None or df.empty:
        print("No results.")
        return

    print(f"\n{'metric':<22}{'CURRENT':>12}{'MOMENTUM':>12}{'edge':>10}")
    print("-" * 56)

    def line(label, cur_v, mom_v, pct=True, higher_better=True):
        edge = mom_v - cur_v
        arrow = ""
        if (edge > 0) == higher_better and abs(edge) > 1e-9:
            arrow = "  ✓ MOM"
        elif abs(edge) > 1e-9:
            arrow = "  ✓ CUR"
        if pct:
            print(f"{label:<22}{cur_v:>11.1f}%{mom_v:>11.1f}%{edge:>+9.1f}{arrow}")
        else:
            print(f"{label:<22}{cur_v:>12.2f}{mom_v:>12.2f}{edge:>+10.2f}{arrow}")

    cur = df[df["method"] == "CURRENT"]
    mom = df[df["method"] == "MOMENTUM"]

    line("Picks", len(cur), len(mom), pct=False)
    line(f"Reached +3%", 100 * cur["gain_3pct"].mean(), 100 * mom["gain_3pct"].mean())
    line(f"Reached +5%", 100 * cur["gain_5pct"].mean(), 100 * mom["gain_5pct"].mean())
    line("Avg max gain", cur["max_gain"].mean(), mom["max_gain"].mean(), pct=False)
    line("Median max gain", cur["max_gain"].median(), mom["max_gain"].median(), pct=False)

    # Daily win comparison (which method had higher +3% rate that day)
    print("\nDaily +3% hit rate (per day):")
    cur_d = cur.groupby("scan_date")["gain_3pct"].mean() * 100
    mom_d = mom.groupby("scan_date")["gain_3pct"].mean() * 100
    joined = pd.DataFrame({"cur": cur_d, "mom": mom_d}).dropna()
    mom_wins = (joined["mom"] > joined["cur"]).sum()
    cur_wins = (joined["cur"] > joined["mom"]).sum()
    ties = (joined["cur"] == joined["mom"]).sum()
    print(f"  Momentum better on {mom_wins} days · "
          f"Current better on {cur_wins} days · ties {ties}")
    print(f"  Avg daily +3%:  CURRENT {joined['cur'].mean():.1f}%  "
          f"MOMENTUM {joined['mom'].mean():.1f}%")

    # Overlap: how different are the two lists?
    print("\nList overlap:")
    ov = []
    for d in joined.index:
        cs = set(cur[cur["scan_date"] == d]["symbol"])
        ms = set(mom[mom["scan_date"] == d]["symbol"])
        if cs or ms:
            ov.append(len(cs & ms) / max(len(cs | ms), 1))
    if ov:
        print(f"  Avg Jaccard overlap of picks: {100*np.mean(ov):.0f}% "
              "(low = genuinely different lists)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--start", type=str)
    ap.add_argument("--end", type=str)
    ap.add_argument("--gain", type=float, default=3.0)
    ap.add_argument("--dump", type=str)
    args = ap.parse_args()

    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days)

    df = run(start_date, end_date, args.gain)
    if args.dump and df is not None and not df.empty:
        df.to_csv(args.dump, index=False)
        print(f"\nRaw rows saved to {args.dump}")
    report(df, args.gain)


if __name__ == "__main__":
    main()
