#!/usr/bin/env python3
"""
Head-to-head: current For-Tom (coil / quiet filters) vs momentum-tilted score.

Same dates, same EOD snapshot, same next-day outcome:
    win = next day's high reached +3% (or +5%) from the scan-day close.

Usage:
    python compare_tom.py --days 30
    python compare_tom.py --days 30 --top 20
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

import fetch
import stocks as st
import scan
import panel as pnl
import tom as tomscan


def load(start_date: date, end_date: date):
    test_days = (end_date - start_date).days
    n_sessions = 250 + test_days + 10
    print(f"Loading {n_sessions} sessions…")
    raw = fetch.load_history(end=end_date + timedelta(days=5), n_sessions=n_sessions)
    if raw is None or raw.empty:
        return None, None

    smap_path = fetch.CACHE_DIR / "sector_map.json"
    if not smap_path.exists():
        print(f"No sector map at {smap_path}")
        return None, None
    smap = pd.read_json(smap_path, orient="index").rename_axis("symbol").reset_index()

    print("Building panel + indicators…")
    stocks, panel = pnl.build(raw, smap)
    if stocks is None or stocks.empty:
        return None, None
    stocks = st.add_indicators(stocks)
    return stocks, panel


def score_picks(tom_rows: pd.DataFrame, next_prices: dict, method: str, scan_date, next_date):
    rows = []
    if tom_rows is None or tom_rows.empty:
        return rows
    for _, r in tom_rows.iterrows():
        sym = r["symbol"]
        if sym not in next_prices:
            continue
        scan_price = float(r["ltp"])
        if scan_price <= 0:
            continue
        npd = next_prices[sym]
        next_high = float(npd.get("high") or 0)
        max_gain = (next_high / scan_price - 1.0) * 100.0
        rows.append({
            "method": method,
            "scan_date": scan_date,
            "next_date": next_date,
            "symbol": sym,
            "kind": r.get("kind", "momentum"),
            "score": r.get("score"),
            "scan_price": scan_price,
            "next_high": next_high,
            "gain_3pct": next_high >= scan_price * 1.03,
            "gain_5pct": next_high >= scan_price * 1.05,
            "max_gain": max_gain,
        })
    return rows


def run(start_date: date, end_date: date, top_n: int) -> pd.DataFrame:
    stocks, panel = load(start_date, end_date)
    if stocks is None:
        return pd.DataFrame()

    all_dates = sorted(stocks["date"].unique())
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    test_dates = [d for d in all_dates if start_ts <= pd.Timestamp(d) <= end_ts]
    print(f"Comparing both methods on {len(test_dates)} days (top {top_n} each)\n")

    results = []
    for i, scan_date in enumerate(test_dates):
        eod_stocks = stocks[stocks["date"] <= scan_date]
        last_day = eod_stocks[eod_stocks["date"] == scan_date]
        if last_day.empty:
            continue

        nexts = [d for d in all_dates if d > scan_date]
        if not nexts:
            continue
        next_date = nexts[0]
        next_day = stocks[stocks["date"] == next_date]
        if next_day.empty:
            continue
        next_prices = next_day.set_index("symbol")[["adj", "high", "open"]].to_dict("index")

        live = last_day[["symbol"]].copy()
        live["ltp"] = last_day["adj"]
        live["volume"] = last_day["volume"]
        live["pchange"] = last_day["ret"] * 100 if "ret" in last_day.columns else 0
        live["high"] = last_day["high"] if "high" in last_day.columns else last_day["adj"]

        eod_panel = panel[panel["date"] <= scan_date]
        scan_rows = scan.classify(eod_panel, as_of=scan_date)
        ready = scan.recommend_sectors(scan_rows, eod_panel)
        coil_all = st.scan(eod_stocks, top=10_000)
        buys = coil_all[coil_all["sector"].isin(ready)] if not coil_all.empty else coil_all

        current = tomscan.for_tomorrow(eod_stocks, live, buys, scan_rows=scan_rows)
        momentum = tomscan.for_tomorrow_momentum(
            eod_stocks, live, scan_rows=scan_rows, top_n=top_n,
        )
        if current is not None and not current.empty:
            current = current.head(top_n)

        results.extend(score_picks(current, next_prices, "current", scan_date, next_date))
        results.extend(score_picks(momentum, next_prices, "momentum", scan_date, next_date))

        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(test_dates)} days…")

    return pd.DataFrame(results)


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "hit3": 0, "hit5": 0, "avg": 0.0, "med": 0.0, "per_day": 0.0}
    return {
        "n": len(df),
        "hit3": 100 * df["gain_3pct"].mean(),
        "hit5": 100 * df["gain_5pct"].mean(),
        "avg": df["max_gain"].mean(),
        "med": df["max_gain"].median(),
        "per_day": df.groupby("scan_date").size().mean(),
    }


def report(df: pd.DataFrame, top_n: int) -> None:
    if df.empty:
        print("No results.")
        return

    cur = df[df["method"] == "current"]
    mom = df[df["method"] == "momentum"]
    sc, sm = _stats(cur), _stats(mom)

    print("\n" + "=" * 68)
    print(f"HEAD-TO-HEAD  ·  top {top_n} picks / day")
    print("=" * 68)
    print(f"\n{'':22}{'CURRENT (coil)':>18}{'MOMENTUM':>16}{'delta':>12}")
    print("-" * 68)
    print(f"{'Predictions':22}{sc['n']:>18}{sm['n']:>16}")
    print(f"{'Per day':22}{sc['per_day']:>18.1f}{sm['per_day']:>16.1f}")
    print(f"{'+3% hit rate':22}{sc['hit3']:>17.1f}%{sm['hit3']:>15.1f}%{sm['hit3']-sc['hit3']:>+11.1f}pp")
    print(f"{'+5% hit rate':22}{sc['hit5']:>17.1f}%{sm['hit5']:>15.1f}%{sm['hit5']-sc['hit5']:>+11.1f}pp")
    print(f"{'Avg max gain':22}{sc['avg']:>17.1f}%{sm['avg']:>15.1f}%{sm['avg']-sc['avg']:>+11.1f}pp")
    print(f"{'Median max gain':22}{sc['med']:>17.1f}%{sm['med']:>15.1f}%")

    # Overlap / exclusive winners
    print("\n" + "-" * 68)
    print("Overlap (same symbol, same day)")
    print("-" * 68)
    both = cur.merge(mom, on=["scan_date", "symbol"], suffixes=("_c", "_m"))
    only_c = cur.merge(mom[["scan_date", "symbol"]], on=["scan_date", "symbol"], how="left", indicator=True)
    only_c = only_c[only_c["_merge"] == "left_only"]
    only_m = mom.merge(cur[["scan_date", "symbol"]], on=["scan_date", "symbol"], how="left", indicator=True)
    only_m = only_m[only_m["_merge"] == "left_only"]

    print(f"  Shared picks:              {len(both)}")
    if not both.empty:
        print(f"    +3% on shared:           {100*both['gain_3pct_c'].mean():.1f}%")
    print(f"  Current-only:              {len(only_c)}   +3% {100*only_c['gain_3pct'].mean():.1f}%"
          if not only_c.empty else "  Current-only:              0")
    print(f"  Momentum-only:             {len(only_m)}   +3% {100*only_m['gain_3pct'].mean():.1f}%"
          if not only_m.empty else "  Momentum-only:             0")

    # Daily consistency
    print("\n" + "-" * 68)
    print("Daily +3% hit-rate consistency")
    print("-" * 68)
    dcur = cur.groupby("scan_date")["gain_3pct"].mean() * 100
    dmom = mom.groupby("scan_date")["gain_3pct"].mean() * 100
    if not dcur.empty:
        print(f"  Current  best/worst/avg:  {dcur.max():.1f}% / {dcur.min():.1f}% / {dcur.mean():.1f}%")
    if not dmom.empty:
        print(f"  Momentum best/worst/avg:  {dmom.max():.1f}% / {dmom.min():.1f}% / {dmom.mean():.1f}%")

    # Current by kind vs momentum overall
    print("\n" + "-" * 68)
    print("Current method, by category (for context)")
    print("-" * 68)
    for kind, g in cur.groupby("kind"):
        print(f"  {kind:<12} n={len(g):<5}  +3% {100*g['gain_3pct'].mean():5.1f}%  "
              f"+5% {100*g['gain_5pct'].mean():5.1f}%  avg {g['max_gain'].mean():5.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Compare coil vs momentum For-Tom")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--start", type=str)
    ap.add_argument("--end", type=str)
    ap.add_argument("--top", type=int, default=40, help="Picks per method per day")
    ap.add_argument("--output", type=str)
    args = ap.parse_args()

    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days)

    df = run(start_date, end_date, args.top)
    if args.output and not df.empty:
        df.to_csv(args.output, index=False)
        print(f"\nSaved {args.output}")
    report(df, args.top)


if __name__ == "__main__":
    main()
