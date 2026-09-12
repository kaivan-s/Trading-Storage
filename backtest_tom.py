#!/usr/bin/env python3
"""
Backtest "For Tom" predictions on historical data.

For each day in the lookback period:
1. Run for_tomorrow() on that day's EOD data
2. Check next day's results (did price close above trigger?)
3. Aggregate hit rates by category

Usage:
    python backtest_tom.py --days 60
    python backtest_tom.py --start 2026-07-01 --end 2026-09-01
"""

import argparse
from datetime import date, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

import fetch
import stocks as st
import scan
import panel as pnl
import tom as tomscan


def run_backtest(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Run for_tomorrow on each day from start_date to end_date,
    then check next-day results.
    """
    print(f"Backtesting For Tom: {start_date} to {end_date}")
    
    # Calculate how many sessions we need
    # ~250 days lookback for indicators + test period + buffer
    test_days = (end_date - start_date).days
    n_sessions = 250 + test_days + 10  # indicators + test window + buffer
    
    print(f"Loading {n_sessions} sessions of historical data...")
    raw = fetch.load_history(
        end=end_date + timedelta(days=5),  # buffer for next-day checks
        n_sessions=n_sessions
    )
    
    if raw is None or raw.empty:
        print("No data available")
        return pd.DataFrame()
    
    # Load sector map from cache
    print("Loading sector map...")
    smap_path = fetch.CACHE_DIR / "sector_map.json"
    if not smap_path.exists():
        print(f"No sector map at {smap_path}. Run: python run.py sectors")
        return pd.DataFrame()
    smap = pd.read_json(smap_path, orient="index").rename_axis("symbol").reset_index()
    
    # Build panel (this adds 'tri' column and other computed fields)
    print("Building sector panel...")
    stocks, panel = pnl.build(raw, smap)
    
    if stocks is None or stocks.empty:
        print("Failed to build panel")
        return pd.DataFrame()
    
    # Add coil indicators
    print("Computing coil indicators...")
    stocks = st.add_indicators(stocks)
    
    if stocks is None or stocks.empty:
        print("Failed to compute indicators")
        return pd.DataFrame()
    
    # Get all trading days
    all_dates = sorted(stocks["date"].unique())
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    test_dates = [d for d in all_dates if start_ts <= d <= end_ts]
    
    print(f"Testing {len(test_dates)} trading days")
    
    results = []
    
    for i, scan_date in enumerate(test_dates):
        # Get EOD data up to scan_date
        eod_stocks = stocks[stocks["date"] <= scan_date].copy()
        
        if eod_stocks.empty:
            continue
        
        # Build "live" snapshot from EOD (simulating EOD close)
        last_day = eod_stocks[eod_stocks["date"] == scan_date].copy()
        if last_day.empty:
            continue
        
        live = last_day[["symbol"]].copy()
        live["ltp"] = last_day["adj"]
        live["volume"] = last_day["volume"]
        live["pchange"] = last_day["ret"] * 100 if "ret" in last_day.columns else 0
        live["high"] = last_day["high"] if "high" in last_day.columns else last_day["adj"]
        
        # Get sector panel for that date
        eod_panel = panel[panel["date"] <= scan_date].copy()
        scan_rows = scan.classify(eod_panel, as_of=scan_date)

        # Momentum-tilted For Tom (production method)
        tom_rows = tomscan.for_tomorrow_momentum(
            eod_stocks, live, scan_rows=scan_rows, top_n=40,
        )
        
        if tom_rows.empty:
            continue
        
        # Get next trading day
        next_dates = [d for d in all_dates if d > scan_date]
        if not next_dates:
            continue
        next_date = next_dates[0]
        
        # Get next day's data
        next_day = stocks[stocks["date"] == next_date]
        if next_day.empty:
            continue
        
        next_prices = next_day.set_index("symbol")[["adj", "high", "open"]].to_dict(orient="index")
        
        # Check each prediction
        for _, row in tom_rows.iterrows():
            sym = row["symbol"]
            kind = row["kind"]
            trigger = row["trigger"]
            
            if sym not in next_prices:
                continue
            
            np_data = next_prices[sym]
            next_close = np_data.get("adj", 0)
            next_high = np_data.get("high", 0)
            next_open = np_data.get("open", 0)
            
            # Win conditions
            scan_price = row["ltp"]
            closed_above = next_close > trigger
            gapped_above = next_open > trigger
            touched_trigger = next_high > trigger
            
            # Simple gain metrics (did next day's high reach X% above scan price?)
            gain_3pct = next_high >= scan_price * 1.03
            gain_5pct = next_high >= scan_price * 1.05
            max_gain = (next_high / scan_price - 1) * 100 if scan_price > 0 else 0
            
            results.append({
                "scan_date": scan_date,
                "next_date": next_date,
                "symbol": sym,
                "kind": kind,
                "trigger": trigger,
                "scan_price": scan_price,
                "next_open": next_open,
                "next_high": next_high,
                "next_close": next_close,
                "closed_above": closed_above,
                "gapped_above": gapped_above,
                "touched_trigger": touched_trigger,
                "gain_3pct": gain_3pct,
                "gain_5pct": gain_5pct,
                "max_gain": max_gain,
            })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_dates)} days...")
    
    return pd.DataFrame(results)


def summarize_results(df: pd.DataFrame) -> None:
    """Print summary statistics by category."""
    
    if df.empty:
        print("\nNo results to summarize.")
        return
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS: For Tom")
    print("=" * 60)
    
    total = len(df)
    total_3pct = df["gain_3pct"].sum()
    total_5pct = df["gain_5pct"].sum()
    avg_max_gain = df["max_gain"].mean()
    
    print(f"\nOverall: {total} predictions")
    print(f"  Reached +3%:  {total_3pct} ({100*total_3pct/total:.1f}%)")
    print(f"  Reached +5%:  {total_5pct} ({100*total_5pct/total:.1f}%)")
    print(f"  Avg max gain: {avg_max_gain:.1f}%")
    
    print("\n" + "-" * 60)
    print("By Category:")
    print("-" * 60)
    
    for kind in ["through", "momentum", "setup", "early", "near", "potential"]:
        subset = df[df["kind"] == kind]
        if subset.empty:
            continue
        
        n = len(subset)
        g3 = subset["gain_3pct"].sum()
        g5 = subset["gain_5pct"].sum()
        avg_gain = subset["max_gain"].mean()
        
        print(f"\n{kind.upper()}  (n={n})")
        print(f"  Reached +3%:   {g3:4d} ({100*g3/n:5.1f}%)")
        print(f"  Reached +5%:   {g5:4d} ({100*g5/n:5.1f}%)")
        print(f"  Avg max gain:  {avg_gain:5.1f}%")
    
    # Daily breakdown
    print("\n" + "-" * 60)
    print("Daily Summary:")
    print("-" * 60)
    
    daily = df.groupby("scan_date").agg({
        "symbol": "count",
        "gain_3pct": "sum",
        "gain_5pct": "sum",
        "max_gain": "mean",
    }).rename(columns={"symbol": "predictions"})
    
    daily["hit_rate_3"] = daily["gain_3pct"] / daily["predictions"] * 100
    
    print(f"\nAverage predictions per day: {daily['predictions'].mean():.1f}")
    print(f"Average +3% hit rate: {daily['hit_rate_3'].mean():.1f}%")
    print(f"Average max gain: {daily['max_gain'].mean():.1f}%")
    print(f"Best day (+3%): {daily['hit_rate_3'].max():.1f}%")
    print(f"Worst day (+3%): {daily['hit_rate_3'].min():.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Backtest For Tom predictions")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, help="Save results to CSV")
    
    args = parser.parse_args()
    
    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=1)  # yesterday
        start_date = end_date - timedelta(days=args.days)
    
    results = run_backtest(start_date, end_date)
    
    if args.output and not results.empty:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    
    summarize_results(results)


if __name__ == "__main__":
    main()
