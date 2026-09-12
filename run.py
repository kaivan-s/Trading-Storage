#!/usr/bin/env python3
"""
Sector accumulation scan.

    python run.py sectors                 # one-time: build symbol -> industry map
    python run.py scan                    # today's classification
    python run.py scan --days 120 --csv out.csv
    python run.py backtest --days 180     # flags + forward returns + base rates
    python run.py sector "Paper & Paper Products" --days 90
    python run.py coil --days 220         # daily pre-breakout watchlist
    python run.py coiltest --days 320     # forward-test vs all-stock base rate
    python run.py buytest --days 320      # forward-test the buy funnel (shape ∩ coil)
    python run.py tom --days 220          # live overlay at run time → for-tomorrow list

The sector map is a separate step because it takes ~12 minutes and only needs
doing once (re-run it monthly; NSE reviews the classification annually but
individual stocks get reclassified more often than that).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch          # noqa: E402
import panel as pnl   # noqa: E402
import scan as sc     # noqa: E402
import stocks as stk  # noqa: E402
import buytest as bt  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def _load(days: int, end: date, sector_level: str):
    print(f"Fetching {days} sessions ending {end} …")
    raw = fetch.load_history(end, days)

    smap_path = fetch.CACHE_DIR / "sector_map.json"
    if not smap_path.exists():
        print("\nNo sector map found. Run `python run.py sectors` first.")
        sys.exit(1)
    smap = pd.read_json(smap_path, orient="index").rename_axis("symbol").reset_index()
    filled = smap[sector_level].notna() & (smap[sector_level].astype(str).str.strip() != "")
    if int(filled.sum()) == 0:
        print("\nSector map exists but every industry is blank.")
        print("NSE's old quote-equity endpoint no longer returns industryInfo,")
        print("so a finished `sectors` run can still be empty. Re-run:")
        print("    python run.py sectors")
        sys.exit(1)

    print("\nBuilding panel …")
    stocks, p = pnl.build(raw, smap, sector_level=sector_level)
    print(f"  {stocks['symbol'].nunique():,} stocks, "
          f"{p['sector'].nunique()} sectors, "
          f"{p['date'].nunique()} sessions")
    if stocks.empty:
        print("Nothing survived cleaning — check the sector map and the EQ filter.")
        sys.exit(1)
    return stocks, p


def cmd_sectors(args):
    fetch.build_sector_map()
    print("\nSector map cached to", fetch.CACHE_DIR / "sector_map.json")


def cmd_scan(args):
    _, p = _load(args.days, args.end, args.sector_level)
    res = sc.classify(p)
    if res.empty:
        print("No sectors classified — check that the last date has data.")
        return

    cols = ["sector", "klass", "T", "T_rel", "B", "deliv_quality_rel",
            "cmf", "cmf_rel", "rs", "rs_chg_5", "n_stocks", "n_adv", "top_share", "note"]
    show = res[cols].copy()
    for c in ("T", "T_rel", "B", "deliv_quality_rel", "cmf", "cmf_rel", "rs", "rs_chg_5", "top_share"):
        show[c] = show[c].round(2)

    print(f"\n=== {res['date'].iloc[0]:%d-%m-%Y} ===\n")
    actionable = show[show["klass"].isin(["CROSSING", "PULLBACK"])]
    if actionable.empty:
        print("No crossings today.\n")
        print("Markets do not offer a setup every day. A scan that returns "
              "nothing is the scan working.\n")
    else:
        print(actionable.to_string(index=False), "\n")

    print("--- full table ---")
    print(show.to_string(index=False))

    if args.csv:
        res.to_csv(args.csv, index=False)
        print(f"\nWritten to {args.csv}")


def cmd_sector(args):
    stocks, p = _load(args.days, args.end, args.sector_level)
    hist = p[p["sector"].str.lower() == args.name.lower()].sort_values("date")
    if hist.empty:
        near = sorted({s for s in p["sector"].unique()
                       if args.name.lower()[:6] in s.lower()})
        print(f"No sector matching {args.name!r}."
              + (f" Did you mean: {near}" if near else ""))
        return

    cols = ["date", "T", "T_rel", "B", "B_deliv", "deliv_quality_rel",
            "cmf", "rs", "n_adv", "n_stocks", "top_share", "ret"]
    print(f"\n=== {hist['sector'].iloc[0]} ===\n")
    print(hist[cols].tail(20).round(3).to_string(index=False))

    last = stocks[(stocks["sector"].str.lower() == args.name.lower())
                  & (stocks["date"] == stocks["date"].max())]
    scols = ["symbol", "close", "ret", "turnover", "deliv_pct",
             "deliv_quality", "cmf"]
    print("\n--- constituents, latest session (by turnover) ---")
    print(last[scols].sort_values("turnover", ascending=False)
              .head(20).round(3).to_string(index=False))


def cmd_backtest(args):
    _, p = _load(args.days, args.end, args.sector_level)
    print("\nRunning forward test …")
    flags = sc.backtest(p)
    rates = sc.base_rates(flags)

    print("\n=== flags by class ===")
    print(rates.round(4).to_string(index=False))
    print(
        "\nRead `edge_N` against `n`. An edge computed on fewer than ~30 flags "
        "is noise wearing a decimal point.\n"
        "If edge is near zero or negative, the rule is not real — and "
        "thresholds tuned until a past move fits will fit that move only."
    )

    if args.csv:
        flags.to_csv(args.csv, index=False)
        print(f"Flag log written to {args.csv}")


def cmd_coil(args):
    stocks, _ = _load(args.days, args.end, args.sector_level)
    print("\nComputing coil indicators …")
    stocks = stk.add_indicators(stocks)
    hits = stk.scan(stocks, top=args.top)
    miss = stk.near_miss(stocks)

    as_of = stocks["date"].max()
    print(f"\n=== coil  {as_of:%d-%m-%Y} ===\n")
    if hits.empty:
        print("No names in a coil today.\n")
        print("The scan is supposed to return nothing most days. A list of "
              "forty names is the scan being too loose, not too useful.\n")
    else:
        show = hits.copy()
        num = [c for c in show.columns if c != "coil"
               and pd.api.types.is_numeric_dtype(show[c])]
        show[num] = show[num].round(3)
        show["coil"] = show["coil"].round(1)
        print(show.to_string(index=False), "\n")

    print("--- near misses (fail exactly one filter) ---")
    if miss.empty:
        print("none")
    else:
        print(miss.round(3).to_string(index=False))

    if args.csv:
        hits.to_csv(args.csv, index=False)
        print(f"\nWritten to {args.csv}")


def cmd_coiltest(args):
    stocks, _ = _load(args.days, args.end, args.sector_level)
    print("\nComputing coil indicators …")
    stocks = stk.add_indicators(stocks)
    print("Running forward test …")
    ft = stk.forward_test(stocks)

    print(f"\n=== coil forward test  ({len(ft)} dates) ===\n")
    means = ft.drop(columns=["date"]).mean()
    print(means.to_frame("mean").T.round(4).to_string(index=False))
    print(
        f"\nMean names flagged per day: {ft['n_flagged'].mean():.1f}"
        f"  (of {ft['n_universe'].mean():.0f} liquid)\n"
    )
    print(
        "Read `edge_N` against how many dates actually flagged anyone. "
        "If edge is near zero the scan is a well-tested way to generate "
        "random watchlists — find that out here, not from a P&L."
    )

    if args.csv:
        ft.to_csv(args.csv, index=False)
        print(f"\nWritten to {args.csv}")


def cmd_tom(args):
    stocks, _ = _load(max(args.days, 220), args.end, args.sector_level)
    print("\nComputing coil indicators …")
    stocks = stk.add_indicators(stocks)
    print("Fetching live last prices …")
    import tom as tomscan
    live = fetch.live_snapshot()
    print(f"  {len(live):,} live prints")
    rows = tomscan.for_tomorrow_momentum(stocks, live, top_n=40)
    print(f"\n=== for tomorrow  ({len(rows)} names) ===\n")
    if rows.empty:
        print("Nothing near or through a trigger on this snapshot.")
        return
    show = rows.drop(columns=["why"], errors="ignore")
    print(show.round(3).to_string(index=False))
    if args.csv:
        rows.to_csv(args.csv, index=False)
        print(f"\nWritten to {args.csv}")


def cmd_buytest(args):
    stocks, p = _load(args.days, args.end, args.sector_level)
    print("\nComputing coil indicators …")
    stocks = stk.add_indicators(stocks)

    def progress(i, total, d):
        if i % 10 == 0 or i == total:
            print(f"  {i}/{total}  ({d:%d-%m-%Y})")

    print("Running buy-funnel forward test … (slow: reclassifies every day)")
    ft = bt.buy_forward_test(stocks, p, on_progress=progress)
    if ft.empty:
        print("Not enough history. Use --days 260+ for a meaningful window.")
        return

    fired = int((ft["n_buys"] > 0).sum())
    print(f"\n=== buy funnel forward test  ({len(ft)} dates, "
          f"{fired} with a buy setup) ===\n")
    print(bt.summary(ft).round(4).to_string(index=False))
    print(
        f"\nMean buy setups on days that fired: "
        f"{ft[ft['n_buys'] > 0]['n_buys'].mean():.1f}"
        f"   ·   coil hits/day: {ft['n_coil'].mean():.1f}\n"
    )
    print(
        "Read `edge_vs_base` against how often the funnel fired, and "
        "`edge_vs_coil` to see whether the sector overlay beats plain coil. "
        "If edge_vs_coil is ~0, the badge is a coil rename."
    )

    if args.csv:
        ft.to_csv(args.csv, index=False)
        print(f"\nPer-date log written to {args.csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # These live on a parent parser so they work AFTER the subcommand
    # (`run.py scan --days 90`), which is how everyone types it. Declaring them
    # on the top-level parser only makes them valid before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--days", type=int, default=90,
                        help="trading sessions to load (default 90; use 200+ for backtest)")
    common.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                        default=date.today(), help="last session, YYYY-MM-DD")
    common.add_argument("--sector-level", default="basic_industry",
                        choices=["basic_industry", "industry", "sector", "macro"])
    common.add_argument("--csv", help="write results here")

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sectors", parents=[common],
                   help="build the symbol -> industry map (one-time)")
    sub.add_parser("scan", parents=[common],
                   help="classify every sector as of the latest session")
    sub.add_parser("backtest", parents=[common],
                   help="historical flags with forward returns")
    sp = sub.add_parser("sector", parents=[common], help="drill into one sector")
    sp.add_argument("name")
    cp = sub.add_parser("coil", parents=[common],
                        help="daily stock-level pre-breakout watchlist")
    cp.add_argument("--top", type=int, default=40, help="names to show (default 40)")
    sub.add_parser("coiltest", parents=[common],
                   help="forward-test the coil scan against the all-stock base rate")
    sub.add_parser("buytest", parents=[common],
                   help="forward-test the buy funnel (sector shape ∩ coil)")
    sub.add_parser("tom", parents=[common],
                   help="live overlay at run time: potential breakouts for tomorrow")

    args = ap.parse_args()
    {"sectors": cmd_sectors, "scan": cmd_scan, "backtest": cmd_backtest,
     "sector": cmd_sector, "coil": cmd_coil, "coiltest": cmd_coiltest,
     "buytest": cmd_buytest, "tom": cmd_tom}[args.cmd](args)


if __name__ == "__main__":
    main()
