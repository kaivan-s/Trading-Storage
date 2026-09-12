#!/usr/bin/env python3
"""
Pattern analysis for the "For Tom" method — two robustness studies.

TASK 1 — Why do shortlisted stocks FAIL?
    Among the stocks we shortlisted, compare the ones that gained >=3% the
    next day (winners) against the ones that did not (failures). Surface the
    scan-day features that most separate the two so we know what to tighten.

TASK 2 — What WINNERS did we miss?
    For each day, find every stock that gained >=3% the next day. Split into
    "we shortlisted it" vs "we missed it". For the misses, classify WHY the
    For-Tom gates excluded them, so we can see if a filter is too strict.

Usage:
    python analyze_patterns.py --days 30
    python analyze_patterns.py --days 60 --gain 3 --dump patterns.csv
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

# Scan-day features we carry through for both studies.
FEATURE_COLS = [
    "adj", "ema20", "ema50", "ema200", "rsi", "cmf", "vol_ratio",
    "pos_hi", "to_trigger", "trigger", "range20", "contraction",
    "ext_ema20", "atr_pct", "base_days",
]


# --------------------------------------------------------------------------
# Data loading (same pipeline the server / backtest use)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Miss-reason classifier
# --------------------------------------------------------------------------
def classify_miss(f: dict) -> str:
    """
    Why did For-Tom exclude this stock (that went on to gain >=3%)?
    Evaluated in priority order against the same gates for_tomorrow uses.
    """
    adj = f.get("adj")
    ema50 = f.get("ema50")
    ema200 = f.get("ema200")
    to_trig = f.get("to_trigger")
    pos_hi = f.get("pos_hi") or 0
    rsi = f.get("rsi")
    cmf = f.get("cmf")
    vol = f.get("vol_ratio") or 0

    # 1. Trend gate — we only ever consider uptrends.
    in_trend = (adj is not None and ema50 is not None and ema200 is not None
                and adj > ema50 and ema50 > ema200)
    if not in_trend:
        return "downtrend"

    # 2. Distance gate — outside the 6% potential zone entirely.
    if to_trig is None or to_trig > tomscan.POTENTIAL_TRIGGER:
        return "far_from_high"

    # In the zone and in trend — so a sub-filter blocked it. Find which.
    # 3. Not actually near its own high (coiling lower in the band).
    if pos_hi < 0.88:
        return "not_near_high"
    # 4. Money flow was negative — accumulation not confirmed.
    if cmf is not None and cmf < 0:
        return "weak_flow"
    # 5. RSI too hot for potential, volume too weak for momentum.
    if rsi is not None and rsi > 72 and vol < tomscan.MOMENTUM_VOL_RATIO:
        return "rsi_hot_no_vol"
    # 6. Quiet — near the high, positive flow, but no volume build/heat.
    if vol < tomscan.VOL_BUILDING:
        return "too_quiet"
    return "in_zone_other"


# --------------------------------------------------------------------------
# Core run
# --------------------------------------------------------------------------
def run(start_date: date, end_date: date, gain_pct: float = 3.0):
    stocks, panel = load_data(start_date, end_date)
    if stocks is None:
        return None, None

    all_dates = sorted(stocks["date"].unique())
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    test_dates = [d for d in all_dates if start_ts <= pd.Timestamp(d) <= end_ts]
    print(f"Analyzing {len(test_dates)} trading days\n")

    thresh = 1.0 + gain_pct / 100.0
    shortlisted_rows = []   # Task 1
    universe_winners = []   # Task 2

    for i, scan_date in enumerate(test_dates):
        eod_stocks = stocks[stocks["date"] <= scan_date]
        last_day = eod_stocks[eod_stocks["date"] == scan_date].copy()
        if last_day.empty:
            continue

        # next trading day
        nexts = [d for d in all_dates if d > scan_date]
        if not nexts:
            continue
        next_date = nexts[0]
        next_day = stocks[stocks["date"] == next_date]
        if next_day.empty:
            continue

        nd = next_day.set_index("symbol")[["adj", "high", "open"]]
        feat = last_day.set_index("symbol")

        # sector class map for the day
        eod_panel = panel[panel["date"] <= scan_date]
        scan_rows = scan.classify(eod_panel, as_of=scan_date)
        ready = scan.recommend_sectors(scan_rows, eod_panel)
        klass_map = dict(zip(scan_rows["sector"], scan_rows["klass"]))

        # --- Run For-Tom exactly as production would ---------------------
        live = last_day[["symbol"]].copy()
        live["ltp"] = last_day["adj"]
        live["volume"] = last_day["volume"]
        live["pchange"] = last_day["ret"] * 100 if "ret" in last_day.columns else 0
        live["high"] = last_day["high"] if "high" in last_day.columns else last_day["adj"]

        coil_all = st.scan(eod_stocks, top=10_000)
        buys = (coil_all[coil_all["sector"].isin(ready)]
                if not coil_all.empty else coil_all)
        tom_rows = tomscan.for_tomorrow(eod_stocks, live, buys, scan_rows=scan_rows)
        shortlisted = set(tom_rows["symbol"]) if not tom_rows.empty else set()

        # --- Task 1: outcomes + features for shortlisted -----------------
        if not tom_rows.empty:
            for _, r in tom_rows.iterrows():
                sym = r["symbol"]
                if sym not in nd.index or sym not in feat.index:
                    continue
                scan_price = float(r["ltp"])
                if scan_price <= 0:
                    continue
                nhigh = float(nd.loc[sym, "high"])
                max_gain = (nhigh / scan_price - 1.0) * 100.0
                rec = {c: feat.loc[sym].get(c) for c in FEATURE_COLS}
                rec.update({
                    "scan_date": scan_date,
                    "symbol": sym,
                    "kind": r["kind"],
                    "sector_klass": r.get("sector_klass", ""),
                    "max_gain": max_gain,
                    "win": max_gain >= gain_pct,
                })
                shortlisted_rows.append(rec)

        # --- Task 2: every stock that gained >=X% next day ---------------
        merged = feat.join(nd[["high"]], rsuffix="_next", how="inner")
        merged = merged[merged["adj"] > 0]
        gained = merged[merged["high"] >= merged["adj"] * thresh]
        for sym, row in gained.iterrows():
            scan_price = float(row["adj"])
            max_gain = (float(row["high"]) / scan_price - 1.0) * 100.0
            was_listed = sym in shortlisted
            f = {c: row.get(c) for c in FEATURE_COLS}
            reason = "" if was_listed else classify_miss(f)
            rec = {c: f.get(c) for c in FEATURE_COLS}
            rec.update({
                "scan_date": scan_date,
                "symbol": sym,
                "sector": row.get("sector"),
                "sector_klass": klass_map.get(row.get("sector"), ""),
                "max_gain": max_gain,
                "shortlisted": was_listed,
                "miss_reason": reason,
            })
            universe_winners.append(rec)

        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(test_dates)} days…")

    return pd.DataFrame(shortlisted_rows), pd.DataFrame(universe_winners)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def report_failures(df: pd.DataFrame, gain_pct: float):
    print("\n" + "=" * 68)
    print(f"TASK 1 — Why shortlisted stocks FAIL (win = reached +{gain_pct:.0f}%)")
    print("=" * 68)
    if df is None or df.empty:
        print("No shortlisted rows.")
        return

    n = len(df)
    win = df[df["win"]]
    fail = df[~df["win"]]
    print(f"\nShortlisted: {n} · Winners: {len(win)} ({100*len(win)/n:.1f}%) · "
          f"Failures: {len(fail)} ({100*len(fail)/n:.1f}%)")

    # Feature means: winners vs failures + the gap.
    metrics = ["to_trigger", "pos_hi", "rsi", "cmf", "vol_ratio",
               "range20", "contraction", "ext_ema20", "atr_pct", "base_days"]
    print("\nScan-day feature averages:")
    print(f"  {'feature':<14}{'WIN':>10}{'FAIL':>10}{'gap':>10}")
    print("  " + "-" * 42)
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        w = pd.to_numeric(win[m], errors="coerce").mean()
        fl = pd.to_numeric(fail[m], errors="coerce").mean()
        if pd.isna(w) or pd.isna(fl):
            continue
        gap = w - fl
        rows.append((m, w, fl, gap))
        print(f"  {m:<14}{w:>10.3f}{fl:>10.3f}{gap:>+10.3f}")

    # Rank features by how strongly they separate win/fail (normalised gap).
    print("\nMost separating features (|gap| / failure spread):")
    seps = []
    for m, w, fl, gap in rows:
        sd = pd.to_numeric(fail[m], errors="coerce").std()
        if sd and sd > 0:
            seps.append((m, abs(gap) / sd, gap))
    seps.sort(key=lambda x: x[1], reverse=True)
    for m, strength, gap in seps[:5]:
        direction = "higher" if gap > 0 else "lower"
        print(f"  {m:<14} winners run {direction:<7} (strength {strength:.2f})")

    # Failure rate by kind.
    print("\nFailure rate by category:")
    for kind, g in df.groupby("kind"):
        fr = 100 * (~g["win"]).mean()
        print(f"  {kind:<12} n={len(g):<5} fail {fr:5.1f}%")


def report_misses(df: pd.DataFrame, gain_pct: float):
    print("\n" + "=" * 68)
    print(f"TASK 2 — Winners we MISSED (stocks that gained +{gain_pct:.0f}% next day)")
    print("=" * 68)
    if df is None or df.empty:
        print("No winners found.")
        return

    total = len(df)
    caught = df[df["shortlisted"]]
    missed = df[~df["shortlisted"]]
    print(f"\nAll +{gain_pct:.0f}% winners: {total}")
    print(f"  Caught (shortlisted): {len(caught)} ({100*len(caught)/total:.1f}%)")
    print(f"  Missed:               {len(missed)} ({100*len(missed)/total:.1f}%)")

    if missed.empty:
        return

    print("\nWhy we missed them:")
    reason_labels = {
        "downtrend": "Below EMA50/EMA200 — not an uptrend (out of scope)",
        "far_from_high": ">6% below its 20d high — ran from far back",
        "not_near_high": "In zone by price but <88% of its own high",
        "weak_flow": "Near high but CMF negative (no accumulation)",
        "rsi_hot_no_vol": "RSI hot (>72) without momentum volume",
        "too_quiet": "Near high, good flow, but volume too quiet",
        "in_zone_other": "In zone, passed sub-gates loosely — edge case",
    }
    counts = missed["miss_reason"].value_counts()
    for reason, c in counts.items():
        avg = missed[missed["miss_reason"] == reason]["max_gain"].mean()
        label = reason_labels.get(reason, reason)
        print(f"  {c:>5} ({100*c/len(missed):4.1f}%)  avg +{avg:4.1f}%  {label}")

    # How many misses were "close" — in trend and within the zone?
    close = missed[missed["miss_reason"].isin(
        ["not_near_high", "weak_flow", "rsi_hot_no_vol", "too_quiet", "in_zone_other"])]
    print(f"\n'Close misses' (in trend + within 6% zone): {len(close)} "
          f"({100*len(close)/len(missed):.1f}% of misses)")
    print("These are the actionable ones — a filter tweak could catch them.")

    # Sector-class breakdown of close misses.
    if not close.empty:
        print("\nClose misses by sector class:")
        for k, g in close.groupby("sector_klass"):
            lab = k if k else "(not accumulating)"
            print(f"  {lab:<24} {len(g):>4}  avg +{g['max_gain'].mean():.1f}%")


def main():
    ap = argparse.ArgumentParser(description="For-Tom pattern analysis")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--start", type=str)
    ap.add_argument("--end", type=str)
    ap.add_argument("--gain", type=float, default=3.0, help="Win threshold %%")
    ap.add_argument("--dump", type=str, help="Save raw rows to CSV prefix")
    args = ap.parse_args()

    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days)

    shortlisted, winners = run(start_date, end_date, args.gain)

    if args.dump:
        if shortlisted is not None and not shortlisted.empty:
            shortlisted.to_csv(f"{args.dump}_shortlisted.csv", index=False)
        if winners is not None and not winners.empty:
            winners.to_csv(f"{args.dump}_winners.csv", index=False)
        print(f"\nRaw rows saved with prefix '{args.dump}'.")

    report_failures(shortlisted, args.gain)
    report_misses(winners, args.gain)


if __name__ == "__main__":
    main()
