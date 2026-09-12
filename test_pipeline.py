"""
Verify the compute path on synthetic bhavcopy-shaped data.

The sandbox cannot reach nseindia.com, so this exercises everything downstream
of the fetch: cleaning, per-stock metrics, sector aggregation, gates, backtest.
It also plants the specific edge cases that silently corrupt real runs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel as pnl
import scan as sc
import stocks as stk

rng = np.random.default_rng(7)

SECTORS = {
    "Paper & Paper Products": 6,
    "Refineries & Marketing": 6,
    "Commodity Chemicals": 7,
    "Pharmaceuticals": 8,
    "Private Sector Bank": 6,
}
N_DAYS = 120


def make_data():
    dates = pd.bdate_range("2026-03-16", periods=N_DAYS)
    rows, smap = [], []

    for sector, n in SECTORS.items():
        # One sector gets a planted crossing: quiet, then expansion on breadth.
        planted = sector == "Refineries & Marketing"
        for k in range(n):
            sym = f"{sector[:4].upper().replace(' ', '')}{k:02d}"
            smap.append({"symbol": sym, "basic_industry": sector})
            price = 100.0 + rng.normal(0, 5)
            base_vol = rng.uniform(2e5, 2e6)

            for i, d in enumerate(dates):
                drift = 0.0006
                vol_mult = 1.0
                if planted and i >= N_DAYS - 6:
                    drift = 0.012          # markup
                    vol_mult = 2.8         # turnover expansion
                elif planted and i >= N_DAYS - 30:
                    drift = 0.0015         # quiet accumulation
                    vol_mult = 0.75

                ret = rng.normal(drift, 0.018)
                prev = price
                price = max(1.0, price * (1 + ret))
                spread = abs(rng.normal(0, 0.012)) + 0.004
                high = max(prev, price) * (1 + spread)
                low = min(prev, price) * (1 - spread)
                vol = base_vol * vol_mult * rng.uniform(0.6, 1.5)
                dp = float(np.clip(rng.normal(45 if not planted else 52, 8), 5, 98))

                # --- planted edge cases -----------------------------------
                if k == 0 and i == 40:
                    high = low = price          # circuit lock: high == low
                if k == 1 and i == 55:
                    dp = np.nan                 # missing delivery ("-" in the file)
                if k == 2 and i == 70:
                    prev = price * 10           # 1:10 split (NSE adjusts prev_close)

                rows.append({
                    "symbol": sym, "series": "EQ", "date": d,
                    "prev_close": prev, "open": prev, "high": high, "low": low,
                    "last": price, "close": price, "vwap": (high + low) / 2,
                    "volume": vol, "turnover": vol * price / 1e5,
                    "trades": int(vol / 300), "deliv_qty": vol * (dp or 0) / 100,
                    "deliv_pct": dp,
                })

    # Junk that must be filtered out.
    for d in dates:
        rows.append({"symbol": "SMEJUNK", "series": "SM", "date": d,
                     "prev_close": 10, "open": 10, "high": 10, "low": 10,
                     "last": 10, "close": 10, "vwap": 10, "volume": 100,
                     "turnover": 0.01, "trades": 1, "deliv_qty": 100,
                     "deliv_pct": 100})
        rows.append({"symbol": "TINYCAP", "series": "EQ", "date": d,
                     "prev_close": 5, "open": 5, "high": 5, "low": 5,
                     "last": 5, "close": 5, "vwap": 5, "volume": 50,
                     "turnover": 0.002, "trades": 1, "deliv_qty": 50,
                     "deliv_pct": 100})
    smap += [{"symbol": "SMEJUNK", "basic_industry": "Paper & Paper Products"},
             {"symbol": "TINYCAP", "basic_industry": "Paper & Paper Products"}]

    return pd.DataFrame(rows), pd.DataFrame(smap)


def main():
    raw, smap = make_data()
    print(f"synthetic raw: {len(raw):,} rows, {raw['symbol'].nunique()} symbols")

    stocks, p = pnl.build(raw, smap)
    print(f"after clean:   {stocks['symbol'].nunique()} symbols, "
          f"{p['sector'].nunique()} sectors, {p['date'].nunique()} sessions")

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")
        ok &= bool(cond)

    print("\n--- cleaning ---")
    check("SME series dropped", "SMEJUNK" not in set(stocks["symbol"]))
    check("illiquid dropped", "TINYCAP" not in set(stocks["symbol"]))

    print("\n--- per-stock metrics ---")
    check("MFM finite everywhere", np.isfinite(stocks["mfm"]).all(),
          f"(min {stocks['mfm'].min():.2f}, max {stocks['mfm'].max():.2f})")
    check("MFM within [-1, 1]", stocks["mfm"].abs().max() <= 1.0 + 1e-9)
    check("CMF has no inf", not np.isinf(stocks["cmf"].dropna()).any())
    check("CMF within [-1, 1]", stocks["cmf"].dropna().abs().max() <= 1.0 + 1e-9)
    split_ret = stocks.loc[stocks["ret"].abs() > 0.6, "ret"]
    check("split return suppressed, not propagated", split_ret.empty,
          f"(max |ret| = {stocks['ret'].abs().max():.3f})")
    check("delivery quality computed", stocks["deliv_quality"].notna().sum() > 0,
          f"({stocks['deliv_quality'].notna().sum():,} rows)")

    print("\n--- sector panel ---")
    check("T populated", p["T"].notna().sum() > 0)
    check("T_rel median ~1 per day",
          abs(p.groupby("date")["T_rel"].median().dropna().median() - 1.0) < 0.05)
    check("deliv_quality_rel median ~1 per day",
          abs(p.groupby("date")["deliv_quality_rel"].median().dropna().median() - 1.0) < 0.05)
    check("B within [-1, 1]", p["B"].dropna().abs().max() <= 1.0 + 1e-9)
    check("top_share within (0, 1]",
          (p["top_share"].dropna() > 0).all() and (p["top_share"].dropna() <= 1).all())
    check("sector return finite", np.isfinite(p["ret"].dropna()).all())

    print("\n--- classification ---")
    res = sc.classify(p)
    print(res[["sector", "klass", "T", "T_rel", "B", "deliv_quality_rel",
               "rs_chg_5", "top_share", "note"]].round(2).to_string(index=False))
    planted = res[res["sector"] == "Refineries & Marketing"]
    check("planted sector not idle", not planted.empty
          and planted.iloc[0]["klass"] in
          {"CROSSING", "CROSSING_UNVERIFIED", "PULLBACK", "BASE"},
          f"(got {planted.iloc[0]['klass'] if not planted.empty else 'missing'})")

    print("\n--- backtest ---")
    flags = sc.backtest(p, min_history=30)
    print(f"  {len(flags):,} flags across {flags['date'].nunique()} dates")
    check("forward returns attached", flags["fwd_5"].notna().sum() > 0,
          f"({flags['fwd_5'].notna().sum():,} with fwd_5)")
    rates = sc.base_rates(flags)
    print(rates[["klass", "n", "fwd_5", "base_5", "edge_5"]].round(4).to_string(index=False))

    ok &= test_coil()
    ok &= test_robustness()

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


def test_coil():
    """Stock-level coil scan on the same synthetic harness."""
    raw, smap = make_data()
    stocks, _ = pnl.build(raw, smap)
    s = stk.add_indicators(stocks)

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")
        ok &= bool(cond)

    print("\n--- coil scan ---")
    need = ["ema10", "ema20", "ema50", "ema200", "rsi", "atr", "pos_hi",
            "trigger", "to_trigger", "vol_ratio", "range20", "contraction",
            "base_days", "deliv_quality_rel", "ext_ema20"]
    missing = [c for c in need if c not in s.columns]
    check("indicators attached", not missing, f"(missing {missing})" if missing else "")
    check("RSI in [0, 100]", s["rsi"].dropna().between(0, 100).all())
    check("ATR finite", np.isfinite(s["atr"].dropna()).all())
    check("deliv_quality_rel median ~1 per day",
          abs(s.groupby("date")["deliv_quality_rel"].median().dropna().median() - 1.0) < 0.05)

    # The split-adjusted series must not inherit the 1:10 crash. A raw close
    # drop of that size would wreck every EMA and pos_hi.
    adj_ret = s.groupby("symbol")["adj"].pct_change()
    check("adj series ignores the planted split",
          (adj_ret.dropna().abs() < 0.6).all(),
          f"(max |adj ret| = {adj_ret.abs().max():.3f})")

    hits = stk.scan(s)
    miss = stk.near_miss(s)
    check("scan returns coil column or empty", hits.empty or "coil" in hits.columns)
    check("near_miss names the missing filter or empty",
          miss.empty or "missing" in miss.columns)

    ft = stk.forward_test(s, horizons=(5, 10), min_history=40)
    check("forward test produced rows", len(ft) > 0, f"({len(ft)} dates)")
    check("edge columns present",
          {"flag_5", "base_5", "edge_5"}.issubset(ft.columns))
    return ok


def test_robustness():
    """Flags log, delivery coverage, buy-funnel forward test, shape buy_ready."""
    import tempfile
    from pathlib import Path

    import buytest as bt
    import fetch
    import flagslog

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")
        ok &= bool(cond)

    print("\n--- delivery coverage ---")
    d1, d2 = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02")
    raw = pd.DataFrame({
        "date": [d1, d1, d2, d2],
        "deliv_pct": [40.0, 50.0, np.nan, np.nan],
        "source": ["sec_bhav", "sec_bhav", "udiff", "udiff"],
    })
    cov = fetch.delivery_coverage(raw)
    check("UDiFF day counted as missing", cov["missing"] == 1, f"(missing={cov['missing']})")
    check("as_of is the UDiFF day so as_of_ok is false", cov["as_of_ok"] is False)
    check("reason tagged udiff",
          any(x.get("reason") == "udiff" for x in cov["missing_detail"]))

    inferred = fetch._ensure_source(pd.DataFrame({
        "deliv_pct": [np.nan, np.nan],
    }))
    check("old cache without source inferred as udiff",
          (inferred["source"] == "udiff").all())

    print("\n--- flags log ---")
    tmp = Path(tempfile.mkdtemp())
    flagslog.FLAGS_PATH = tmp / "flags.csv"
    flagslog.SHAPE_PATH = tmp / "shape.csv"
    buys = pd.DataFrame({
        "symbol": ["AAA"], "sector": ["Banks"], "adj": [100.0], "coil": [60.0],
        "to_trigger": [0.03], "pos_hi": [0.96], "rsi": [55.0], "vol_ratio": [0.7],
        "range20": [0.08], "cmf": [0.1], "deliv_quality_rel": [1.1], "base_days": [40],
    })
    scan_rows = pd.DataFrame({
        "sector": ["Banks", "Paper"],
        "klass": ["PULLBACK", "NONE"],
        "buy_ready": [True, False],
        "T_rel": [1.1, 0.8],
        "B": [0.2, 0.0],
    })
    n_b, n_s = flagslog.append_day("2026-09-08", buys, scan_rows, {"Banks": "orderly"})
    check("wrote one buy-setup row", n_b == 1)
    check("wrote one actionable sector row", n_s == 1, f"(n_s={n_s})")
    n_b2, _ = flagslog.append_day("2026-09-08", buys, scan_rows, {"Banks": "orderly"})
    check("re-run of same as_of does not duplicate",
          len(flagslog.load_flags()) == 1, f"(n={len(flagslog.load_flags())})")
    flagslog.append_day("2026-09-09", buys, scan_rows, {"Banks": "orderly"})
    check("next day appends", len(flagslog.load_flags()) == 2)

    print("\n--- buy funnel forward test ---")
    raw, smap = make_data()
    stocks, p = pnl.build(raw, smap)
    stocks = stk.add_indicators(stocks)
    ft = bt.buy_forward_test(stocks, p, horizons=(5,), min_history=40)
    check("buytest produced rows", len(ft) > 0, f"({len(ft)} dates)")
    check("buytest has buy/coil/base/edge",
          {"buy_5", "coil_5", "base_5", "edge_5", "n_buys"}.issubset(ft.columns))
    sm = bt.summary(ft, horizons=(5,))
    check("summary has edge_vs_base and edge_vs_coil",
          {"edge_vs_base", "edge_vs_coil"}.issubset(sm.columns))

    print("\n--- shape buy_ready ---")
    dates = pd.bdate_range("2026-08-01", periods=20)
    rows = []
    for i, d in enumerate(dates):
        if i == 14:
            T, T_rel, B, q = 2.0, 1.6, 0.4, 4
        elif i > 14:
            T, T_rel, B, q = 0.9, 0.8, -0.1, 2
        else:
            T, T_rel, B, q = 0.7, 0.8, 0.1, 4
        rows.append({"date": d, "sector": "Test", "T": T, "T_rel": T_rel, "B": B,
                     "T_quiet_prior_5": q})
    _, rep = sc.shape_report(pd.DataFrame(rows))
    check("orderly pullback from quiet is buy_ready",
          rep["verdict"] == "orderly" and rep["buy_ready"] is True,
          f"(verdict={rep['verdict']}, buy_ready={rep['buy_ready']})")
    rows[-6]["T_quiet_prior_5"] = 1  # crossing day prior quiet fails
    # index 14 is the crossing
    hist2 = pd.DataFrame(rows)
    hist2.loc[hist2.index[14], "T_quiet_prior_5"] = 1
    _, rep2 = sc.shape_report(hist2)
    check("failed quiet check is not buy_ready",
          rep2["buy_ready"] is False, f"(verdict={rep2['verdict']})")
    why = sc.explain_setup(rep, {
        "symbol": "FOO", "sector": "Test", "pos_hi": 0.95,
        "to_trigger": 0.03, "vol_ratio": 0.7,
    })
    check("explain_setup names the sector and the trigger",
          "Test woke up" in why and "20-day high" in why and "FOO is coiled" in why,
          f"({why[:80]}…)")

    print("\n--- breakout confirmation ---")
    import breakouts as bo
    dates = pd.bdate_range("2026-08-03", periods=25)
    rows = []
    for i, d in enumerate(dates):
        vol = 1000.0 if i < 24 else 2000.0
        adj = 100.0 if i < 24 else 104.0
        rows.append({
            "date": d, "symbol": "FOO", "sector": "Banks",
            "adj": adj, "volume": vol, "cmf": 0.05,
        })
    hist = pd.DataFrame(rows)
    flags = pd.DataFrame([{
        "as_of": dates[23].strftime("%Y-%m-%d"),
        "symbol": "FOO", "sector": "Banks",
        "adj": 100.0, "to_trigger": 0.03, "trigger": 103.0,
    }])
    hit = bo.confirm_breakouts(flags, hist, dates[-1])
    check("close through trigger on 2× volume is a breakout",
          len(hit) == 1 and hit.iloc[0]["symbol"] == "FOO",
          f"(n={len(hit)})")
    if not hit.empty:
        check("explain_breakout names the flag and the trigger",
              "setup on" in hit.iloc[0]["why"] and "103" in hit.iloc[0]["why"])

    same = bo.confirm_breakouts(flags, hist, dates[23])
    check("same-day flag does not confirm", same.empty)

    quiet = hist.copy()
    quiet.loc[quiet.index[-1], "volume"] = 1000.0
    weak = bo.confirm_breakouts(flags, quiet, dates[-1])
    check("close through without volume expansion is not a breakout", weak.empty)

    late = hist.copy()
    late.loc[late.index[-1], "adj"] = 102.0
    miss = bo.confirm_breakouts(flags, late, dates[-1])
    check("close still under trigger is not a breakout", miss.empty)

    why_b = bo.explain_breakout({
        "symbol": "FOO", "flagged": "2026-09-08", "broke": "2026-09-09",
        "trigger": 103.0, "adj": 104.0, "vol_expand": 1.8, "ret_since_flag": 0.04,
    })
    check("explain_breakout is readable",
          "FOO was a setup" in why_b and "1.8×" in why_b)

    print("\n--- stock lookup ---")
    import analyze
    uni = pd.DataFrame({
        "symbol": ["UNIONBANK", "TCS", "INFY"],
        "sector": ["Public Sector Bank", "IT", "IT"],
    })
    check("exact match ignores case",
          analyze.resolve("unionbank", uni)["symbol"] == "UNIONBANK")
    check("spaces stripped",
          analyze.resolve("union bank", uni)["symbol"] == "UNIONBANK")
    check("unique prefix resolves",
          analyze.resolve("UNION", uni)["symbol"] == "UNIONBANK")
    miss = analyze.resolve("XYZNONE", uni)
    check("unknown is not found", miss["found"] is False)
    amb = analyze.resolve("I", pd.DataFrame({
        "symbol": ["INFY", "ITC"], "sector": ["IT", "FMCG"],
    }))
    check("ambiguous prefix returns suggestions",
          amb["found"] is False and len(amb["near"]) == 2)

    clean = {
        "adj": 100.0, "ema50": 90.0, "ema200": 80.0, "pos_hi": 0.96,
        "rsi": 55.0, "ext_ema20": 0.02, "vol_ratio": 0.7, "range20": 0.08,
        "cmf": 0.1, "deliv_quality_rel": 1.1, "base_days": 40.0, "contraction": 0.8,
    }
    fil = stk.evaluate_filters(clean)
    check("clean row passes every coil filter", all(f["ok"] for f in fil))
    loud = {**clean, "vol_ratio": 1.4}
    fil2 = stk.evaluate_filters(loud)
    check("volume expansion fails only the dry-up gate",
          sum(1 for f in fil2 if not f["ok"]) == 1
          and next(f["id"] for f in fil2 if not f["ok"]) == "vol")
    why_w = analyze.explain({
        "symbol": "FOO", "phase": "watching", "sector": "Banks",
        "sector_klass": "NONE",
        "metrics": {"pos_hi": 0.70, "to_trigger": 0.12},
        "filters": [{"ok": False, "text": "Volume dry-up (5-day ≤ 20-day)"}],
    })
    check("watching explain is not a buy", "Nothing here is a buy" in why_w)

    pot = {
        "phase": "potential", "sector_klass": "PULLBACK",
        "metrics": {
            "adj": 100.0, "trigger": 103.0, "atr": 2.0, "lo20": 94.0,
            "hi_n": 110.0, "ema20": 99.0, "cmf": 0.08, "range20": 0.09,
            "ext_ema20": 0.02,
        },
        "shape": {"verdict": "orderly"},
    }
    wait = analyze.trade_plan(pot, 100)
    check("coil entry is wait, buy is at trigger",
          wait["action"] == "wait" and wait["suggested_entry"] == 103.0,
          f"(action={wait['action']}, sug={wait['suggested_entry']})")
    buy = analyze.trade_plan(pot, 103)
    check("entry at trigger is buy", buy["action"] == "buy")
    check("stop sits under the base or 1.5 ATR",
          buy["stop"] is not None and buy["stop"] < buy["entry"])
    check("target is 2R above entry",
          buy["target"] is not None and buy["target"] > buy["entry"])

    live = {
        "phase": "broke_out", "sector_klass": "PULLBACK",
        "metrics": {
            "adj": 104.0, "trigger": 103.0, "atr": 2.0, "lo20": 94.0,
            "hi_n": 112.0, "ema20": 101.0, "cmf": 0.1, "vol_expand": 1.8,
            "ext_ema20": 0.03,
        },
        "shape": {"verdict": "orderly"},
    }
    hold = analyze.trade_plan(live, 103)
    check("working breakout from trigger is hold", hold["action"] == "hold")
    stopped = analyze.trade_plan({
        **live, "metrics": {**live["metrics"], "adj": 95.0},
    }, 103)
    check("close through the stop is sell", stopped["action"] == "sell")

    dates = pd.bdate_range("2026-08-03", periods=25)
    rows = []
    for i, d in enumerate(dates):
        vol = 1000.0 if i < 24 else 28000.0
        adj = 100.0 if i < 24 else 115.0
        trig = 103.0 if i < 24 else 115.0
        rows.append({
            "date": d, "symbol": "WHEELS", "sector": "Auto",
            "adj": adj, "volume": vol, "trigger": trig,
            "ema50": 90.0, "ema200": 80.0, "turnover": 200.0, "cmf": 0.4,
        })
    vb = bo.volume_breaks(pd.DataFrame(rows), dates[-1])
    check("volume break lists an unflagged thrust through the high",
          len(vb) == 1 and vb.iloc[0]["symbol"] == "WHEELS"
          and vb.iloc[0]["kind"] == "momentum",
          f"(n={len(vb)})")
    mom = analyze.explain({
        "symbol": "WHEELS", "phase": "momentum", "sector": "Auto",
        "sector_klass": "DISQUALIFIED",
        "metrics": {"vol_expand": 27.8, "rsi": 86.0},
        "filters": [],
    })
    check("momentum explain does not treat sector class as a veto",
          "not a veto" in mom and "volume break" in mom.lower())

    print("\n--- for tomorrow overlay ---")
    import tom as tomscan
    eod_dates = pd.bdate_range("2026-08-03", periods=25)
    eod_rows = []
    for i, d in enumerate(eod_dates):
        eod_rows.append({
            "date": d, "symbol": "AAA", "sector": "Banks",
            "adj": 100.0, "volume": 1000.0, "trigger": 103.0,
            "ema50": 90.0, "ema200": 80.0, "pos_hi": 0.96, "rsi": 55.0, "cmf": 0.1,
        })
    eod = pd.DataFrame(eod_rows)
    live = pd.DataFrame([{
        "symbol": "AAA", "ltp": 102.5, "volume": 1800.0, "pchange": 1.2,
    }])
    buys = pd.DataFrame([{"symbol": "AAA"}])
    ft = tomscan.for_tomorrow(eod, live, buys)
    check("live near-trigger setup is for tom",
          len(ft) == 1 and ft.iloc[0]["kind"] == "setup",
          f"(n={len(ft)} kind={None if ft.empty else ft.iloc[0]['kind']})")
    live2 = pd.DataFrame([{
        "symbol": "AAA", "ltp": 104.0, "volume": 3000.0, "pchange": 3.0,
    }])
    th = tomscan.for_tomorrow(eod, live2, buys)
    check("live through trigger is tagged through",
          not th.empty and th.iloc[0]["kind"] == "through")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
