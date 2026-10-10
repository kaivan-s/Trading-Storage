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
    "Paper & Paper Products": 8,
    "Refineries & Marketing": 9,
    "Commodity Chemicals": 10,
    "Pharmaceuticals": 12,
    "Private Sector Bank": 8,
    # Under the 8-name floor: must not survive cleaning.
    "Thin Sector": 5,
}
# Long enough that the 200-EMA is actually defined: the coil gates read
# `adj > ema50 > ema200`, and ema200 now needs 200 observations before it
# returns anything, so a shorter panel can only ever test that the scan
# correctly refuses to flag.
N_DAYS = 260


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
    check("sector under 8 names has no panel row", "Thin Sector" not in set(p["sector"]))
    check("its stocks stay in the stock frame",
          "Thin Sector" in set(stocks["sector"]))

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
    ok &= test_episodes()
    ok &= test_fetch_guards()

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
                     "T_quiet_prior_5": q, "rs_chg_5": 1.0,
                     "deliv_quality_rel": 1.1, "n_adv": 5, "top_share": 0.3})
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

    print("\n--- pullback needs a verified crossing ---")

    def sector_hist(cross_top_share):
        out = []
        for i, d in enumerate(pd.bdate_range("2026-08-01", periods=20)):
            base = {"date": d, "sector": "Test", "B_deliv": 0.0, "cmf": 0.05,
                    "cmf_rel": 0.02, "cmf_rel_chg_5": 0.01, "rs": 60.0,
                    "rs_chg_5": 1.0, "deliv_quality": 1.0, "n_stocks": 10,
                    "B_green_10": 5, "T_quiet_10": 4}
            if i == 16:
                base.update(T=2.0, T_rel=1.6, B=0.4, T_quiet_prior_5=4,
                            deliv_quality_rel=1.1, n_adv=6,
                            top_share=cross_top_share)
            elif i > 16:
                base.update(T=1.2, T_rel=0.9, B=-0.1, T_quiet_prior_5=2,
                            deliv_quality_rel=1.05, n_adv=4, top_share=0.3)
            else:
                base.update(T=0.8, T_rel=0.9, B=0.1, T_quiet_prior_5=4,
                            deliv_quality_rel=1.0, n_adv=5, top_share=0.3)
            out.append(base)
        return pd.DataFrame(out)

    good = sc.classify_one(sector_hist(0.30), sc.Thresholds())
    check("lighter red after a verified crossing is PULLBACK",
          good["klass"] == "PULLBACK", f"(got {good['klass']})")
    narrow = sc.classify_one(sector_hist(0.70), sc.Thresholds())
    check("lighter red after a one-stock expansion is not PULLBACK",
          narrow["klass"] != "PULLBACK", f"(got {narrow['klass']})")
    _, rep3 = sc.shape_report(sector_hist(0.70))
    check("shape report ignores the unverified expansion",
          rep3["verdict"] == "no_crossing", f"(verdict={rep3['verdict']})")

    print("\n--- leaders at rest entry test ---")
    import position as pos
    as_of = pd.Timestamp("2026-09-01")
    universe = pd.DataFrame({
        "symbol": [f"U{i:02d}" for i in range(10)] + ["LEAD", "STEADY", "LAGGARD", "THIN", "DQ"],
        "date": as_of,
        "adj": 100.0,
        "med_turn60": [500.0] * 13 + [50.0, 500.0],
        "own_sessions": 300,
        "mom12_1": [0.05 * i for i in range(10)] + [0.90, 0.70, 0.01, 0.95, 0.85],
        "mom_vadj": [0.1 * i for i in range(10)] + [1.5, 2.5, 0.02, 3.0, 2.0],
    })
    coils = pd.DataFrame({
        "symbol": ["LEAD", "STEADY", "LAGGARD", "THIN", "DQ"],
        "sector_klass": ["PULLBACK", "NONE", "NONE", "BASE", "DISQUALIFIED"],
    })
    rest = pos.leaders_at_rest(coils, universe, as_of=as_of)
    names = list(rest["symbol"])
    check("bottom-of-market momentum excluded", "LAGGARD" not in names, f"({names})")
    check("sub-₹100L turnover excluded", "THIN" not in names, f"({names})")
    check("disqualified sector excluded", "DQ" not in names, f"({names})")
    check("ordered by risk-adjusted momentum",
          names == ["STEADY", "LEAD"], f"({names})")
    ann = pos.annotate_momentum(coils, universe, as_of=as_of)
    check("pool rows keep their momentum reading",
          ann["mom12_1"].notna().all() and ann["mom_pct"].notna().sum() == 4)

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
    check("watching explain reports no setup", "No structural setup here" in why_w)

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

    return ok


def test_episodes():
    """
    Drive the episode state machine through every transition by hand.

    Synthetic rather than measured on purpose: the replay over real data tells
    us the distribution of outcomes, but only a scripted sequence proves that
    a three-session gap is bridged, that a one-day dip is not a failed
    breakout, and that a breakout on the same day a base leaves the filters is
    still reported as a breakout.
    """
    import episodes as eps

    print("\n--- base episodes ---")
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")
        ok &= bool(cond)

    def day(qualifies, price, vol=1.0, trigger=110.0, **flags):
        """One symbol's cross-section row, as gate_flags would produce it."""
        row = {
            "symbol": "AAA", "sector": "Banks", "adj": float(price),
            "trigger": trigger, "vol_ratio": vol, "mom12_1": 0.3,
            "n_fail": 0 if qualifies else 1,
        }
        for c in stk.FLAG_COLS:
            row[c] = True
        if not qualifies:
            row[flags.get("lost", "f_vol")] = False
        return pd.DataFrame([row])

    dates = pd.bdate_range("2025-01-01", periods=40)

    # --- a base that breaks out on heavy volume ---
    e = []
    eps.advance(e, day(True, 100.0), dates[0])
    check("episode opens on first qualifying session",
          len(e) == 1 and e[0]["state"] == "basing", f"(n={len(e)})")
    check("trigger frozen at entry", e[0]["entry_trigger"] == 110.0)
    eps.advance(e, day(True, 101.0), dates[1])
    eps.advance(e, day(True, 112.0, vol=2.0), dates[2])
    check("close through the level triggers",
          e[0]["state"] == "triggered", f"(state={e[0]['state']})")
    check("heavy volume flagged", e[0]["trigger_vol"] is True)

    # --- the frozen level is what matters, not the rolling one ---
    e = []
    eps.advance(e, day(True, 100.0, trigger=110.0), dates[0])
    eps.advance(e, day(True, 112.0, trigger=125.0), dates[1])
    check("a rising rolling high cannot move the target",
          e[0]["state"] == "triggered", f"(state={e[0]['state']})")

    # --- one dip back inside is a retest, two closes is a failure ---
    e = []
    eps.advance(e, day(True, 100.0), dates[0])
    eps.advance(e, day(True, 112.0), dates[1])
    eps.advance(e, day(False, 108.0), dates[2])
    check("one close back inside is not a failure",
          e[0]["state"] == "triggered", f"(state={e[0]['state']})")
    eps.advance(e, day(False, 107.0), dates[3])
    check("two straight closes inside is a failure",
          e[0]["state"] == "failed", f"(state={e[0]['state']})")
    check("failure resolves the episode", e[0]["resolved_on"] is not None)

    # --- a three-session gap is bridged, a four-session gap is not ---
    e = []
    eps.advance(e, day(True, 100.0), dates[0])
    for i in (1, 2, 3):
        eps.advance(e, day(False, 100.0), dates[i])
    check("still basing through a three-session gap",
          e[0]["state"] == "basing" and len(e) == 1,
          f"(state={e[0]['state']} gap={e[0]['gap']})")
    eps.advance(e, day(True, 100.0), dates[4])
    check("bridged gap keeps one episode, streak continues",
          len(e) == 1 and e[0]["coil_sessions"] == 2,
          f"(n={len(e)} sessions={e[0]['coil_sessions']})")

    e = []
    eps.advance(e, day(True, 100.0), dates[0])
    for i in (1, 2, 3, 4):
        eps.advance(e, day(False, 100.0, lost="f_trend"), dates[i])
    check("a four-session gap drops the base",
          e[0]["state"] == "dropped", f"(state={e[0]['state']})")
    check("the lost filter is recorded",
          e[0]["lost_gates"] == "trend" and "uptrend" in e[0]["reason"].lower(),
          f"(lost={e[0]['lost_gates']!r} reason={e[0]['reason']!r})")
    eps.advance(e, day(True, 100.0), dates[5])
    check("requalifying after a real gap starts a new episode",
          len(e) == 2 and e[1]["state"] == "basing", f"(n={len(e)})")

    # --- a dropped base still gets credit for a late breakout ---
    e = []
    eps.advance(e, day(True, 100.0), dates[0])
    for i in range(1, 5):
        eps.advance(e, day(False, 100.0), dates[i])
    check("dropped before the late move", e[0]["state"] == "dropped")
    eps.advance(e, day(False, 115.0), dates[5])
    check("a breakout after dropping off is still a breakout",
          e[0]["state"] == "triggered", f"(state={e[0]['state']})")

    # --- price events beat list membership on the same session ---
    e = []
    eps.advance(e, day(True, 100.0), dates[0])
    for i in (1, 2, 3):
        eps.advance(e, day(False, 100.0), dates[i])
    eps.advance(e, day(False, 118.0, lost="f_vol"), dates[4])
    check("breaking out on the session it leaves is not a drop-off",
          e[0]["state"] == "triggered", f"(state={e[0]['state']})")

    # --- breakdown ---
    e = []
    eps.advance(e, day(True, 100.0), dates[0])
    eps.advance(e, day(True, 92.0), dates[1])
    check("an 8% fall from entry breaks down",
          e[0]["state"] == "broke_down", f"(state={e[0]['state']})")
    e = []
    eps.advance(e, day(True, 100.0), dates[0])
    eps.advance(e, day(True, 95.0), dates[1])
    check("a 5% fall does not break down",
          e[0]["state"] == "basing", f"(state={e[0]['state']})")

    # --- nothing happens for the whole horizon ---
    e = []
    for i in range(eps.HORIZON + 2):
        eps.advance(e, day(True, 100.0), dates[i])
    check("a base that never resolves goes stale",
          e[0]["state"] == "stale" and e[0]["resolved_on"] is not None,
          f"(state={e[0]['state']} age={e[0]['age']})")

    # --- tagging a displayed list ---
    e = []
    eps.advance(e, day(True, 100.0), dates[0])
    eps.advance(e, day(False, 100.0), dates[1])
    eps.advance(e, day(True, 100.0), dates[2])
    df = eps.frame(e)
    rows = pd.DataFrame([{"symbol": "AAA", "coil_days": 1}])
    tagged = eps.tag(rows, df, dates[2])
    check("bridged age overrides a reset coil_days",
          int(tagged.iloc[0]["episode_days"]) == 2,
          f"(coil_days=1 episode_days={tagged.iloc[0]['episode_days']})")
    check("an older base is not labelled new",
          not bool(tagged.iloc[0]["episode_new"]))

    # --- seeding from the published list, which is the live path ---
    def two(qualifies_b):
        """Cross-section with AAA qualifying and BBB optionally qualifying."""
        d = pd.concat([day(True, 100.0), day(qualifies_b, 100.0)],
                      ignore_index=True)
        d.loc[1, "symbol"] = "BBB"
        return d

    e = []
    eps.advance(e, two(True), dates[0], eligible={"AAA"})
    check("eligible restricts which names may open",
          len(e) == 1 and e[0]["symbol"] == "AAA",
          f"({[x['symbol'] for x in e]})")

    e = []
    eps.advance(e, two(False), dates[0], eligible={"AAA", "BBB"})
    check("a published name opens even if it fails the gates",
          len(e) == 2,
          "(the list is the authority on what was shown)")

    e = []
    eps.advance(e, two(True), dates[0], eligible=set())
    check("an empty published list opens nothing",
          len(e) == 0,
          "(None would mean unrestricted -- a real bug once)")

    e = []
    eps.advance(e, two(True), dates[0], eligible={"AAA"})
    eps.advance(e, two(True), dates[1], eligible={"AAA", "BBB"})
    check("a later list adds only the new entrant",
          len(e) == 2 and e[1]["symbol"] == "BBB" and e[1]["age"] == 0,
          f"(n={len(e)})")
    check("the carried-forward episode advanced, not restarted",
          e[0]["age"] == 1 and e[0]["coil_sessions"] == 2,
          f"(age={e[0]['age']} sessions={e[0]['coil_sessions']})")

    # --- the digest counts today, not the backlog ---
    dg = eps.digest(df, dates[2])
    check("digest reports a quiet session as quiet",
          dg["new"] == 0 and dg["triggered"] == 0 and dg["live"] == 1,
          f"({dg})")
    check("digest counts a new base on the day it appears",
          eps.digest(df, dates[0])["new"] == 1)

    return ok



def test_fetch_guards():
    """
    A holiday must not become a session.

    NSE answers a request for a non-trading date with the PREVIOUS session's
    bhavcopy rather than an error, so the only things standing between that
    and a duplicated day in the panel are the date check in the parsers and
    the repeat-session guard that repairs caches written before it existed.
    Both are checked here because the failure is silent: a duplicated session
    has zero returns everywhere and quietly distorts every range, volatility
    and session-counting indicator downstream.
    """
    from datetime import date as _date
    import fetch

    print("\n--- fetch guards ---")
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")
        ok &= bool(cond)

    header = ("SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE,"
              " LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY,"
              " TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER")

    def csv_for(stamp):
        rows = [header]
        for sym in ("AAA", "BBB"):
            rows.append(f"{sym}, EQ, {stamp}, 100, 100, 101, 99, 100, 100,"
                        f" 100, 1000, 10, 50, 500, 50.0")
        return "\n".join(rows) + "\n"

    # The file agrees with the request: accepted.
    got = fetch._parse_sec_bhav(csv_for("11-Sep-2026"), _date(2026, 9, 11))
    check("a file dated as requested parses", len(got) == 2, f"({len(got)} rows)")

    # The file is the previous session's: rejected, which is what a holiday
    # request actually returns from NSE.
    try:
        fetch._parse_sec_bhav(csv_for("11-Sep-2026"), _date(2026, 9, 14))
        check("a stale file is rejected", False, "(it was accepted)")
    except fetch.StaleBhavcopy as exc:
        check("a stale file is rejected", True, f"({exc})")

    # StaleBhavcopy has to stay catchable by the existing fallback chain.
    check("StaleBhavcopy is a RuntimeError",
          issubclass(fetch.StaleBhavcopy, RuntimeError))

    # Repeat-session guard. Dates far in the past so no cache file exists and
    # the purge step is a no-op.
    base = pd.DataFrame({
        "symbol": ["AAA", "BBB"] * 3,
        "close": [10.0, 20.0, 10.0, 20.0, 11.0, 21.0],
        "date": pd.to_datetime(
            ["1999-01-04", "1999-01-04", "1999-01-05", "1999-01-05",
             "1999-01-06", "1999-01-06"]),
    })
    kept = fetch.drop_repeat_sessions(base, verbose=False)
    check("a session identical to the one before it is dropped",
          sorted(str(d.date()) for d in kept["date"].unique())
          == ["1999-01-04", "1999-01-06"],
          f"({[str(d.date()) for d in sorted(kept['date'].unique())]})")

    # Two genuinely different sessions must both survive, even when close.
    near = pd.DataFrame({
        "symbol": ["AAA", "BBB"] * 2,
        "close": [10.0, 20.0, 10.0, 20.01],
        "date": pd.to_datetime(
            ["1999-02-01", "1999-02-01", "1999-02-02", "1999-02-02"]),
    })
    check("a nearly-identical but real session is kept",
          fetch.drop_repeat_sessions(near, verbose=False)["date"].nunique() == 2)

    # Only the copy goes, never the original.
    check("the earlier session is the one retained",
          str(pd.Timestamp(kept["date"].min()).date()) == "1999-01-04")

    # Negative cache. The age guard is the subtle half: a request for a
    # session whose file has not published yet is answered with the previous
    # one, which is indistinguishable from a holiday. Recording that would
    # blind the loader to a real session permanently.
    from datetime import date as _d, timedelta as _td
    saved = fetch._no_session
    try:
        fetch._no_session = set()
        recent = _d.today() - _td(days=1)
        fetch._mark_no_session(recent)
        check("a date younger than the guard is not recorded",
              recent.isoformat() not in fetch._no_session_set())

        old_holiday = _d(2026, 1, 26)
        fetch._mark_no_session(old_holiday)
        check("a long-past holiday is recorded",
              old_holiday.isoformat() in fetch._no_session_set())
    finally:
        fetch._no_session = saved

    return ok



if __name__ == "__main__":
    raise SystemExit(main())
