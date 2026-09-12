"""
Forward-test the full buy funnel: sector shape ∩ coil.

`coiltest` measures the coil scan alone. This measures what the "Buy setup"
badge actually asserts — a coiled name inside a PULLBACK sector whose shape
checks all passed (quiet-to-loud crossing, then red days on lighter turnover).

For every historical session it rebuilds the sector panel classification, keeps
the buy-ready sectors, intersects them with that day's coil hits, and attaches
forward returns for three groups on the same dates:

    buy_N   the funnel output (coil ∩ buy-ready sector)
    coil_N  every coil hit that day (the wider net)
    base_N  every liquid stock that day (the base rate)

edge_N = buy_N − base_N. If it is not clearly above coil_N's own edge, the
sector overlay is adding nothing and the badge is just a coil rename.

This is slow — it reclassifies every sector on every date — so run it on a
long window occasionally, not in the request path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import scan as sc
import stocks as stk


def buy_forward_test(stocks: pd.DataFrame, panel: pd.DataFrame,
                     th: sc.Thresholds | None = None,
                     p: stk.CoilParams | None = None,
                     horizons=(5, 10, 20),
                     min_history: int = 210,
                     on_progress=None) -> pd.DataFrame:
    th = th or sc.Thresholds()
    p = p or stk.CoilParams()

    s = stocks.sort_values(["symbol", "date"]).copy()
    for h in horizons:
        s[f"fwd_{h}"] = s.groupby("symbol", sort=False)["tri"].transform(
            lambda x, h=h: x.shift(-h) / x - 1.0
        )

    dates = sorted(s["date"].unique())[min_history:]
    total = len(dates)
    rows = []
    for i, as_of in enumerate(dates, 1):
        if on_progress is not None:
            on_progress(i, total, as_of)

        panel_upto = panel[panel["date"] <= as_of]
        scan_rows = sc.classify(panel_upto, th, as_of=as_of)
        ready = sc.recommend_sectors(scan_rows, panel_upto)

        coil_hits = stk.scan(s[s["date"] <= as_of], p, as_of=as_of, top=10_000)
        buy_syms = (coil_hits[coil_hits["sector"].isin(ready)]["symbol"]
                    if not coil_hits.empty else pd.Series(dtype=object))

        day = s[s["date"] == as_of]
        rec = {
            "date": as_of,
            "n_ready_sectors": len(ready),
            "n_coil": int(len(coil_hits)),
            "n_buys": int(len(buy_syms)),
            "n_universe": int(len(day)),
        }
        for h in horizons:
            col = f"fwd_{h}"
            rec[f"buy_{h}"] = (day[day["symbol"].isin(buy_syms)][col].mean()
                              if len(buy_syms) else np.nan)
            rec[f"coil_{h}"] = (day[day["symbol"].isin(coil_hits["symbol"])][col].mean()
                               if not coil_hits.empty else np.nan)
            rec[f"base_{h}"] = day[col].mean()
            rec[f"edge_{h}"] = rec[f"buy_{h}"] - rec[f"base_{h}"]
        rows.append(rec)

    return pd.DataFrame(rows)


def summary(ft: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    """
    Collapse the per-date log into one row per horizon, weighting each day by
    how many buys it produced (a day with 5 setups should count more than a day
    with 1). Days with zero buys drop out of the buy/edge means but still count
    toward how often the funnel fires.
    """
    rows = []
    fired = ft[ft["n_buys"] > 0]
    for h in horizons:
        w = fired["n_buys"]
        buy = np.average(fired[f"buy_{h}"], weights=w) if len(fired) else np.nan
        base = np.average(fired[f"base_{h}"], weights=w) if len(fired) else np.nan
        coil_days = ft[ft["n_coil"] > 0]
        coil = (np.average(coil_days[f"coil_{h}"], weights=coil_days["n_coil"])
                if len(coil_days) else np.nan)
        rows.append({
            "horizon": h,
            "buy": buy,
            "coil": coil,
            "base": base,
            "edge_vs_base": buy - base,
            "edge_vs_coil": buy - coil,
        })
    return pd.DataFrame(rows)
