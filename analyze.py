"""
Single-name read-through of the same gates the dashboard already runs.

This is not a new model and not a buy call. It resolves a symbol, then says
where that name sits in the existing pipeline: coil filters, sector shape,
logged setup, confirmed breakout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import flagslog
import scan as sc
import stocks as stk

PHASES = {
    "broke_out": "Broke out",
    "potential": "Potential",
    "coiled": "Coiled",
    "near_miss": "Near miss",
    "at_high": "At the high",
    "momentum": "Volume break",
    "watching": "Watching",
}

def _norm_q(q: str) -> str:
    return "".join(ch for ch in (q or "").upper() if ch.isalnum())


def resolve(query: str, universe: pd.DataFrame) -> dict:
    """
    Match a typed name to one symbol. Exact and unique prefix win;
    otherwise return suggestions. `universe` needs a `symbol` column.
    """
    raw = (query or "").strip()
    q = _norm_q(raw)
    empty = {"found": False, "query": raw, "symbol": None, "near": []}
    if not q or universe is None or universe.empty:
        return empty

    u = universe.copy()
    u["symbol"] = u["symbol"].astype(str)
    u["_key"] = u["symbol"].map(_norm_q)
    u = u.drop_duplicates(subset=["_key"])

    exact = u[u["_key"] == q]
    if len(exact) == 1:
        row = exact.iloc[0]
        return {"found": True, "query": raw, "symbol": row["symbol"],
                "near": [], "sector": row["sector"] if "sector" in row.index else None}

    pref = u[u["_key"].str.startswith(q, na=False)]
    if len(pref) == 1:
        row = pref.iloc[0]
        return {"found": True, "query": raw, "symbol": row["symbol"],
                "near": [], "sector": row["sector"] if "sector" in row.index else None}

    hits = pref if not pref.empty else u[u["_key"].str.contains(q, na=False)]
    near = []
    if not hits.empty:
        cols = [c for c in ["symbol", "sector"] if c in hits.columns]
        near = hits[cols].head(8).to_dict(orient="records")
    return {**empty, "near": near}


def _iso(ts) -> str | None:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)) or pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _f(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) or pd.isna(v):
        return None
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def _vol_expand(hist: pd.DataFrame, as_of) -> float | None:
    h = hist[hist["date"] <= pd.Timestamp(as_of)].sort_values("date")
    if len(h) < 12:
        return None
    today = h.iloc[-1]
    base = h.iloc[:-1]["volume"].tail(20).mean()
    if pd.isna(base) or float(base) <= 0:
        return None
    return float(today["volume"]) / float(base)


def explain(report: dict) -> str:
    """Plain-language read of one name. Not a market order."""
    symbol = report.get("symbol") or "This name"
    phase = report.get("phase")
    sector = report.get("sector") or "its sector"
    klass = report.get("sector_klass") or "unclassified"
    metrics = report.get("metrics") or {}
    filters = report.get("filters") or []
    failed = [f["text"] for f in filters if not f.get("ok")]
    trig = metrics.get("to_trigger")
    vol_x = metrics.get("vol_expand")
    pos = metrics.get("pos_hi")

    if phase == "broke_out":
        br = report.get("breakout") or {}
        if br.get("why"):
            return br["why"] + " Still check CMF and delivery — this is confirmation, not an execution level."
        return (
            f"{symbol} has closed through its trigger on rising volume. "
            "That is the confirmation the setup was waiting for; the structure "
            "is defined at the trigger, not above it."
        )

    if phase == "potential":
        why = (report.get("setup") or {}).get("why")
        if why:
            return why
        return (
            f"{symbol} is a potential breakout: coiled under the 20-day high "
            f"inside a clean {sector} pullback. The structural trigger sits "
            "above today's close, so nothing is confirmed yet."
        )

    if phase == "coiled":
        return (
            f"{symbol} passes the coil filters — tight, quiet, near the high — "
            f"but {sector} is {klass}, not a clean pullback. It is on the coil "
            "list, not a potential breakout, until the sector shape is ready."
        )

    if phase == "near_miss":
        miss = next((f["text"] for f in filters if not f.get("ok")), "one filter")
        extra = ""
        if trig is not None:
            extra = f" Distance to the 20-day high is {trig * 100:.1f}%."
        return (
            f"{symbol} fails only one coil gate: {miss[0].lower() + miss[1:]}. "
            f"Everything else looks like a base.{extra} Not a setup."
        )

    if phase == "momentum":
        vol_s = f"{vol_x:.1f}×" if vol_x is not None else "rising"
        rsi = metrics.get("rsi")
        late = rsi is not None and rsi > 68
        extra = (
            f" RSI is {rsi:.0f} — the first push is already spent, so this is "
            "a late-stage break, not a coil."
            if late else
            " Coil filters fail on purpose: this is the loud day, not the quiet base."
        )
        return (
            f"{symbol} closed through the 20-day high on {vol_s} the 20-day "
            f"average volume. That is a volume break, not a coil. "
            f"{sector} being {klass} is the industry, not a veto on this print."
            f"{extra}"
        )

    if phase == "at_high":
        vol_s = f" Today's volume is {vol_x:.1f}× the 20-day average." if vol_x else ""
        return (
            f"{symbol} is already at or through the 20-day high, so it is past "
            f"the coil phase.{vol_s} Without a real volume expansion this is "
            "just sitting on the high, not a break."
        )

    bits = []
    if pos is not None:
        bits.append(f"sitting at {pos * 100:.0f}% of its 85-day high")
    if trig is not None:
        bits.append(f"{trig * 100:.1f}% from the 20-day high")
    where = (", ".join(bits) + ". ") if bits else ""
    fail_s = ""
    if failed:
        fail_s = " Fails: " + "; ".join(x[0].lower() + x[1:] for x in failed[:3]) + "."
    return (
        f"{symbol} is not in a coil or a confirmed breakout. {where}"
        f"{sector} is {klass}.{fail_s} No structural setup here."
    )



def build(query: str, *, stocks: pd.DataFrame, panel: pd.DataFrame,
          scan_rows: pd.DataFrame, buys: pd.DataFrame,
          breakouts: pd.DataFrame, as_of, entry=None) -> dict:
    """Full single-name report from an already-loaded engine snapshot."""
    as_of = pd.Timestamp(as_of) if as_of is not None else stocks["date"].max()
    latest = stocks[stocks["date"] == stocks["date"].max()]
    uni_cols = [c for c in ["symbol", "sector"] if c in latest.columns]
    hit = resolve(query, latest[uni_cols] if uni_cols else latest)

    if not hit["found"]:
        return {**hit, "as_of": _iso(as_of)}

    symbol = hit["symbol"]
    hist = stocks[stocks["symbol"] == symbol].sort_values("date")
    if hist.empty:
        return {"found": False, "query": query, "symbol": symbol,
                "near": [], "as_of": _iso(as_of)}

    row = hist[hist["date"] <= as_of]
    row = row.iloc[-1] if not row.empty else hist.iloc[-1]
    sector = row["sector"] if "sector" in row.index and pd.notna(row["sector"]) else None

    p = stk.CoilParams()
    filters = stk.evaluate_filters(row, p)
    n_fail = sum(1 for f in filters if not f["ok"])
    med_to = hist["turnover"].median() if "turnover" in hist.columns else np.nan
    liquid = bool(pd.notna(med_to) and med_to >= p.min_median_turnover_lacs)

    prior = hist[hist["date"] < row["date"]]
    prior_trigger = (prior.iloc[-1]["trigger"]
                     if not prior.empty and "trigger" in prior.columns
                     else stk._cell(row, "trigger"))
    adj = stk._cell(row, "adj")
    closed_through = (adj is not None and prior_trigger is not None
                      and pd.notna(prior_trigger) and adj >= float(prior_trigger))
    vol_x = _vol_expand(hist, row["date"])
    window = hist[hist["date"] <= row["date"]].tail(20)
    lo20 = (float(window["adj_low"].min())
            if "adj_low" in window.columns and window["adj_low"].notna().any()
            else None)
    hi_n = stk._cell(row, "hi_n")
    atr = stk._cell(row, "atr")

    setup_row = None
    if buys is not None and not buys.empty:
        m = buys[buys["symbol"] == symbol]
        if not m.empty:
            setup_row = m.iloc[0].to_dict()

    broke_row = None
    if breakouts is not None and not breakouts.empty:
        m = breakouts[breakouts["symbol"] == symbol]
        if not m.empty:
            broke_row = m.iloc[-1].to_dict()

    flags = flagslog.load_flags()
    flag_hist = []
    if flags is not None and not flags.empty:
        fh = flags[flags["symbol"].astype(str) == symbol].sort_values("as_of")
        if not fh.empty:
            flag_hist = [
                {"as_of": str(r["as_of"]),
                 "trigger": _f(r["trigger"]) if "trigger" in r.index else None,
                 "to_trigger": _f(r["to_trigger"]) if "to_trigger" in r.index else None}
                for _, r in fh.tail(5).iterrows()
            ]

    klass, note, shape = None, None, None
    if sector and panel is not None and not panel.empty:
        ph = panel[panel["sector"] == sector].sort_values("date")
        if not ph.empty:
            _, shape = sc.shape_report(ph)
    if sector and scan_rows is not None and not scan_rows.empty:
        sr = scan_rows[scan_rows["sector"] == sector]
        if not sr.empty:
            klass = sr.iloc[0].get("klass")
            note = sr.iloc[0].get("note")

    buy_ready = bool((shape or {}).get("buy_ready") and klass == "PULLBACK")

    if broke_row:
        phase = "broke_out"
    elif setup_row:
        phase = "potential"
    elif closed_through and vol_x is not None and vol_x >= 1.5:
        phase = "momentum"
    elif n_fail == 0 and liquid:
        phase = "coiled"
    elif closed_through:
        phase = "at_high"
    elif n_fail == 1 and liquid:
        phase = "near_miss"
    else:
        phase = "watching"

    metrics = {
        "adj": _f(stk._cell(row, "adj")),
        "close": _f(stk._cell(row, "close")),
        "trigger": _f(stk._cell(row, "trigger")),
        "prior_trigger": _f(prior_trigger),
        "to_trigger": _f(stk._cell(row, "to_trigger")),
        "pos_hi": _f(stk._cell(row, "pos_hi")),
        "rsi": _f(stk._cell(row, "rsi")),
        "vol_ratio": _f(stk._cell(row, "vol_ratio")),
        "vol_expand": _f(vol_x),
        "range20": _f(stk._cell(row, "range20")),
        "contraction": _f(stk._cell(row, "contraction")),
        "cmf": _f(stk._cell(row, "cmf")),
        "deliv_pct": _f(stk._cell(row, "deliv_pct")),
        "deliv_quality_rel": _f(stk._cell(row, "deliv_quality_rel")),
        "base_days": _f(stk._cell(row, "base_days")),
        "atr": _f(atr),
        "atr_pct": _f(stk._cell(row, "atr_pct")),
        "lo20": _f(lo20),
        "hi_n": _f(hi_n),
        "ext_ema20": _f(stk._cell(row, "ext_ema20")),
        "ema20": _f(stk._cell(row, "ema20")),
        "ema50": _f(stk._cell(row, "ema50")),
        "ema200": _f(stk._cell(row, "ema200")),
        "coil": _f(stk.coil_score(row)),
        "closed_through": bool(closed_through),
        "liquid": liquid,
        "median_turnover": _f(med_to),
        "n_fail": int(n_fail),
    }

    tail_cols = [c for c in [
        "date", "adj", "close", "volume", "turnover", "pos_hi", "cmf",
        "deliv_pct", "rsi", "vol_ratio",
    ] if c in hist.columns]
    tail = hist[hist["date"] <= as_of][tail_cols].tail(20).copy()
    if "date" in tail.columns:
        tail["date"] = pd.to_datetime(tail["date"]).dt.strftime("%Y-%m-%d")

    # Fundamentals (cache-first, no blocking fetch on miss)
    fundamentals_data = None
    try:
        import fundamentals as fun
        fundamentals_data = fun.get_fundamentals(symbol)
    except Exception:
        pass

    fundamentals_summary = None
    if fundamentals_data:
        fundamentals_summary = {
            "market_cap_cr": _f(fundamentals_data.get("market_cap_cr")),
            "pe_ttm": _f(fundamentals_data.get("pe_ttm")),
            "sector_pe": _f(fundamentals_data.get("sector_pe")),
            "promoter_pct": _f(fundamentals_data.get("promoter_pct")),
            "promoter_pledge_pct": _f(fundamentals_data.get("promoter_pledge_pct")),
            "ttm_revenue_cr": _f(fundamentals_data.get("ttm_revenue_cr")),
            "ttm_pat_cr": _f(fundamentals_data.get("ttm_pat_cr")),
            "latest_revenue_growth_yoy": _f(fundamentals_data.get("latest_revenue_growth_yoy")),
            "latest_pat_growth_yoy": _f(fundamentals_data.get("latest_pat_growth_yoy")),
            "latest_opm_pct": _f(fundamentals_data.get("latest_opm_pct")),
            "earnings_trend": fundamentals_data.get("earnings_trend"),
            "quarters": fundamentals_data.get("quarters"),
            "updated_at": fundamentals_data.get("updated_at"),
        }

    report = {
        "found": True,
        "query": query,
        "symbol": symbol,
        "sector": sector,
        "as_of": _iso(row["date"]),
        "phase": phase,
        "phase_label": PHASES[phase],
        "sector_klass": klass,
        "sector_note": note,
        "buy_ready": buy_ready,
        "shape": shape,
        "filters": filters,
        "metrics": metrics,
        "fundamentals": fundamentals_summary,
        "setup": ({"why": setup_row.get("why"), "coil": _f(setup_row.get("coil"))}
                  if setup_row else None),
        "breakout": ({k: _f(broke_row.get(k)) if k != "why" else broke_row.get("why")
                      for k in ["flagged", "broke", "trigger", "adj",
                                "vol_expand", "why"]}
                     if broke_row else None),
        "flags": flag_hist,
        "history": tail.replace({np.nan: None, np.inf: None, -np.inf: None})
                       .to_dict(orient="records"),
        "near": [],
    }
    report["why"] = explain(report)
    return report
