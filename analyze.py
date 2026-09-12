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

ACTIONS = {
    "buy": "Buy",
    "add": "Add",
    "hold": "Hold",
    "sell": "Sell",
    "wait": "Wait",
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
            return br["why"] + " Still check CMF and delivery — this is confirmation, not a market order."
        return (
            f"{symbol} has closed through its trigger on rising volume. "
            "That is the confirmation the setup was waiting for — not a buy at any price."
        )

    if phase == "potential":
        why = (report.get("setup") or {}).get("why")
        if why:
            return why
        return (
            f"{symbol} is a potential breakout: coiled under the 20-day high "
            f"inside a clean {sector} pullback. Put an alert at the trigger; "
            "do not buy today's close."
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
            "a late first unit, not a coil."
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
        f"{sector} is {klass}.{fail_s} Nothing here is a buy."
    )


def _round_px(v: float | None) -> float | None:
    if v is None or pd.isna(v):
        return None
    x = float(v)
    if x >= 100:
        return round(x, 1)
    if x >= 10:
        return round(x, 2)
    return round(x, 2)


def suggested_entry(report: dict) -> float | None:
    """The price this system would act at, if it acts at all."""
    m = report.get("metrics") or {}
    phase = report.get("phase")
    trigger = m.get("trigger")
    px = m.get("adj")
    if phase in ("potential", "coiled", "near_miss") and trigger:
        return _round_px(trigger)
    return _round_px(px)


def trade_plan(report: dict, entry=None) -> dict:
    """
    Long-only plan from this system's structure: 20-day base, ATR, trigger.

    Not a validated edge. The stop is where the base is wrong; the target is
    2R plus a measured move. Action assumes a long from `entry` (or the
    suggested entry if omitted).
    """
    m = report.get("metrics") or {}
    phase = report.get("phase")
    px = m.get("adj")
    trigger = m.get("trigger") or m.get("prior_trigger")
    atr = m.get("atr")
    if atr is None and m.get("atr_pct") and px:
        atr = px * m["atr_pct"]
    lo20 = m.get("lo20")
    hi_n = m.get("hi_n")
    ema20 = m.get("ema20")
    cmf = m.get("cmf")
    vol_x = m.get("vol_expand")
    verdict = (report.get("shape") or {}).get("verdict")
    klass = report.get("sector_klass")

    if entry is None:
        entry = suggested_entry(report)
    try:
        entry = float(entry) if entry is not None else None
    except (TypeError, ValueError):
        entry = None
    if entry is not None and entry <= 0:
        entry = None

    empty = {
        "action": "wait",
        "action_label": ACTIONS["wait"],
        "entry": entry,
        "suggested_entry": suggested_entry(report),
        "stop": None,
        "target": None,
        "target2": None,
        "risk": None,
        "reward": None,
        "rr": None,
        "why": "Need a positive entry price and enough history to place a stop.",
    }
    if entry is None or px is None:
        return empty

    sl_atr = (entry - 1.5 * atr) if atr else None
    sl_base = (lo20 * 0.997) if lo20 else None
    below = [x for x in (sl_atr, sl_base) if x is not None and x < entry]
    stop = max(below) if below else sl_atr
    if stop is None:
        return {**empty, "entry": _round_px(entry)}
    if atr and (entry - stop) < 0.6 * atr:
        stop = entry - 1.5 * atr
    if stop >= entry:
        stop = entry * 0.97

    risk = entry - stop
    target = entry + 2.0 * risk
    measured = None
    if trigger and lo20 and trigger > lo20:
        measured = (trigger if phase in ("potential", "coiled") else entry) + (trigger - lo20)
    if measured is None and m.get("range20") and trigger:
        measured = trigger * (1.0 + float(m["range20"]))
    target2 = None
    for cand in (measured, hi_n):
        if cand is not None and cand > target * 1.01:
            target2 = cand
            break
    if target2 is None and measured is not None and measured > entry:
        target2 = measured if measured > target else None
    if hi_n and hi_n > entry and (target2 is None or hi_n > target2):
        if hi_n > target * 1.02:
            target2 = hi_n

    rr = (target - entry) / risk if risk > 0 else None
    entering_here = abs(px - entry) / entry <= 0.008
    has_fill = abs(px - entry) / entry > 0.008
    planned_breakout = (
        phase == "potential" and trigger is not None
        and entry >= trigger * 0.997 and px < trigger
    )
    in_position = has_fill and not planned_breakout
    already_long = in_position and px > entry
    failed = (
        in_position
        and (cmf is not None and cmf < -0.05 and trigger is not None and px < trigger)
        and phase in ("broke_out", "at_high")
    )
    sellers = verdict == "sellers_won"
    confirmed = phase in ("broke_out", "momentum") or (
        phase == "at_high" and vol_x is not None and vol_x >= 1.5
    )
    rsi = m.get("rsi")
    late = (rsi is not None and rsi > 68) or ((m.get("ext_ema20") or 0) > 0.06)
    at_target = in_position and px >= target
    pullback = False
    if confirmed and trigger and px > stop:
        line = max(x for x in (trigger, ema20) if x is not None) if (trigger or ema20) else trigger
        if line is not None:
            near_line = px <= line * 1.008
            not_extended = atr is None or px <= entry + 0.3 * atr
            pullback = near_line and not_extended and px > stop and already_long

    if in_position and px <= stop:
        action = "sell"
        why = (
            f"Last close {px:.2f} is at or through the stop {stop:.2f}. "
            "The base is wrong from this entry — exit."
        )
    elif failed:
        action = "sell"
        why = (
            f"Back under the trigger at {trigger:.2f} with CMF negative. "
            "Treat this as a failed breakout and get out."
        )
    elif in_position and sellers and px < entry:
        action = "sell"
        why = (
            "Sector structure broke (sellers won or the sector is disqualified) "
            "and the close is below your entry. Do not average down."
        )
    elif at_target and (cmf is not None and cmf < 0 or (m.get("ext_ema20") or 0) > 0.08):
        action = "sell"
        why = (
            f"First target {target:.2f} is reached and the name is extended or "
            "money flow is fading. Bank the 2R; do not add."
        )
    elif at_target:
        action = "hold"
        why = (
            f"First target {target:.2f} is tagged. Hold the rest only if you trail "
            f"under the last swing; do not add. Stretch sits near {target2:.2f}."
            if target2 else
            f"First target {target:.2f} is tagged. Hold and trail; do not add."
        )
    elif phase == "potential" and trigger and entry < trigger * 0.997 and entering_here:
        action = "wait"
        why = (
            f"This is still the coil. The buy this system allows is a close "
            f"through the trigger at {trigger:.2f}, not {entry:.2f}. "
            f"If you already own it, the stop is {stop:.2f}."
        )
    elif phase == "potential" and trigger and entry >= trigger * 0.997:
        action = "buy"
        why = (
            f"Entry is at the trigger. Buy only on a close through {trigger:.2f} "
            f"on rising volume. Stop {stop:.2f}, first target {target:.2f} (2R)."
        )
    elif confirmed and entering_here and not already_long and px > stop:
        action = "buy"
        vol_s = f" on {vol_x:.1f}× volume" if vol_x else ""
        late_s = (
            " RSI / extension say the first push is spent — first unit only, do not chase size."
            if late else ""
        )
        why = (
            f"Volume break through the 20-day high{vol_s}. Buy the first unit "
            f"near {entry:.2f}. Stop {stop:.2f}, first target {target:.2f}."
            f"{late_s} Sector class is the industry, not a veto on this print."
        )
    elif confirmed and pullback:
        action = "add"
        why = (
            f"You are long from {entry:.2f} and price has come back to the "
            f"breakout line. One add is allowed above {stop:.2f}, not a full "
            f"new position. First target remains {target:.2f}."
        )
    elif confirmed and px > stop:
        action = "hold"
        why = (
            f"Long from {entry:.2f} is working. Hold as long as the close stays "
            f"above {stop:.2f}. First target {target:.2f}; do not chase an add "
            "up here."
        )
    elif already_long and px > stop:
        action = "hold"
        why = (
            f"You are already long from {entry:.2f}, but this is not a clean "
            f"breakout setup. Hold only if you accept that; stop {stop:.2f}. "
            "Do not add."
        )
    else:
        action = "wait"
        why = (
            f"No buy here. If you use {entry:.2f} anyway, the mechanical stop is "
            f"{stop:.2f} and 2R is {target:.2f} — that is risk math, not a signal."
        )

    return {
        "action": action,
        "action_label": ACTIONS[action],
        "entry": _round_px(entry),
        "suggested_entry": suggested_entry(report),
        "stop": _round_px(stop),
        "target": _round_px(target),
        "target2": _round_px(target2),
        "risk": _round_px(risk),
        "reward": _round_px(target - entry),
        "rr": round(rr, 1) if rr else None,
        "why": why,
    }


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
    report["plan"] = trade_plan(report, entry)
    return report
