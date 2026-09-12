"""
The gates, and the harness that tells you whether the gates are worth anything.

Two entry points:
  classify(panel)  -> what each sector looks like on the latest date
  backtest(panel)  -> every historical flag with forward returns, plus base rate

The second matters more than the first. A rule that only gets logged when it
works will always look excellent.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Thresholds:
    """
    Starting values, not validated parameters. Every one of these was reasoned
    from a handful of sectors on a single date, which is the weakest possible
    evidence. Move them once you have 30 logged flags — and if you find yourself
    tuning until a past move fits, you are fitting that move and nothing else.
    """
    cross_t_rel: float = 1.30       # turnover expansion vs cross-sector median
    cross_b: float = 0.15           # breadth must be clearly, not marginally, green
    quiet_t: float = 1.00
    quiet_days_10: int = 6          # of last 10 sessions below own average
    green_days_10: int = 7          # of last 10 sessions with positive breadth
    prior_quiet_5: int = 3          # expansion must come FROM quiet
    quality_floor: float = 1.00     # delivery quality vs OTHER SECTORS that day
    quality_fail: float = 0.80      # expansion below this is churn, not buying
    min_advancing: int = 3
    max_top_share: float = 0.50     # largest name's share of sector turnover
    late_t_rel: float = 2.50
    late_rs: float = 95.0           # already top of the market
    rs_decay: float = -3.0          # materially worse than 5 sessions ago
    pullback_b_floor: float = -0.35   # softer than this is a rout, not a rest
    pullback_t_floor: float = 0.45    # of the crossing day's turnover


def _disqualify(r: pd.Series, th: Thresholds) -> str | None:
    """Hard stops. Checked first, before any positive evidence is considered."""
    if (pd.notna(r["cmf_rel"]) and r["cmf_rel"] < 0
            and pd.notna(r["cmf_rel_chg_5"]) and r["cmf_rel_chg_5"] < 0):
        return "distributing (CMF below peers and falling)"
    if pd.notna(r["T_rel"]) and r["T_rel"] >= 1.5 and pd.notna(r["B"]) and r["B"] < 0:
        return "heavy volume into falling stocks"
    if (pd.notna(r["T_rel"]) and r["T_rel"] >= 1.2
            and pd.notna(r["deliv_quality_rel"]) and r["deliv_quality_rel"] < th.quality_fail):
        return f"expansion on poor delivery ({r['deliv_quality_rel']:.2f}x peers)"
    if pd.notna(r["rs_chg_5"]) and r["rs_chg_5"] < th.rs_decay:
        return "relative strength decaying"
    if (pd.notna(r["T_rel"]) and r["T_rel"] >= th.late_t_rel
            and pd.notna(r["rs"]) and r["rs"] >= th.late_rs):
        return "markup already public"
    return None


def _gate1(r: pd.Series, th: Thresholds) -> bool:
    """Quiet base. Earns a watchlist slot and nothing more."""
    return bool(
        pd.notna(r["B_green_10"]) and r["B_green_10"] >= th.green_days_10
        and pd.notna(r["T_quiet_10"]) and r["T_quiet_10"] >= th.quiet_days_10
        and pd.notna(r["cmf_rel"]) and (r["cmf_rel"] > 0 or (pd.notna(r["cmf_rel_chg_5"]) and r["cmf_rel_chg_5"] > 0))
        and pd.notna(r["rs_chg_5"]) and r["rs_chg_5"] >= 0
    )


def _gate2(r: pd.Series, th: Thresholds) -> bool:
    """The crossing. Quiet stops being quiet while breadth holds."""
    return bool(
        pd.notna(r["T_rel"]) and r["T_rel"] >= th.cross_t_rel
        and pd.notna(r["B"]) and r["B"] >= th.cross_b
        and pd.notna(r["T_quiet_prior_5"]) and r["T_quiet_prior_5"] >= th.prior_quiet_5
        and pd.notna(r["rs_chg_5"]) and r["rs_chg_5"] > 0
    )


def _gate3(r: pd.Series, th: Thresholds) -> bool:
    """Is this a sector, or one stock wearing a sector's clothes?"""
    return bool(
        pd.notna(r["deliv_quality_rel"]) and r["deliv_quality_rel"] >= th.quality_floor
        and r["n_adv"] >= th.min_advancing
        and pd.notna(r["top_share"]) and r["top_share"] < th.max_top_share
    )


def _recent_crossing(hist: pd.DataFrame, th: Thresholds, window: int = 5):
    """The most recent crossing day in the trailing window, if there was one."""
    tail = hist.iloc[-(window + 1):-1]
    hits = tail[(tail["T_rel"] >= th.cross_t_rel) & (tail["B"] >= th.cross_b)]
    return None if hits.empty else hits.iloc[-1]


def _is_expand(row: pd.Series, th: Thresholds) -> bool:
    return bool(pd.notna(row.get("T_rel")) and row["T_rel"] >= th.cross_t_rel
                and pd.notna(row.get("B")) and row["B"] >= th.cross_b)


def _quiet_prior(row: pd.Series, th: Thresholds) -> tuple[bool | None, int | None]:
    q = row.get("T_quiet_prior_5")
    if q is None or pd.isna(q):
        return None, None
    n = int(q)
    return n >= th.prior_quiet_5, n


def _fmt_day(ts) -> str:
    return pd.Timestamp(ts).strftime("%d-%m")


def shape_report(hist: pd.DataFrame, th: Thresholds | None = None,
                 lookback: int = 20) -> tuple[pd.DataFrame, dict]:
    """
    The questions the drawer used to leave to the reader, answered on the
    last `lookback` sessions:

      1. Did T_rel expand from quiet while B stayed green?
      2. Did later red days trade less than that crossing day?
      3. If a red day traded more, sellers won — it is not a pullback.

    Returns (annotated tail, report). Each tail row gets `mark`:
    crossing | pullback | heavy_red | "".
    """
    th = th or Thresholds()
    tail = hist.sort_values("date").tail(lookback).copy()
    tail["mark"] = ""

    empty = {
        "crossing": None,
        "checks": [],
        "verdict": "no_crossing",
        "verdict_text": "No T_rel expansion with green breadth in the last "
                        f"{lookback} sessions. Nothing to pull back from.",
        "buy_ready": False,
    }
    if tail.empty:
        return tail, empty

    expand = tail[tail.apply(lambda r: _is_expand(r, th), axis=1)]
    if expand.empty:
        return tail, empty

    # Most recent expansion is the reference crossing for pullback math.
    cross = expand.iloc[-1]
    from_quiet, quiet_n = _quiet_prior(cross, th)
    cross_date = pd.Timestamp(cross["date"])
    cross_t = float(cross["T"]) if pd.notna(cross["T"]) else None
    cross_t_rel = float(cross["T_rel"]) if pd.notna(cross["T_rel"]) else None
    cross_b = float(cross["B"]) if pd.notna(cross["B"]) else None

    after = tail[pd.to_datetime(tail["date"]) > cross_date]
    heavy = []
    lighter_reds = []
    for _, r in after.iterrows():
        if pd.isna(r.get("B")) or r["B"] >= 0:
            continue
        if cross_t is None or pd.isna(r.get("T")):
            continue
        rec = {
            "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
            "T": float(r["T"]),
            "ratio": float(r["T"] / cross_t) if cross_t else None,
        }
        if r["T"] >= cross_t:
            heavy.append(rec)
        else:
            lighter_reds.append(rec)

    # Marks: all expansion days, then post-crossing reds.
    for i, r in tail.iterrows():
        if _is_expand(r, th):
            tail.at[i, "mark"] = "crossing"
        elif pd.to_datetime(r["date"]) > cross_date and pd.notna(r.get("B")) and r["B"] < 0:
            if cross_t is not None and pd.notna(r.get("T")) and r["T"] >= cross_t:
                tail.at[i, "mark"] = "heavy_red"
            elif cross_t is not None and pd.notna(r.get("T")) and r["T"] < cross_t:
                tail.at[i, "mark"] = "pullback"

    last = tail.iloc[-1]
    last_is_cross = _is_expand(last, th)

    checks = []
    checks.append({
        "id": "expand",
        "ok": True,
        "text": (
            f"T_rel expanded to {cross_t_rel:.2f} on {_fmt_day(cross_date)} "
            f"while B was {cross_b:+.2f}"
        ),
    })

    if from_quiet is None:
        checks.append({
            "id": "quiet",
            "ok": None,
            "text": "Not enough history to say whether the expansion came from quiet.",
        })
    elif from_quiet:
        checks.append({
            "id": "quiet",
            "ok": True,
            "text": (
                f"Came from quiet: {quiet_n} of the prior 5 sessions were "
                "below 1.2× own average"
            ),
        })
    else:
        checks.append({
            "id": "quiet",
            "ok": False,
            "text": (
                f"Expansion on {_fmt_day(cross_date)} but only {quiet_n} of the "
                "prior 5 sessions were quiet — not a quiet-to-loud crossing"
            ),
        })

    if last_is_cross:
        checks.append({
            "id": "pullback",
            "ok": None,
            "text": "Crossing is today. No pullback yet — do not chase the expansion day.",
        })
        verdict = "crossing"
        verdict_text = "Quiet-to-loud crossing is in place." if from_quiet else (
            "Expansion is here, but it did not come from a quiet base."
        )
        if from_quiet is False:
            verdict = "expand_not_quiet"
    elif heavy:
        worst = max(heavy, key=lambda x: x["T"])
        checks.append({
            "id": "pullback",
            "ok": False,
            "text": (
                f"Red day {_fmt_day(worst['date'])} traded T {worst['T']:.2f} vs "
                f"crossing T {cross_t:.2f} ({worst['ratio']:.2f}×) — sellers won"
            ),
        })
        verdict = "sellers_won"
        verdict_text = (
            "A red day traded more than the crossing day. This is not a pullback."
        )
    elif lighter_reds:
        max_t = max(x["T"] for x in lighter_reds)
        checks.append({
            "id": "pullback",
            "ok": True,
            "text": (
                f"Pullback on less turnover: red-day T peaked at {max_t:.2f} vs "
                f"crossing T {cross_t:.2f} ({max_t / cross_t:.2f}×)"
            ),
        })
        verdict = "orderly"
        verdict_text = "Orderly pullback after the crossing — lighter red days."
    else:
        checks.append({
            "id": "pullback",
            "ok": None,
            "text": "No red day since the crossing. Still waiting for the rest.",
        })
        verdict = "waiting"
        verdict_text = "Crossing is in, but there has been no pullback yet."

    buy_ready = all(c.get("ok") is True for c in checks)
    report = {
        "crossing": {
            "date": cross_date.strftime("%Y-%m-%d"),
            "T": cross_t,
            "T_rel": cross_t_rel,
            "B": cross_b,
            "from_quiet": from_quiet,
            "quiet_days_prior": quiet_n,
        },
        "checks": checks,
        "verdict": verdict,
        "verdict_text": verdict_text,
        "heavy_reds": heavy,
        "buy_ready": buy_ready,
    }
    return tail, report


def explain_setup(shape: dict, row) -> str:
    """
    Plain-language reason this name is a buy setup. Three beats:
    the sector woke up from quiet, it then rested on lighter volume,
    and this stock is coiled under the 20-day high.
    """
    if hasattr(row, "to_dict"):
        row = row.to_dict() if not isinstance(row, dict) else row

    parts = []
    sector = row.get("sector") or "This sector"
    symbol = row.get("symbol") or "This name"
    cross = (shape or {}).get("crossing") or {}

    t_rel = cross.get("T_rel")
    b = cross.get("B")
    quiet_n = cross.get("quiet_days_prior")
    cdate = cross.get("date")
    day = _fmt_day(cdate) if cdate else None

    if t_rel is not None and b is not None and day:
        extra = f" — {int(quiet_n)} of the 5 days before that were quiet" if quiet_n else ""
        parts.append(
            f"{sector} woke up on {day}: it traded {t_rel:.2f}× a typical "
            f"sector that day, and more money was going into rising names "
            f"than falling ones (breadth {b:+.2f}){extra}."
        )
    else:
        parts.append(f"{sector} just had a clean quiet-to-loud crossing, then a lighter pullback.")

    checks = (shape or {}).get("checks") or []
    pb = next((c for c in checks if c.get("id") == "pullback"), None)
    if pb and pb.get("ok"):
        parts.append(
            "The red days after that jump traded less than the crossing day "
            "— a rest, not sellers taking over."
        )

    bits = []
    pos = row.get("pos_hi")
    trig = row.get("to_trigger")
    vol = row.get("vol_ratio")
    if pos is not None and pd.notna(pos):
        bits.append(f"sitting at {float(pos) * 100:.0f}% of its 4-month high")
    if vol is not None and pd.notna(vol):
        bits.append(f"volume dried up ({float(vol):.2f}× the 20-day average)")
    if trig is not None and pd.notna(trig):
        bits.append(f"{float(trig) * 100:.1f}% below the 20-day high")
    if bits:
        parts.append(f"{symbol} is coiled: " + ", ".join(bits) + ".")
    parts.append(
        "The setup is a close through that 20-day high on rising volume — "
        "not a buy at today's price."
    )
    return " ".join(parts)


def recommend_sectors(scan_rows: pd.DataFrame, panel: pd.DataFrame) -> set[str]:
    """PULLBACK sectors whose shape checks all passed — the only buy neighborhood."""
    if scan_rows.empty:
        return set()
    pull = scan_rows[scan_rows["klass"] == "PULLBACK"]
    ready = set()
    for sector in pull["sector"]:
        hist = panel[panel["sector"] == sector]
        _, shape = shape_report(hist)
        if shape.get("buy_ready"):
            ready.add(sector)
    return ready


def classify_one(hist: pd.DataFrame, th: Thresholds) -> dict:
    """
    hist = one sector's rows, sorted by date. Classifies the last row.

    Only CROSSING and PULLBACK are actionable. BASE is a watchlist slot.
    """
    r = hist.iloc[-1]
    res = {
        "sector": r["sector"], "date": r["date"],
        "T": r["T"], "T_rel": r["T_rel"], "B": r["B"], "B_deliv": r["B_deliv"],
        "cmf": r["cmf"], "cmf_rel": r["cmf_rel"], "rs": r["rs"], "rs_chg_5": r["rs_chg_5"],
        "deliv_quality": r["deliv_quality"],
        "deliv_quality_rel": r["deliv_quality_rel"], "n_stocks": r["n_stocks"],
        "n_adv": r["n_adv"], "top_share": r["top_share"],
        "gate1": _gate1(r, th), "gate2": _gate2(r, th), "gate3": _gate3(r, th),
    }

    why = _disqualify(r, th)
    if why:
        return {**res, "klass": "DISQUALIFIED", "note": why}

    if res["gate2"]:
        if res["gate3"]:
            return {**res, "klass": "CROSSING", "note": "all gates clear"}
        return {**res, "klass": "CROSSING_UNVERIFIED",
                "note": "expansion present, width or delivery not confirmed"}

    # Post-crossing pullback: the shape you want after a crossing fires. The
    # pullback MUST come on lower turnover than the crossing day — if the red
    # day traded more than the green day, sellers are winning and it is not a
    # pullback at all.
    cross = _recent_crossing(hist, th)
    if cross is not None and pd.notna(r["T"]) and pd.notna(cross["T"]):
        if r["T"] >= cross["T"] and pd.notna(r["B"]) and r["B"] < 0:
            return {**res, "klass": "DISQUALIFIED",
                    "note": "pullback traded more than the crossing day"}
        # A pullback is the move COOLING. Three things must still hold, or the
        # base is failing rather than resting:
        #   - breadth is soft, not routed
        #   - volume is easing, not collapsing (abandonment reads as a pullback
        #     otherwise: Telecom Equipment passed at T_rel 0.40 with zero
        #     advancing stocks and one name at 95% of turnover)
        #   - width and concentration still pass, exactly as on the CROSSING path
        orderly = (
            pd.notna(r["B"]) and r["B"] > th.pullback_b_floor
            and r["T"] >= th.pullback_t_floor * cross["T"]
            and _gate3(r, th)
        )
        if r["T"] < cross["T"]:
            if orderly:
                return {**res, "klass": "PULLBACK",
                        "note": f"crossing {cross['date']:%d-%m}, orderly pullback"}

    if res["gate1"]:
        return {**res, "klass": "BASE", "note": "quiet, bought, no expansion yet"}

    # Quiet and going nowhere is a different state from quiet and accumulating.
    if (pd.notna(r["T"]) and r["T"] < th.quiet_t
            and pd.notna(r["rs_chg_5"]) and r["rs_chg_5"] < 0):
        return {**res, "klass": "NEGLECT", "note": "quiet and drifting down"}

    return {**res, "klass": "NONE", "note": ""}


def classify(panel: pd.DataFrame, th: Thresholds | None = None,
             as_of=None) -> pd.DataFrame:
    """Classify every sector as of `as_of` (default: the last date in the panel)."""
    th = th or Thresholds()
    as_of = pd.Timestamp(as_of) if as_of is not None else panel["date"].max()
    panel = panel[panel["date"] <= as_of].sort_values(["sector", "date"])

    rows = []
    for sector, hist in panel.groupby("sector", sort=False):
        if hist.empty or hist.iloc[-1]["date"] != as_of:
            continue
        rows.append(classify_one(hist, th))

    order = {"CROSSING": 0, "PULLBACK": 1, "CROSSING_UNVERIFIED": 2,
             "BASE": 3, "NONE": 4, "NEGLECT": 5, "DISQUALIFIED": 6}
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_o"] = out["klass"].map(order)
    return (out.sort_values(["_o", "T_rel"], ascending=[True, False])
               .drop(columns="_o").reset_index(drop=True))


# --------------------------------------------------------------------------
# Forward test
# --------------------------------------------------------------------------

def backtest(panel: pd.DataFrame, th: Thresholds | None = None,
             horizons=(3, 5, 10), min_history: int = 25) -> pd.DataFrame:
    """
    Classify every sector on every date, then attach forward returns.

    Logs everything, including the sectors that were dropped. That is the point:
    a hit rate computed only over the flags you liked is not a hit rate.
    """
    th = th or Thresholds()
    panel = panel.sort_values(["sector", "date"]).copy()

    # Forward sector returns. Build a total-return index per sector first, then
    # every horizon is a single shift — no reverse-rolling gymnastics.
    panel["_tri"] = panel.groupby("sector", sort=False)["ret"].transform(
        lambda s: (1.0 + s.fillna(0.0)).cumprod()
    )
    for h in horizons:
        panel[f"fwd_{h}"] = panel.groupby("sector", sort=False)["_tri"].transform(
            lambda s, h=h: s.shift(-h) / s - 1.0
        )

    dates = sorted(panel["date"].unique())
    rows = []
    for as_of in dates[min_history:]:
        upto = panel[panel["date"] <= as_of]
        for sector, hist in upto.groupby("sector", sort=False):
            if hist.empty or hist.iloc[-1]["date"] != as_of:
                continue
            r = classify_one(hist, th)
            last = hist.iloc[-1]
            for h in horizons:
                r[f"fwd_{h}"] = last.get(f"fwd_{h}", np.nan)
            rows.append(r)

    return pd.DataFrame(rows)


def base_rates(flags: pd.DataFrame, horizons=(3, 5, 10)) -> pd.DataFrame:
    """
    Mean forward return by class, against the all-sector average on the same
    dates. The second number is the base rate, and it is the only thing that
    makes the first mean anything.
    """
    rows = []
    for klass, grp in flags.groupby("klass"):
        row = {"klass": klass, "n": len(grp)}
        for h in horizons:
            col = f"fwd_{h}"
            same_dates = flags[flags["date"].isin(grp["date"])]
            row[f"fwd_{h}"] = grp[col].mean()
            row[f"base_{h}"] = same_dates[col].mean()
            row[f"edge_{h}"] = row[f"fwd_{h}"] - row[f"base_{h}"]
            row[f"hit_{h}"] = (grp[col] > same_dates[col].mean()).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
