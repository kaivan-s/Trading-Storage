"""
Confirmed breakouts: a prior buy setup that later closed through its trigger
on rising volume.

A name drops off Setups the moment it breaks out (it fails the coil filters).
This module is the other half — it looks backward at flagged names and asks
whether a later session actually completed the setup.

Confirmation is a later session than the flag. Same-day flags cannot confirm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import flagslog
import scan as sc
import stocks as stk

SETUP_LOOKBACK = 20   # sessions of prior flags to consider
RECENT_SESSIONS = 5   # first close-through must fall in this window
VOL_MULT = 1.5        # today's volume vs the prior 20-day average
REPLAY_SESSIONS = 15  # recompute setups on recent days we may not have logged


COLUMNS = [
    "symbol", "sector", "kind", "flagged", "broke", "trigger", "adj",
    "vol_expand", "ret_since_flag", "cmf", "score", "why",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _fmt_day(ts) -> str:
    return pd.Timestamp(ts).strftime("%d-%m")


def trigger_price(row) -> float | None:
    """Flag-time 20-day high. Prefer the stored trigger; else reconstruct."""
    if hasattr(row, "to_dict") and not isinstance(row, dict):
        row = row.to_dict()
    t = row.get("trigger")
    if t is not None and pd.notna(t) and float(t) > 0:
        return float(t)
    adj = row.get("adj")
    tt = row.get("to_trigger")
    if adj is None or pd.isna(adj):
        return None
    extra = float(tt) if tt is not None and pd.notna(tt) else 0.0
    return float(adj) * (1.0 + extra)


def explain_breakout(row) -> str:
    if hasattr(row, "to_dict") and not isinstance(row, dict):
        row = row.to_dict()
    symbol = row.get("symbol") or "This name"
    flagged = _fmt_day(row["flagged"]) if row.get("flagged") else "an earlier session"
    broke = _fmt_day(row["broke"]) if row.get("broke") else "a later session"
    vol = row.get("vol_expand")
    trig = row.get("trigger")
    close = row.get("adj")
    ret = row.get("ret_since_flag")
    vol_s = f"{float(vol):.1f}× the 20-day average volume" if vol is not None and pd.notna(vol) else "rising volume"
    levels = ""
    if trig is not None and pd.notna(trig) and close is not None and pd.notna(close):
        levels = f" closed {float(close):.2f} through the trigger at {float(trig):.2f}"
    ret_s = ""
    if ret is not None and pd.notna(ret):
        ret_s = f" ({float(ret) * 100:+.1f}% since the flag)"
    extra = ""
    cmf = row.get("cmf")
    if cmf is not None and pd.notna(cmf) and float(cmf) < -0.02:
        extra = " CMF is negative on the break day — treat as weaker."
    return (
        f"{symbol} was a setup on {flagged}. On {broke} it{levels} on "
        f"{vol_s}{ret_s}.{extra}"
    )


def replay_setups(panel: pd.DataFrame, coil_stocks: pd.DataFrame,
                  as_of, n_sessions: int = REPLAY_SESSIONS) -> pd.DataFrame:
    """
    Recompute buy setups on the last `n_sessions` before `as_of`.

    The flags log only has days the user actually refreshed. Replay fills the
    gaps so a name flagged two sessions ago can still confirm today.
    """
    as_of = pd.Timestamp(as_of)
    dates = sorted(panel.loc[panel["date"] < as_of, "date"].unique())
    dates = dates[-int(n_sessions):]
    chunks = []
    for d in dates:
        slice_p = panel[panel["date"] <= d]
        scan_rows = sc.classify(slice_p, as_of=d)
        ready = sc.recommend_sectors(scan_rows, slice_p)
        if not ready:
            continue
        hits = stk.scan(coil_stocks, as_of=d, top=10_000)
        if hits.empty:
            continue
        part = hits[hits["sector"].isin(ready)].copy()
        if part.empty:
            continue
        part["as_of"] = _iso(d)
        chunks.append(part)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _normalize_flags(flags: pd.DataFrame) -> pd.DataFrame:
    if flags is None or flags.empty:
        return pd.DataFrame()
    out = flags.copy()
    if "as_of" not in out.columns:
        return pd.DataFrame()
    out["as_of"] = pd.to_datetime(out["as_of"])
    return out


def collect_flags(as_of, panel: pd.DataFrame | None = None,
                  coil_stocks: pd.DataFrame | None = None,
                  replay: bool = True) -> pd.DataFrame:
    """Union of the persisted flags log and an optional recent replay."""
    log = _normalize_flags(flagslog.load_flags())
    parts = [log] if not log.empty else []
    if replay and panel is not None and coil_stocks is not None:
        extra = replay_setups(panel, coil_stocks, as_of)
        extra = _normalize_flags(extra)
        if not extra.empty:
            parts.append(extra)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["symbol", "as_of"])
    out = out.drop_duplicates(subset=["symbol", "as_of"], keep="last")
    return out


def confirm_breakouts(flags: pd.DataFrame, stocks: pd.DataFrame, as_of,
                      recent: int = RECENT_SESSIONS,
                      vol_mult: float = VOL_MULT,
                      lookback: int = SETUP_LOOKBACK) -> pd.DataFrame:
    """
    Prior flags whose first close through the flag-time trigger, on a later
    session, landed in the last `recent` sessions and printed `vol_mult`×
    the prior 20-day average volume.
    """
    as_of = pd.Timestamp(as_of)
    f = _normalize_flags(flags)
    if f.empty or stocks is None or stocks.empty:
        return _empty()

    f = f[f["as_of"] < as_of]
    if f.empty:
        return _empty()

    sess = sorted(stocks.loc[stocks["date"] <= as_of, "date"].unique())
    if not sess:
        return _empty()
    keep_from = sess[-lookback] if len(sess) >= lookback else sess[0]
    f = f[f["as_of"] >= pd.Timestamp(keep_from)]
    if f.empty:
        return _empty()

    f = f.sort_values("as_of").groupby("symbol", as_index=False).tail(1)
    recent_dates = set(sess[-int(recent):])

    s = stocks.sort_values(["symbol", "date"])
    rows = []
    for rec in f.to_dict(orient="records"):
        trig = trigger_price(rec)
        if trig is None:
            continue
        sym = rec["symbol"]
        flagged = pd.Timestamp(rec["as_of"])
        hist = s[(s["symbol"] == sym) & (s["date"] > flagged) & (s["date"] <= as_of)]
        if hist.empty:
            continue
        crossed = hist[hist["adj"] >= trig]
        if crossed.empty:
            continue
        first = crossed.iloc[0]
        if first["date"] not in recent_dates:
            continue
        prior = s[(s["symbol"] == sym) & (s["date"] < first["date"])]
        base = prior["volume"].tail(20).mean() if not prior.empty else np.nan
        expand = (float(first["volume"]) / float(base)
                  if pd.notna(base) and float(base) > 0 else np.nan)
        if pd.isna(expand) or expand < vol_mult:
            continue
        flag_adj = rec.get("adj")
        ret = (float(first["adj"]) / float(flag_adj) - 1.0
               if flag_adj is not None and pd.notna(flag_adj) and float(flag_adj) > 0
               else np.nan)
        cmf = first["cmf"] if "cmf" in first.index and pd.notna(first.get("cmf")) else np.nan
        row = {
            "symbol": sym,
            "sector": rec.get("sector") or first.get("sector"),
            "kind": "setup",
            "flagged": _iso(flagged),
            "broke": _iso(first["date"]),
            "trigger": float(trig),
            "adj": float(first["adj"]),
            "vol_expand": float(expand),
            "ret_since_flag": None if pd.isna(ret) else float(ret),
            "cmf": None if pd.isna(cmf) else float(cmf),
        }
        row["why"] = explain_breakout(row)
        rows.append(row)

    if not rows:
        return _empty()
    return (pd.DataFrame(rows)
            .sort_values(["broke", "vol_expand"], ascending=[False, False])
            .reset_index(drop=True))


def explain_volume_break(row) -> str:
    symbol = row.get("symbol") or "This name"
    vol = row.get("vol_expand")
    trig = row.get("trigger")
    close = row.get("adj")
    score = row.get("score")
    vol_s = f"{float(vol):.1f}× volume" if vol is not None and pd.notna(vol) else "rising volume"
    levels = ""
    if trig is not None and pd.notna(trig) and close is not None and pd.notna(close):
        levels = f" Closed {float(close):.2f} through {float(trig):.2f}."
    score_s = f" Energy score {float(score):.2f}." if score is not None and pd.notna(score) else ""
    return (
        f"{symbol} closed through the 20-day high on {vol_s}.{levels}"
        f"{score_s} This is the confirmation half of For Tom — the move "
        "already printed. First unit only if RSI is already spent."
    )


def _energy_score(day: pd.DataFrame) -> pd.Series:
    """Same energy idea as tom.for_tomorrow_momentum, clipped 0–1."""
    def clip01(s, lo, hi):
        return ((pd.to_numeric(s, errors="coerce") - lo) / (hi - lo)).clip(0, 1)

    vol_r = day["vol_ratio"] if "vol_ratio" in day.columns else day["vol_x"]
    s_vol = clip01(vol_r.fillna(0), 0.8, 2.5)
    r = day["rsi"].fillna(50) if "rsi" in day.columns else pd.Series(50.0, index=day.index)
    s_rsi = np.where(r <= 72, clip01(r, 55, 72), 1.0 - clip01(r, 72, 85))
    s_rsi = pd.Series(s_rsi, index=day.index).clip(0, 1)
    s_atr = clip01(day["atr_pct"], 0.015, 0.06) if "atr_pct" in day.columns else 0.0
    s_ext = clip01(day["ext_ema20"], 0.0, 0.15) if "ext_ema20" in day.columns else 0.0
    s_cmf = clip01(day["cmf"], 0.0, 0.25) if "cmf" in day.columns else 0.0
    return 0.30 * s_vol + 0.20 * s_rsi + 0.18 * s_atr + 0.14 * s_ext + 0.18 * s_cmf


def volume_breaks(stocks: pd.DataFrame, as_of,
                  vol_mult: float = 1.2, top: int = 40,
                  sessions: int = RECENT_SESSIONS) -> pd.DataFrame:
    """
    Energy names that already closed through the prior 20-day high.

    Looks back `sessions` days (same window as setup confirmations). Volume
    gate is 1.2× — confirmation, not the old 1.5× coil-era bar — then ranked
    by the same energy score For Tom uses.
    """
    if stocks is None or stocks.empty:
        return _empty()
    as_of = pd.Timestamp(as_of)
    need = {"symbol", "date", "adj", "volume", "trigger"}
    if not need.issubset(stocks.columns):
        return _empty()

    s = stocks[stocks["date"] <= as_of].sort_values(["symbol", "date"]).copy()
    dates = sorted(s["date"].unique())
    recent = set(dates[-int(sessions):])
    if not recent:
        return _empty()

    g = s.groupby("symbol", sort=False)
    s["prior_trigger"] = g["trigger"].shift(1)
    s["v20_prior"] = g["volume"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).mean()
    )
    s["vol_x"] = s["volume"] / s["v20_prior"].replace(0, np.nan)
    day = s[s["date"].isin(recent)].copy()
    if day.empty:
        return _empty()

    trend = True
    if {"ema50", "ema200"}.issubset(day.columns):
        trend = (day["adj"] > day["ema50"]) & (day["ema50"] > day["ema200"])
    rsi_ok = True
    if "rsi" in day.columns:
        rsi_ok = day["rsi"].fillna(50) <= 85
    hit = day[
        day["prior_trigger"].notna()
        & (day["adj"] >= day["prior_trigger"])
        & (day["vol_x"].fillna(0) >= vol_mult)
        & trend
        & rsi_ok
    ].copy()
    if hit.empty:
        return _empty()

    p = stk.CoilParams()
    if "turnover" in stocks.columns:
        med = stocks.groupby("symbol")["turnover"].median()
        liquid = med[med >= p.min_median_turnover_lacs].index
        hit = hit[hit["symbol"].isin(liquid)]
    if hit.empty:
        return _empty()

    hit["score"] = _energy_score(hit)
    # One row per symbol — keep the most recent break, highest score
    hit = (hit.sort_values(["date", "score"], ascending=[False, False])
              .drop_duplicates("symbol")
              .head(int(top)))

    rows = []
    for rec in hit.to_dict(orient="records"):
        row = {
            "symbol": rec["symbol"],
            "sector": rec.get("sector"),
            "kind": "momentum",
            "flagged": None,
            "broke": _iso(rec["date"]),
            "trigger": float(rec["prior_trigger"]),
            "adj": float(rec["adj"]),
            "vol_expand": float(rec["vol_x"]),
            "ret_since_flag": None,
            "cmf": float(rec["cmf"]) if rec.get("cmf") is not None and pd.notna(rec.get("cmf")) else None,
            "score": float(rec["score"]) if rec.get("score") is not None and pd.notna(rec.get("score")) else None,
        }
        row["why"] = explain_volume_break(row)
        rows.append(row)
    return pd.DataFrame(rows)


def find_breakouts(panel: pd.DataFrame, coil_stocks: pd.DataFrame, as_of,
                   replay: bool = True) -> pd.DataFrame:
    """Prior coil setups that confirmed, plus recent energy breaks."""
    flags = collect_flags(as_of, panel, coil_stocks, replay=replay)
    setups = confirm_breakouts(flags, coil_stocks, as_of)
    vol = volume_breaks(coil_stocks, as_of)
    if setups.empty and vol.empty:
        return _empty()
    if not setups.empty and "kind" not in setups.columns:
        setups = setups.copy()
        setups["kind"] = "setup"
    if vol.empty:
        return setups
    if setups.empty:
        cols = list(COLUMNS) + (["score"] if "score" in vol.columns else [])
        return vol.reindex(columns=[c for c in cols if c in vol.columns or c in COLUMNS])
    taken = set(setups["symbol"])
    extra = vol[~vol["symbol"].isin(taken)]
    out = pd.concat([setups, extra], ignore_index=True)
    sort_cols = ["broke", "score", "vol_expand"]
    sort_cols = [c for c in sort_cols if c in out.columns]
    return (out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            .reset_index(drop=True))
