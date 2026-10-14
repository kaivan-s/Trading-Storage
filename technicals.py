"""
Technical and chart reading for the stock page.

Everything here DESCRIBES the price history -- trend structure, where the
price sits in its range, how volume and volatility compare with the stock's
own norm, and the notable days on the chart. Nothing forecasts. The wording
is deliberately past/present tense ("is above", "has traded"), never "will"
or "should", because the page must stay a description of data and not a
recommendation.

Input is one symbol's rows from the engine's feature frame (stocks.py), so
prices are split/dividend-adjusted and line up with the scan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEAR = 250            # sessions in a trading year
PIVOT_SPAN = 5        # a swing high/low must be the extreme of +/- 5 sessions
GAP_PCT = 0.04        # open vs previous close
BIG_DAY_PCT = 0.07    # close vs previous close, when it didn't gap
VOL_SPIKE = 3.0       # volume vs the prior 50-session median
MAX_EVENTS = 14


def _f(v, nd=2):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd)


def _ret(s: pd.Series, n: int):
    if len(s) <= n:
        return None
    a, b = s.iloc[-n - 1], s.iloc[-1]
    if not (pd.notna(a) and pd.notna(b)) or a == 0:
        return None
    return _f(b / a - 1.0, 4)


def _rs(v: float) -> str:
    return f"₹{v:,.2f}" if v < 1000 else f"₹{v:,.0f}"


def _pct(v: float, nd=1, sign=False) -> str:
    s = f"{v * 100:+.{nd}f}%" if sign else f"{abs(v) * 100:.{nd}f}%"
    return s


def _date(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _human_date(ts) -> str:
    return pd.Timestamp(ts).strftime("%-d %b %Y")


# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------

def _trend(h: pd.DataFrame) -> dict:
    last = h.iloc[-1]
    close, e20, e50, e200 = last["adj"], last.get("ema20"), last.get("ema50"), last.get("ema200")
    if pd.isna(e50) or pd.isna(e200):
        return {
            "label": "Not enough history",
            "tone": "muted",
            "text": "Fewer than 200 sessions of history, so the long-term trend can't be read yet.",
            "checks": [],
        }

    e200_prev = h["ema200"].iloc[-21] if len(h) > 21 else np.nan
    slope200 = (e200 / e200_prev - 1.0) if pd.notna(e200_prev) and e200_prev else 0.0
    rising = slope200 > 0.005
    falling = slope200 < -0.005

    above50, above200, stack = close > e50, close > e200, e50 > e200
    if above50 and stack and rising:
        label, tone = "Uptrend", "good"
        text = "Price is above its 50-day and 200-day averages, the 50-day is above the 200-day, and the 200-day is rising."
    elif not above50 and not stack and falling:
        label, tone = "Downtrend", "bad"
        text = "Price is below its 50-day and 200-day averages, the 50-day is below the 200-day, and the 200-day is falling."
    elif stack and not above50:
        label, tone = "Pullback in an uptrend", "warn"
        text = "The 50-day average is still above the 200-day, but price has dipped below the 50-day."
    elif above200 and not stack:
        label, tone = "Recovering", "warn"
        text = "Price is back above its 200-day average, while the 50-day average is still below the 200-day."
    elif not above200 and stack:
        label, tone = "Weakening", "warn"
        text = "Price has fallen below its 200-day average while the 50-day is still above it."
    else:
        label, tone = "Sideways / mixed", "muted"
        text = "The averages are not lined up in either direction and the 200-day is roughly flat."

    checks = [
        {"label": "Above 20-day average", "ok": bool(pd.notna(e20) and close > e20),
         "value": _f(close / e20 - 1.0, 4) if pd.notna(e20) and e20 else None},
        {"label": "Above 50-day average", "ok": bool(above50), "value": _f(close / e50 - 1.0, 4)},
        {"label": "Above 200-day average", "ok": bool(above200), "value": _f(close / e200 - 1.0, 4)},
        {"label": "50-day above 200-day", "ok": bool(stack), "value": None},
        {"label": "200-day rising (last 20 sessions)", "ok": bool(rising), "value": _f(slope200, 4)},
    ]
    return {"label": label, "tone": tone, "text": text, "checks": checks,
            "ema20": _f(e20), "ema50": _f(e50), "ema200": _f(e200)}


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------

def _pivots(h: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    hi, lo = h["adj_high"].to_numpy(), h["adj_low"].to_numpy()
    dates = h["date"].to_numpy()
    k = PIVOT_SPAN
    highs, lows = [], []
    for i in range(k, len(h) - k):
        win_hi = hi[i - k:i + k + 1]
        win_lo = lo[i - k:i + k + 1]
        if np.isfinite(hi[i]) and hi[i] == np.nanmax(win_hi):
            highs.append({"price": float(hi[i]), "date": dates[i]})
        if np.isfinite(lo[i]) and lo[i] == np.nanmin(win_lo):
            lows.append({"price": float(lo[i]), "date": dates[i]})
    return highs, lows


def _nearest(points: list[dict], price: float, above: bool, n: int = 2) -> list[dict]:
    cand = [p for p in points if (p["price"] > price * 1.005 if above else p["price"] < price * 0.995)]
    cand.sort(key=lambda p: abs(p["price"] - price))
    out: list[dict] = []
    for p in cand:
        # Two swings within 1.5% of each other are the same level.
        if any(abs(p["price"] / q["price"] - 1.0) < 0.015 for q in out):
            continue
        out.append(p)
        if len(out) == n:
            break
    return [{"price": _f(p["price"]), "date": _date(p["date"]),
             "pct": _f(p["price"] / price - 1.0, 4)} for p in out]


def _levels(h: pd.DataFrame) -> dict:
    y = h.tail(YEAR)
    close = float(h["adj"].iloc[-1])
    hi_idx, lo_idx = y["adj_high"].idxmax(), y["adj_low"].idxmin()
    hi52, lo52 = float(y.loc[hi_idx, "adj_high"]), float(y.loc[lo_idx, "adj_low"])
    span = hi52 - lo52
    highs, lows = _pivots(y)
    return {
        "close": _f(close),
        "high_52w": _f(hi52), "high_52w_date": _date(y.loc[hi_idx, "date"]),
        "low_52w": _f(lo52), "low_52w_date": _date(y.loc[lo_idx, "date"]),
        "from_high": _f(close / hi52 - 1.0, 4),
        "from_low": _f(close / lo52 - 1.0, 4),
        "range_pos": _f((close - lo52) / span, 4) if span > 0 else None,
        "resistance": _nearest(highs, close, above=True),
        "support": _nearest(lows, close, above=False),
        "sessions": int(len(y)),
    }


# --------------------------------------------------------------------------
# Volume, volatility, momentum
# --------------------------------------------------------------------------

def _activity(h: pd.DataFrame) -> dict:
    vol = h["volume"].astype(float)
    v20 = vol.tail(20).mean()
    v50 = vol.tail(50).mean()
    chg = h["adj"].pct_change()
    last20 = pd.DataFrame({"v": vol, "c": chg}).tail(20)
    up_v = last20.loc[last20["c"] > 0, "v"].sum()
    dn_v = last20.loc[last20["c"] < 0, "v"].sum()

    dp = h["deliv_pct"].astype(float) if "deliv_pct" in h.columns else pd.Series(dtype=float)
    d20 = dp.tail(20).mean() if not dp.empty else np.nan
    d60 = dp.tail(60).mean() if not dp.empty else np.nan

    atr = h["atr_pct"].astype(float) if "atr_pct" in h.columns else pd.Series(dtype=float)
    atr_now = atr.iloc[-1] if not atr.empty else np.nan
    atr_med = atr.tail(YEAR).median() if not atr.empty else np.nan
    vol_rel = atr_now / atr_med if pd.notna(atr_now) and pd.notna(atr_med) and atr_med else np.nan
    if pd.isna(vol_rel):
        vol_state = None
    elif vol_rel < 0.8:
        vol_state = "calmer than usual"
    elif vol_rel > 1.25:
        vol_state = "more volatile than usual"
    else:
        vol_state = "about usual"

    rsi = h["rsi"].iloc[-1] if "rsi" in h.columns else np.nan
    if pd.isna(rsi):
        rsi_text = None
    elif rsi >= 70:
        rsi_text = "above 70 — recent gains have been strong and one-sided"
    elif rsi <= 30:
        rsi_text = "below 30 — recent losses have been strong and one-sided"
    elif rsi >= 50:
        rsi_text = "between 50 and 70 — gains have outweighed losses recently"
    else:
        rsi_text = "between 30 and 50 — losses have outweighed gains recently"

    return {
        "vol_20_vs_50": _f(v20 / v50, 2) if v50 else None,
        "up_down_volume": _f(up_v / dn_v, 2) if dn_v else None,
        "deliv_20": _f(d20, 1),
        "deliv_60": _f(d60, 1),
        "atr_pct": _f(atr_now, 4),
        "atr_vs_usual": _f(vol_rel, 2),
        "volatility_state": vol_state,
        "rsi": _f(rsi, 1),
        "rsi_text": rsi_text,
    }


def _returns(h: pd.DataFrame) -> dict:
    s = h["adj"]
    return {"1w": _ret(s, 5), "1m": _ret(s, 21), "3m": _ret(s, 63),
            "6m": _ret(s, 126), "1y": _ret(s, YEAR)}


# --------------------------------------------------------------------------
# Chart story: ranges, runs, notable days
# --------------------------------------------------------------------------

def _recent_range(h: pd.DataFrame) -> dict | None:
    """Longest recent stretch (ending today) that stayed inside an 18% band."""
    best = None
    for n in (20, 30, 40, 60, 90, 120, 180):
        if len(h) < n:
            break
        w = h.tail(n)
        hi, lo = w["adj_high"].max(), w["adj_low"].min()
        if lo > 0 and hi / lo - 1.0 <= 0.18:
            best = {"sessions": n, "high": _f(hi), "low": _f(lo), "width": _f(hi / lo - 1.0, 4)}
    return best


def _biggest_run(h: pd.DataFrame) -> dict | None:
    """Largest low-to-high advance inside 60 sessions over the last year."""
    y = h.tail(YEAR).reset_index(drop=True)
    if len(y) < 30:
        return None
    lows, highs = y["adj_low"].to_numpy(), y["adj_high"].to_numpy()
    best = (0.0, None, None)
    for j in range(len(y)):
        i0 = max(0, j - 60)
        seg = lows[i0:j + 1]
        if not np.isfinite(seg).any():
            continue
        # Last occurrence of the low, so a flat base doesn't stretch the run.
        i = i0 + len(seg) - 1 - int(np.nanargmin(seg[::-1]))
        if lows[i] > 0 and np.isfinite(highs[j]):
            gain = highs[j] / lows[i] - 1.0
            if gain > best[0]:
                best = (gain, i, j)
    gain, i, j = best
    if i is None or gain < 0.15:
        return None
    return {"gain": _f(gain, 4), "from": _date(y.loc[i, "date"]), "to": _date(y.loc[j, "date"]),
            "from_price": _f(lows[i]), "to_price": _f(highs[j]), "sessions": int(j - i)}


def _events(h: pd.DataFrame) -> list[dict]:
    d = h.copy()
    scale = d["adj"] / d["close"].replace(0, np.nan)
    d["adj_open"] = d["open"] * scale
    d["prev"] = d["adj"].shift(1)
    d["chg"] = d["adj"] / d["prev"] - 1.0
    d["gap"] = d["adj_open"] / d["prev"] - 1.0
    d["vmed"] = d["volume"].astype(float).shift(1).rolling(50, min_periods=20).median()
    d["vx"] = d["volume"] / d["vmed"]
    d["prior_hi"] = d["adj_high"].shift(1).rolling(YEAR, min_periods=120).max()
    d = d.iloc[1:].tail(YEAR)

    raw: list[dict] = []
    last_high_mark = None
    for idx, r in d.iterrows():
        notes, score, up = [], 0.0, None
        if pd.notna(r["gap"]) and abs(r["gap"]) >= GAP_PCT:
            up = r["gap"] > 0
            notes.append(f"Gapped {'up' if up else 'down'} {_pct(r['gap'])}")
            score += abs(r["gap"]) * 10
        elif pd.notna(r["chg"]) and abs(r["chg"]) >= BIG_DAY_PCT:
            up = r["chg"] > 0
            notes.append(f"{'Rose' if up else 'Fell'} {_pct(r['chg'])} in one session")
            score += abs(r["chg"]) * 8
        if pd.notna(r["vx"]) and r["vx"] >= VOL_SPIKE:
            notes.append(f"volume {r['vx']:.1f}× normal")
            score += min(r["vx"], 10) / 3
            if up is None and pd.notna(r["chg"]):
                up = r["chg"] >= 0
        if (pd.notna(r["prior_hi"]) and r["adj"] > r["prior_hi"]
                and (last_high_mark is None or (r["date"] - last_high_mark).days > 30)):
            notes.append("closed at a new 52-week high")
            score += 1.0
            up = True if up is None else up
            last_high_mark = r["date"]
        if not notes:
            continue
        text = notes[0] + ("" if len(notes) == 1 else ", " + ", ".join(notes[1:]))
        text = text[0].upper() + text[1:]
        if pd.notna(r["chg"]) and "Rose" not in notes[0] and "Fell" not in notes[0]:
            text += f" (close {_pct(r['chg'], sign=True)})"
        raw.append({"time": _date(r["date"]), "text": text, "up": bool(up), "score": score})

    raw.sort(key=lambda e: e["score"], reverse=True)
    keep = sorted(raw[:MAX_EVENTS], key=lambda e: e["time"])
    for e in keep:
        e.pop("score", None)
    return keep


def _story(h: pd.DataFrame, trend: dict, levels: dict, act: dict,
           rng: dict | None, run: dict | None, events: list[dict]) -> list[str]:
    out: list[str] = []
    close = levels["close"]
    if run:
        out.append(
            f"Its largest advance in the last year was {_pct(run['gain'], 0)} in {run['sessions']} sessions, "
            f"from {_rs(run['from_price'])} ({_human_date(run['from'])}) to {_rs(run['to_price'])} "
            f"({_human_date(run['to'])})."
        )
    if rng:
        out.append(
            f"For the last {rng['sessions']} sessions it has traded between {_rs(rng['low'])} and "
            f"{_rs(rng['high'])}, a {_pct(rng['width'], 0)} range."
        )
    elif len(h) > 60:
        r60 = h["adj"].iloc[-1] / h["adj"].iloc[-61] - 1.0
        out.append(f"Over the last 60 sessions it has moved {_pct(r60, 1, sign=True)} without a tight range forming.")
    if levels.get("from_high") is not None:
        fh = levels["from_high"]
        if fh > -0.02:
            out.append(f"It is trading at or near its 52-week high of {_rs(levels['high_52w'])}.")
        else:
            out.append(
                f"It is {_pct(fh)} below its 52-week high of {_rs(levels['high_52w'])} "
                f"({_human_date(levels['high_52w_date'])})."
            )
    if levels["resistance"]:
        r = levels["resistance"][0]
        out.append(f"The nearest prior swing high above the price is {_rs(r['price'])} ({_human_date(r['date'])}), {_pct(r['pct'])} away.")
    if levels["support"]:
        s = levels["support"][0]
        out.append(f"The nearest prior swing low below the price is {_rs(s['price'])} ({_human_date(s['date'])}), {_pct(s['pct'])} away.")
    recent = [e for e in events if pd.Timestamp(e["time"]) >= h["date"].iloc[-1] - pd.Timedelta(days=30)]
    if recent:
        e = recent[-1]
        out.append(f"Most recent notable session: {_human_date(e['time'])} — {e['text'][0].lower() + e['text'][1:]}.")
    return out


def _summary(trend: dict, levels: dict, act: dict, rets: dict) -> list[str]:
    out = [f"{trend['label']}: {trend['text']}"]
    if levels.get("from_high") is not None and levels.get("from_low") is not None:
        out.append(
            f"{_pct(levels['from_high'])} below its 52-week high and "
            f"{_pct(levels['from_low'])} above its 52-week low."
        )
    bits = []
    if act.get("vol_20_vs_50") is not None:
        bits.append(f"20-day volume is {act['vol_20_vs_50']:.2f}× its 50-day average")
    if act.get("volatility_state"):
        bits.append(f"daily swings are {act['volatility_state']}")
    if act.get("deliv_20") is not None and act.get("deliv_60") is not None:
        bits.append(f"delivery averaged {act['deliv_20']:.0f}% over 20 sessions vs {act['deliv_60']:.0f}% over 60")
    if bits:
        s = "; ".join(bits)
        out.append(s[0].upper() + s[1:] + ".")
    if rets.get("1y") is not None:
        out.append(f"Return over the last year: {_pct(rets['1y'], 1, sign=True)}.")
    return out


def build(hist: pd.DataFrame) -> dict | None:
    """Technical reading for one symbol. `hist` is that symbol's feature rows."""
    need = {"date", "adj", "adj_high", "adj_low", "close", "open", "volume"}
    if hist is None or hist.empty or not need.issubset(hist.columns):
        return None
    h = hist.sort_values("date").dropna(subset=["adj", "adj_high", "adj_low"]).reset_index(drop=True)
    if len(h) < 30:
        return None

    trend = _trend(h)
    levels = _levels(h)
    act = _activity(h)
    rets = _returns(h)
    rng = _recent_range(h)
    run = _biggest_run(h)
    events = _events(h)

    return {
        "as_of": _date(h["date"].iloc[-1]),
        "summary": _summary(trend, levels, act, rets),
        "trend": trend,
        "levels": levels,
        "activity": act,
        "returns": rets,
        "range": rng,
        "biggest_run": run,
        "events": events,
        "story": _story(h, trend, levels, act, rng, run, events),
    }
