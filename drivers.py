"""
The likely driver of a big move, read from the evidence on record.

Each session's move is tested against a set of move types -- a reaction to
disclosed information, sector or market-wide buying/selling, a breakout from
(or breakdown out of) a base, a speculative spike, a rebound after a steep
fall, profit-taking after a sharp run, a large deal, a trend continuing. Each
type scores the evidence that fits it and lists what doesn't fit; the best
fit becomes the driver, the runner-up is kept as a secondary factor, and the
confidence reflects how strong and how uncontested the fit is.

This is a reading of price, volume, delivery, peers and filings, not a
cause established from inside knowledge, so the wording is "consistent with"
and "points to", never "because". A move nothing on record explains is
labelled that way instead of being given a story.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from carry import _snap_band
import filing_facts as ff

YEAR = 250
BASE_MAX_WIDTH = 0.18


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and not np.isfinite(v)) and pd.notna(v)


def _pct(v: float, nd: int = 0, sign: bool = False) -> str:
    return f"{v * 100:+.{nd}f}%" if sign else f"{abs(v) * 100:.{nd}f}%"


def _rs(v: float) -> str:
    return f"₹{v:,.2f}" if v < 1000 else f"₹{v:,.0f}"


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def features(h: pd.DataFrame, i: int) -> dict | None:
    """Day + prior-history features for row `i` of one symbol's sorted rows."""
    if i < 1:
        return None
    r, p = h.iloc[i], h.iloc[:i]
    close, adj = float(r["close"]), float(r["adj"])
    ret = r.get("ret")
    if not _ok(ret):
        ret = adj / float(p["adj"].iloc[-1]) - 1.0
    prev_close = r.get("prev_close")
    if not _ok(prev_close) or not prev_close:
        prev_close = close / (1.0 + ret)
    hi, lo, op = float(r["high"]), float(r["low"]), float(r["open"])
    rng = hi - lo
    up = ret > 0

    vmed = p["volume"].astype(float).tail(50).median() if len(p) >= 20 else np.nan
    vol_x = float(r["volume"]) / vmed if _ok(vmed) and vmed > 0 else None
    dp = r.get("deliv_pct")
    dq = r.get("deliv_quality")
    at_edge = close >= hi * 0.999 if up else close <= lo * 1.001
    try:
        fno_set = ff.fno_symbols()
    except Exception:
        fno_set = None
    fno = (str(r["symbol"]).upper() in fno_set) if fno_set and "symbol" in r else None
    band = _snap_band(abs(ret)) if at_edge and not fno else None
    band = float(band) if _ok(band) and band else None

    pa = p["adj"].astype(float)
    y = p.tail(YEAR)
    prior_hi = y["adj_high"].max() if len(y) >= 120 else np.nan
    prior_lo = y["adj_low"].min() if len(y) >= 120 else np.nan

    base = None
    for n in (20, 30, 40, 60, 90, 120):
        if len(p) < n:
            break
        w = p.tail(n)
        bh, bl = w["adj_high"].max(), w["adj_low"].min()
        if bl > 0 and bh / bl - 1.0 <= BASE_MAX_WIDTH:
            base = {"sessions": n, "high": float(bh), "low": float(bl)}

    last10 = p.tail(10)
    l_ret = last10["ret"] if "ret" in last10 else last10["adj"].pct_change()
    circ_up = int(((l_ret >= 0.045) & (last10["close"] >= last10["high"] * 0.999)).sum())
    circ_dn = int(((l_ret <= -0.045) & (last10["close"] <= last10["low"] * 1.001)).sum())

    def back(n):
        return float(pa.iloc[-1] / pa.iloc[-n - 1] - 1.0) if len(pa) > n and pa.iloc[-n - 1] else None

    e50 = p["ema50"].iloc[-1] if "ema50" in p else np.nan
    e200 = p["ema200"].iloc[-1] if "ema200" in p else np.nan
    turn = p["turnover"].astype(float).tail(50).median() / 100.0 if "turnover" in p and len(p) >= 20 else None

    return {
        "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
        "ret": float(ret),
        "gap": op / prev_close - 1.0 if prev_close else None,
        "close_loc": (close - lo) / rng if rng > 0 else None,
        "high": hi, "low": lo, "close": close, "adj": adj,
        "vol_x": vol_x,
        "deliv_pct": float(dp) if _ok(dp) else None,
        "deliv_q": float(dq) if _ok(dq) else None,
        "band": band,
        "new_52w": ("high" if up and _ok(prior_hi) and adj > prior_hi
                    else "low" if not up and _ok(prior_lo) and adj < prior_lo else None),
        "turnover_cr": float(r["turnover"]) / 100.0 if "turnover" in r and _ok(r["turnover"]) else None,
        "med_turnover_cr": float(turn) if _ok(turn) else None,
        "ret20": back(20), "ret60": back(60),
        "from_high": float(pa.iloc[-1] / y["adj_high"].max() - 1.0) if len(y) >= 60 else None,
        "prev_above50": bool(pa.iloc[-1] > e50) if _ok(e50) else None,
        "prev_above200": bool(pa.iloc[-1] > e200) if _ok(e200) else None,
        "stack": bool(e50 > e200) if _ok(e50) and _ok(e200) else None,
        "now_above200": bool(adj > e200) if _ok(e200) else None,
        "now_above50": bool(adj > e50) if _ok(e50) else None,
        "rsi_prev": float(p["rsi"].iloc[-1]) if "rsi" in p and _ok(p["rsi"].iloc[-1]) else None,
        "base": base,
        "circ_up_10": 0 if fno else circ_up, "circ_dn_10": 0 if fno else circ_dn,
        "fno": fno,
    }


def peer_context(frame: pd.DataFrame, day, symbol: str, sector: str | None) -> dict:
    """Market median and the stock's sector peers (excluding itself) on `day`."""
    d = frame.loc[frame["date"] == pd.Timestamp(day), ["symbol", "sector", "ret"]].dropna(subset=["ret"])
    out = {"mkt": float(d["ret"].median()) if len(d) else 0.0, "sec": None, "sec_n": 0, "sec_up": 0,
           "sector": sector}
    if sector:
        peers = d.loc[(d["sector"] == sector) & (d["symbol"] != symbol), "ret"]
        if len(peers):
            out.update(sec=float(peers.median()), sec_n=int(len(peers)), sec_up=int((peers > 0).sum()))
    return out


# --------------------------------------------------------------------------
# Move types
# --------------------------------------------------------------------------

class _C:
    def __init__(self, key: str, label: str, score: float, why: str):
        self.key, self.label, self.score, self.why = key, label, score, why
        self.supports: list[str] = []
        self.against: list[str] = []

    def plus(self, pts: float, text: str):
        self.score += pts
        self.supports.append(text)

    def minus(self, pts: float, text: str):
        self.score -= pts
        self.against.append(text)


def _volume_notes(c: _C, f: dict, up: bool, holders_word: str):
    vx, dq, dp = f["vol_x"], f["deliv_q"], f["deliv_pct"]
    if vx is not None:
        if vx >= 3:
            c.plus(0.75, f"Volume {vx:.1f}× its 50-session median")
        elif vx >= 2:
            c.plus(0.5, f"Volume {vx:.1f}× its 50-session median")
        elif vx < 1.3:
            c.minus(0.5, f"Volume was close to normal ({vx:.1f}×) for a move this size")
    if dq is not None and dp is not None:
        base = dp / dq if dq else None
        if dq >= 1.15:
            c.plus(0.5, f"Delivery {dp:.0f}% vs its usual {base:.0f}% — more shares were "
                        f"{holders_word}, not just traded intraday")
        elif dq <= 0.8:
            c.minus(0.5, _low_delivery(dp, base))


def _low_delivery(dp: float, base: float | None) -> str:
    usual = f" vs its usual {base:.0f}%" if base else ""
    if dp < 40:
        return f"Delivery only {dp:.0f}%{usual} — most of the volume was intraday"
    return f"Delivery {dp:.0f}%{usual} — lower than normal, so more of the volume was intraday"


def _close_note(c: _C, f: dict, up: bool):
    loc = f["close_loc"]
    if loc is None:
        return
    if up and loc >= 0.8:
        c.plus(0.25, "Closed near the day's high")
    elif not up and loc <= 0.2:
        c.plus(0.25, "Closed near the day's low")
    elif up and loc < 0.4:
        c.minus(0.5, f"Gave back much of the gain, closing well below the day's high of {_rs(f['high'])}")
    elif not up and loc > 0.6:
        c.minus(0.5, f"Recovered much of the fall, closing well above the day's low of {_rs(f['low'])}")


FILING_LABELS = {
    "Results": "Results reaction",
    "Order win": "Order win",
    "Credit rating": "Credit rating change",
    "Clarification on media report": "Media report",
    "Deal / restructuring": "Corporate deal",
    "Fund raising": "Fund raising",
    "Legal / regulatory": "Legal / regulatory news",
    "Management change": "Management change",
    "Promoter / insider": "Promoter / insider activity",
    "Corporate action": "Corporate action",
    "Board outcome": "Board decision",
    "General Updates": "Company update",
    "Updates": "Company update",
    "Press release": "Press release",
    "Agreement / MoU": "Agreement / MoU",
    "Material disclosure": "Material disclosure",
}


# Disclosures that move prices more often rank ahead when two land together.
CATEGORY_WEIGHT = {
    "Results": 0.5,
    "Order win": 0.25,
    "Credit rating": 0.25,
    "Clarification on media report": 0.25,
    "Deal / restructuring": 0.25,
    "Fund raising": 0.25,
    "Legal / regulatory": 0.25,
    "Management change": -0.25,
    "Corporate action": -0.25,
}


def _filing_candidate(f: dict, up: bool, filing: dict, facts: dict | None, ttm_rev: float | None) -> _C:
    cat = filing["category"]
    when = "after the previous close" if filing.get("after_close") else "during the session"
    high = filing["importance"] == "high"
    detail = ""
    if facts:
        detail = facts.get("summary") or ""
    label = FILING_LABELS.get(cat, cat)
    weight = CATEGORY_WEIGHT.get(cat, 0.0) if high else 0.0
    c = _C("filing", label, (2.75 if high else 1.5) + weight,
           f"Consistent with a reaction to the {cat.lower()} filed {when}"
           + (f" — {detail}" if detail else "") + ".")
    c.supports.append(f"{cat} filed {when} ({filing['date']} {filing['time']})")

    gap = f["gap"]
    if filing.get("after_close") and gap is not None and abs(gap) >= 0.03 and (gap > 0) == up:
        c.plus(0.75, f"Gapped {'up' if up else 'down'} {_pct(gap, 1)} at the open — the move began "
                     "as soon as the market could react to the filing")
    elif filing.get("after_close") and gap is not None and abs(gap) < 0.01:
        c.minus(0.25, "Opened flat despite the overnight filing; the move built during the session")

    res = (facts or {}).get("results")
    if res:
        pc, rc = res.get("profit_chg"), res.get("revenue_chg")
        cmp = res.get("compare") or ""
        swing = res.get("profit_swing")
        good = (pc is not None and pc >= 0.1) or swing == "turned to profit"
        bad = (pc is not None and pc <= -0.1) or swing == "swung to a loss"
        if pc is not None:
            word = f"Profit {_pct(pc, 0, sign=True)} {cmp}"
        elif swing:
            word = f"Profit {swing} {cmp}"
        else:
            word = None
        if word and ((up and good) or (not up and bad)):
            c.plus(0.75, word + (f", revenue {_pct(rc, 0, sign=True)}" if rc is not None else ""))
        elif word and ((up and bad) or (not up and good)):
            c.minus(0.5, f"{word}, yet the stock {'rose' if up else 'fell'} — the reaction may reflect "
                         "margins, guidance or expectations rather than the headline number")
        elif word:
            c.supports.append(word + (f", revenue {_pct(rc, 0, sign=True)}" if rc is not None else ""))
    pf = (facts or {}).get("pdf") or {}
    if pf.get("value_cr"):
        if ttm_rev:
            share = pf["value_cr"] / ttm_rev
            if share >= 0.05:
                c.plus(0.75, f"Order/deal value ₹{pf['value_cr']:,.0f} cr is ~{share * 100:.0f}% of trailing-year revenue")
            elif share < 0.01:
                c.minus(0.5, f"Order/deal value ₹{pf['value_cr']:,.0f} cr is under 1% of trailing-year revenue — small for a move this size")
            else:
                c.supports.append(f"Order/deal value ₹{pf['value_cr']:,.0f} cr (~{share * 100:.1f}% of trailing-year revenue)")
        else:
            c.supports.append(f"Order/deal value ₹{pf['value_cr']:,.0f} cr")
    act = pf.get("rating_action") or ""
    if act:
        positive = act.startswith("upgraded") or act.endswith("positive")
        negative = act.startswith("downgraded") or act.endswith("negative")
        txt = f"Rating {act}" + (f" by {pf['agency']}" if pf.get("agency") else "")
        if (up and positive) or (not up and negative):
            c.plus(0.75, txt)
        elif (up and negative) or (not up and positive):
            c.minus(0.5, txt + f", which doesn't match the stock {'rising' if up else 'falling'}")
        elif act == "reaffirmed":
            c.minus(0.5, txt + " — no change, so unlikely on its own to move the stock")
    if pf.get("news_title"):
        c.supports.append(f"The filing responds to a media report: \u201c{pf['news_title']}\u201d")
    if pf.get("company_stance"):
        c.supports.append(f"The company {pf['company_stance']}")
    _volume_notes(c, f, up, "bought to be held" if up else "sold by holders")
    return c


def _candidates(f: dict, up: bool, peers: dict, filings: list, deals: list | None,
                facts_by_filing: dict, ttm_rev: float | None) -> list[_C]:
    ret, out = f["ret"], []
    sgn = 1 if up else -1
    # The exchange asking about a price/volume spurt follows the move; it isn't its cause.
    filings = [x for x in filings if x["category"] != "Exchange query on price/volume"]
    word_up = "rose" if up else "fell"

    # Disclosed information.
    seen = set()
    for filing in filings[:4]:
        if filing["category"] in seen:
            continue
        seen.add(filing["category"])
        out.append(_filing_candidate(f, up, filing, facts_by_filing.get(id(filing)), ttm_rev))

    # Sector.
    sec, n = peers.get("sec"), peers.get("sec_n", 0)
    if sec is not None and n >= 3 and np.sign(sec) == sgn and abs(sec) >= 0.02:
        same = peers["sec_up"] if up else n - peers["sec_up"]
        c = _C("sector", f"Sector-wide {'buying' if up else 'selling'}", 2.0,
               f"Consistent with sector-wide {'buying' if up else 'selling'} in {peers['sector']}: "
               f"peers moved a median {_pct(sec, 1, sign=True)}, {same} of {n} {word_up}.")
        c.supports.append(f"{peers['sector']} peers: median {_pct(sec, 1, sign=True)}, {same} of {n} {word_up}")
        if abs(sec) >= 0.5 * abs(ret):
            c.plus(1.0, "The sector move accounts for at least half of this stock's move")
        else:
            c.minus(0.5, f"Moved far more than its peers ({_pct(ret, 1, sign=True)} vs a median "
                         f"{_pct(sec, 1, sign=True)}), so the sector explains only part of it")
        if same / n >= 0.7:
            c.plus(0.5, f"Broad participation: {same / n * 100:.0f}% of peers moved the same way")
        out.append(c)

    # Market.
    mkt = peers.get("mkt") or 0.0
    if np.sign(mkt) == sgn and abs(mkt) >= 0.015:
        c = _C("market", f"Market-wide {'rally' if up else 'sell-off'}", 1.5,
               f"Part of a broad market {'rally' if up else 'sell-off'}: the median stock moved "
               f"{_pct(mkt, 1, sign=True)}.")
        if abs(mkt) >= 0.4 * abs(ret):
            c.plus(1.0, f"The market median ({_pct(mkt, 1, sign=True)}) is a large share of this move")
        else:
            c.minus(0.5, "The stock moved several times more than the market")
        out.append(c)

    base = f["base"]
    vx = f["vol_x"]
    # Breakout / breakdown from a base.
    if up and base and base["sessions"] >= 20 and f["adj"] > base["high"]:
        c = _C("breakout", "Breakout from a base", 2.0,
               f"Points to fresh buying after a period of consolidation: it closed above a "
               f"{base['sessions']}-session range ({_rs(base['low'])}–{_rs(base['high'])}).")
        c.supports.append(f"Closed above its {base['sessions']}-session range high of {_rs(base['high'])}")
        _volume_notes(c, f, up, "bought to be held")
        _close_note(c, f, up)
        if f["new_52w"] == "high":
            c.plus(0.25, "Also a new 52-week high")
        out.append(c)
    elif up and f["new_52w"] == "high":
        c = _C("breakout", "Fresh 52-week high", 1.5,
               "Consistent with momentum buying: the stock closed at a new 52-week high.")
        _volume_notes(c, f, up, "bought to be held")
        _close_note(c, f, up)
        out.append(c)
    if not up and base and base["sessions"] >= 20 and f["adj"] < base["low"]:
        c = _C("breakdown", "Breakdown from a range", 2.0,
               f"Points to selling pressure breaking a {base['sessions']}-session range "
               f"({_rs(base['low'])}–{_rs(base['high'])}): it closed below the range low.")
        c.supports.append(f"Closed below its {base['sessions']}-session range low of {_rs(base['low'])}")
        _volume_notes(c, f, up, "sold by holders")
        _close_note(c, f, up)
        out.append(c)
    elif not up and f["prev_above200"] and f["now_above200"] is False:
        c = _C("breakdown", "Fell below the 200-day average", 1.5,
               "Consistent with longer-term holders exiting: it closed below its 200-day average "
               "after trading above it.")
        _volume_notes(c, f, up, "sold by holders")
        _close_note(c, f, up)
        out.append(c)

    # Speculative spike / unwind.
    has_high_filing = any(x["importance"] == "high" for x in filings)
    mt = f["med_turnover_cr"]
    dq, dp = f["deliv_q"], f["deliv_pct"]
    thin = mt is not None and mt < 5
    liquid = f["fno"] or (mt is not None and mt >= 25)
    if not has_high_filing and not liquid and (f["band"] or (thin and vx is not None and vx >= 4)):
        if up:
            c = _C("speculative", "Speculative spike", 1.0,
                   "Looks like a speculative or momentum-driven spike, with no company disclosure on record.")
        else:
            c = _C("speculative", "Speculative unwind", 1.0,
                   "Looks like an unwind of speculative positions, with no company disclosure on record.")
        if f["band"]:
            c.plus(1.0, f"Closed at its {'upper' if up else 'lower'} price band ({f['band'] * 100:.0f}%)")
        if vx is not None and vx >= 4:
            c.plus(0.25, f"Volume {vx:.1f}× normal")
        if dq is not None and dq <= 0.8:
            c.plus(0.75, _low_delivery(dp, dp / dq if dq else None))
        elif dq is not None and dq >= 1.2:
            c.minus(0.75, "Delivery was above normal, which is unusual for purely speculative trading")
        if thin:
            c.plus(0.5, f"Thinly traded: median daily turnover ₹{mt:.1f} cr")
        k = f["circ_up_10"] if up else f["circ_dn_10"]
        if k >= 2:
            c.plus(0.75, f"{k} {'upper' if up else 'lower'} circuits in the previous 10 sessions")
        if not up and (f["ret20"] or 0) >= 0.3:
            c.plus(0.75, f"Followed a {_pct(f['ret20'])} run-up over the previous 20 sessions")
        if not up and f["circ_up_10"] >= 2:
            c.plus(0.5, f"{f['circ_up_10']} upper circuits in the 10 sessions before")
        out.append(c)

    r20, r60 = f["ret20"], f["ret60"]
    rsi = f["rsi_prev"]
    # Rebound after a steep fall.
    fell = r20 is not None and (r20 <= -0.15 or (r20 <= -0.08 and rsi is not None and rsi < 35))
    deep = f["from_high"] is not None and f["from_high"] <= -0.35 and rsi is not None and rsi < 40
    if up and (fell or deep):
        c = _C("rebound", "Rebound after a steep fall", 1.5,
               "Consistent with a rebound after heavy selling"
               + (f": it had fallen {_pct(r20)} over the previous 20 sessions." if fell
                  else f": it was {_pct(f['from_high'])} below its 52-week high."))
        if fell:
            c.supports.append(f"Down {_pct(r20)} over the previous 20 sessions")
        if rsi is not None and rsi < 30:
            c.plus(0.5, f"RSI was {rsi:.0f} the day before — deeply oversold")
        elif rsi is not None and rsi < 40:
            c.plus(0.25, f"RSI was {rsi:.0f} the day before — on the weak side")
        if vx is not None and vx >= 2:
            c.plus(0.25, f"Volume {vx:.1f}× normal")
            # Overnight shorts exist only through futures, so the label is
            # reserved for F&O stocks (or, if the list is unknown, liquid ones).
            can_short = f["fno"] if f["fno"] is not None else (mt is not None and mt >= 25)
            if can_short and dq is not None and dq <= 0.85:
                c.plus(0.5, "High volume with below-normal delivery after a fall in an F&O stock — "
                            "consistent with short covering (traders buying back positions they had sold)")
                c.label = "Short covering after a fall"
                c.why = c.why.replace("a rebound after heavy selling", "short covering after heavy selling")
        if f["now_above50"] is False:
            c.against.append("Still below its 50-day average after the bounce")
        out.append(c)

    # Profit-taking after a sharp run.
    if not up and ((r20 is not None and r20 >= 0.2) or (r60 is not None and r60 >= 0.35)):
        ran, span = (r20, 20) if r20 is not None and r20 >= 0.2 else (r60, 60)
        c = _C("profit_taking", "Profit-taking after a sharp run", 1.75,
               f"Consistent with profit-taking: it had risen {_pct(ran)} over the previous {span} sessions.")
        c.supports.append(f"Up {_pct(ran)} over the previous {span} sessions")
        if rsi is not None and rsi >= 70:
            c.plus(0.5, f"RSI was {rsi:.0f} the day before — stretched after one-sided gains")
        if f["now_above50"]:
            c.supports.append("Still above its 50-day average — the run's structure is intact")
        if f["deliv_q"] is not None and f["deliv_q"] >= 1.3 and vx is not None and vx >= 3:
            c.against.append("Heavy delivery-based selling, which looks more like holders exiting than routine profit-taking")
        out.append(c)

    # Heavy intraday churn: big volume that mostly didn't stay in demat accounts.
    if vx is not None and vx >= 4 and dq is not None and dq <= 0.75 and dp is not None and not has_high_filing:
        c = _C("churn", "Heavy intraday trading", 1.5,
               f"Points to short-term traders driving the move: volume was {vx:.1f}× normal but only "
               f"{dp:.0f}% was taken as delivery, so most shares were bought and sold the same day.")
        if vx >= 10:
            c.plus(0.5, f"Volume {vx:.1f}× its 50-session median")
        if dp < 20:
            c.plus(0.5, f"Delivery just {dp:.0f}%")
        _close_note(c, f, up)
        out.append(c)

    # Delivery-led: big volume that holders took (or gave up) for keeps.
    if vx is not None and vx >= 2.5 and dq is not None and dq >= 1.1 and dp is not None:
        base_dp = dp / dq
        c = _C("delivery", f"Delivery-led {'buying' if up else 'selling'}", 1.75,
               f"Points to investors {'accumulating' if up else 'exiting'}, not just traders: volume was "
               f"{vx:.1f}× normal and delivery {dp:.0f}% vs its usual {base_dp:.0f}%.")
        if vx >= 5:
            c.plus(0.5, f"Volume {vx:.1f}× its 50-session median")
        if dq >= 1.4:
            c.plus(0.5, f"Delivery {dq:.1f}× its own norm")
        _close_note(c, f, up)
        out.append(c)

    # Gap with nothing on file: something happened overnight outside filings.
    gap = f["gap"]
    if gap is not None and abs(gap) >= 0.03 and (gap > 0) == up and not filings:
        c = _C("gap", f"Gap {'up' if up else 'down'} on outside news", 1.5,
               f"Gapped {'up' if up else 'down'} {_pct(gap, 1)} at the open with no exchange filing — "
               "points to news or developments outside company disclosures (media, sector or "
               "policy news, global cues) before the open.")
        if vx is not None and vx >= 2:
            c.plus(0.5, f"Volume {vx:.1f}× normal")
        if abs(ret) >= abs(gap) * 0.85:
            c.plus(0.25, "Held the opening gap through the session")
        else:
            c.against.append("Gave back part of the opening gap during the session")
        out.append(c)

    # A big move on ordinary volume.
    if vx is not None and vx < 1.5 and not filings:
        c = _C("quiet", "Move on ordinary volume", 1.25,
               f"{'Rose' if up else 'Fell'} {_pct(ret, 1)} on near-normal volume ({vx:.1f}×) — "
               f"there was no burst of {'buying' if up else 'selling'} and no disclosure, so nothing "
               "on record marks it as more than an ordinary, if large, day.")
        out.append(c)

    # Large deals.
    if deals:
        side = "BUY" if up else "SELL"
        mine = [d for d in deals if d["side"] == side]
        other = [d for d in deals if d["side"] != side]
        if mine:
            val = sum(d.get("value_cr") or 0 for d in mine)
            top = max(mine, key=lambda d: d.get("value_cr") or 0)
            c = _C("deal", f"Large {'buying' if up else 'selling'} deal", 1.5,
                   f"Consistent with a large {'buyer' if up else 'seller'}: {top['client']} "
                   f"{'bought' if up else 'sold'} ₹{top.get('value_cr') or 0:,.1f} cr in a {top['kind'].lower()} deal.")
            tc = f["turnover_cr"]
            if tc and val:
                share = val / tc
                if share >= 0.2:
                    c.plus(1.0, f"Deals on this side were ~{share * 100:.0f}% of the day's turnover")
                else:
                    c.supports.append(f"Deals on this side were ~{share * 100:.0f}% of the day's turnover")
            if other:
                c.against.append(f"There were also large deals on the other side ({len(other)})")
            out.append(c)

    # Trend continuation.
    if up and f["prev_above50"] and f["stack"] and (r20 or 0) > 0 and not (base and f["adj"] > base["high"]):
        c = _C("trend", "Uptrend continuing", 1.0,
               "Consistent with an existing uptrend extending: it was already above its 50- and 200-day "
               "averages and up over the previous month.")
        if vx is not None and vx >= 1.5:
            c.plus(0.25, f"Volume {vx:.1f}× normal")
        out.append(c)
    if not up and f["prev_above50"] is False and f["stack"] is False and (r20 or 0) < 0:
        c = _C("trend", "Downtrend continuing", 1.0,
               "Consistent with an existing downtrend extending: it was already below its 50- and 200-day "
               "averages and down over the previous month.")
        if vx is not None and vx >= 1.5:
            c.plus(0.25, f"Volume {vx:.1f}× normal")
        out.append(c)
    return out


def classify(f: dict, peers: dict, filings: list | None = None, deals: list | None = None,
             checked: str = "both", facts_by_filing: dict | None = None,
             ttm_rev: float | None = None) -> dict:
    """Likely driver for one session's move. `checked` says what external
    data was available: "both" (filings + deals), "filings", or "none"."""
    up = f["ret"] > 0
    cands = _candidates(f, up, peers, filings or [], deals, facts_by_filing or {}, ttm_rev)
    cands.sort(key=lambda c: c.score, reverse=True)
    top = cands[0] if cands and cands[0].score >= 1.25 else None

    if top is None:
        unseen = {"both": "no exchange filing or large deal is on record",
                  "filings": "no exchange filing is on record (deal data wasn't available)",
                  "none": "filings and deals couldn't be checked",
                  "older": "filings from that far back aren't loaded here"}[checked]
        return {
            "key": "unexplained",
            "label": "No clear driver on record",
            "confidence": "Low",
            "why": (f"None of the usual patterns fit well and {unseen}. The move may reflect news "
                    "outside exchange filings, or trading interest that the data doesn't show."),
            "supports": [],
            "against": [],
            "also": [],
        }

    if top.score >= 4.25 or (top.score >= 3.5 and not top.against):
        conf = "High"
    elif top.score >= 2.25:
        conf = "Medium"
    else:
        conf = "Low"
    if len(top.against) >= 3:
        conf = {"High": "Medium", "Medium": "Low"}.get(conf, conf)
    against = list(top.against)
    if checked in ("none", "older") and top.key != "filing":
        against.append("Exchange filings couldn't be checked for this session" if checked == "none"
                       else "Filings from that far back aren't loaded, so a disclosure can't be ruled out")
        conf = "Low" if conf == "Low" else "Medium"
    also = [{"label": c.label, "why": c.why} for c in cands[1:3]
            if c.score >= 2.0 and c.key != top.key][:2]
    return {
        "key": top.key,
        "label": top.label,
        "confidence": conf,
        "why": top.why,
        "supports": top.supports,
        "against": against,
        "also": also,
    }


# --------------------------------------------------------------------------
# Base rates: what followed past moves with the same label
# --------------------------------------------------------------------------

_RATES_FILE = Path(__file__).resolve().parent / "data" / "base_rates.json"
_rates: dict | None = None


def _base_rates() -> dict:
    global _rates
    if _rates is None:
        try:
            _rates = json.loads(_RATES_FILE.read_text())
        except (FileNotFoundError, ValueError):
            _rates = {}
    return _rates


def base_rate(label: str | None, up: bool) -> dict | None:
    """Outcome stats for past moves with this label and direction (from baserates.py)."""
    data = _base_rates()
    hit = (data.get("rates") or {}).get(f"{label}|{'up' if up else 'down'}")
    if not hit:
        return None
    return {"h5": hit.get("h5"), "h20": hit.get("h20"), "since": (data.get("window") or {}).get("start")}


def attach_base_rates(rows: list[dict]) -> list[dict]:
    """Add `driver.base_rate` to mover rows; rows are copied, not mutated."""
    out = []
    for r in rows or []:
        d = r.get("driver")
        if d and r.get("ret") is not None:
            r = {**r, "driver": {**d, "base_rate": base_rate(d.get("label"), r["ret"] > 0)}}
        out.append(r)
    return out


# --------------------------------------------------------------------------
# Filing facts
# --------------------------------------------------------------------------

def research_filings(symbol: str, day: str, filings: list) -> dict:
    """Facts for the filings that matter, keyed by id(filing)."""
    out = {}
    seen = set()
    for fl in filings[:3]:
        # Consolidated and standalone results arrive as two filings; read once.
        if fl["importance"] != "high" or fl["category"] in seen:
            continue
        seen.add(fl["category"])
        facts: dict = {}
        if fl["category"] == "Results":
            res = ff.results_numbers(symbol, around=day)
            if res:
                facts["results"] = res
                facts["summary"] = ff.describe_results(res)
        else:
            pf = ff.pdf_facts(fl)
            if pf:
                facts["pdf"] = pf
                facts["summary"] = ff.describe_pdf_facts(fl["category"], pf)
        if facts:
            out[id(fl)] = facts
            if facts.get("summary"):
                fl["fact"] = facts["summary"]
    return out
