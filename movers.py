"""
Why it moved: every stock that rose or fell 5%+ in a session, with the facts
that coincided with the move.

For each mover we lay out:

  * how much of the move was the market, the sector, and the stock itself
    (plain decomposition: market median, sector median minus market, and the
    remainder -- no regression, so a reader can check it by hand);
  * the shape of the day: gap at the open vs a move through the session,
    and where it closed in its range;
  * volume against its 50-session median, delivery against its own norm;
  * a price-band (circuit) close and a new 52-week high/low;
  * exchange filings made between the previous close and this close, and
    bulk/block deals printed that session.

The wording is "coincided with", never "because of": a filing on the same
day is a fact, the causal link is the reader's call. When nothing on record
lines up, it says so rather than inventing a reason.
"""

from __future__ import annotations

from datetime import datetime, time as dtime

import numpy as np
import pandas as pd

import db
import drivers
import filing_facts
import news
from carry import _snap_band
from fetch import BASE, NSESession

THRESHOLD = 0.05
MARKET_CLOSE = dtime(15, 30)
LARGE_DEALS = BASE + "/api/snapshot-capital-market-largedeal"
ANNOUNCEMENTS_ALL = (BASE + "/api/corporate-announcements?index=equities"
                     "&from_date={frm}&to_date={to}")


def _f(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _pct(v, nd=1, sign=True):
    s = f"{v * 100:+.{nd}f}%" if sign else f"{abs(v) * 100:.{nd}f}%"
    return s


# --------------------------------------------------------------------------
# External facts: filings and deals
# --------------------------------------------------------------------------

def _filings_by_symbol(sess: NSESession, prev_day: pd.Timestamp, day: pd.Timestamp) -> dict | None:
    """Filings after the previous session's close, up to this session's close."""
    url = ANNOUNCEMENTS_ALL.format(frm=prev_day.strftime("%d-%m-%Y"), to=day.strftime("%d-%m-%Y"))
    try:
        rows = sess.get_json(url, referer=news.REFERER)
    except Exception as exc:
        print(f"movers: filings unavailable: {exc}")
        return None
    if isinstance(rows, dict):
        rows = rows.get("data") or []
    lo = datetime.combine(prev_day.date(), MARKET_CLOSE)
    hi = datetime.combine(day.date(), MARKET_CLOSE)
    out: dict[str, list] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        when = news._parse_dt(r.get("an_dt")) or news._parse_dt(r.get("sort_date"))
        if when is None or not (lo < when <= hi):
            continue
        shaped = news._shape(r)
        if shaped is None or shaped["importance"] == "low":
            continue
        shaped["after_close"] = when.date() == prev_day.date()
        out.setdefault((r.get("symbol") or "").upper(), []).append(shaped)
    rank = {"high": 0, "normal": 1}
    for v in out.values():
        v.sort(key=lambda x: (rank.get(x["importance"], 2), x["date"], x["time"]))
    return out


def _deals_by_symbol(sess: NSESession, day: pd.Timestamp) -> dict | None:
    """Bulk and block deals printed on `day`. NSE only serves the latest
    session here, so an older day comes back as None (unknown), not empty."""
    try:
        d = sess.get_json(LARGE_DEALS, referer=BASE + "/market-data/large-deals")
    except Exception as exc:
        print(f"movers: deals unavailable: {exc}")
        return None
    as_on = pd.to_datetime(d.get("as_on_date"), format="%d-%b-%Y", errors="coerce")
    if pd.isna(as_on) or as_on.normalize() != day.normalize():
        return None
    out: dict[str, list] = {}
    for kind, key in (("Bulk", "BULK_DEALS_DATA"), ("Block", "BLOCK_DEALS_DATA")):
        for r in d.get(key) or []:
            when = pd.to_datetime(r.get("date"), format="%d-%b-%Y", errors="coerce")
            if pd.isna(when) or when.normalize() != day.normalize():
                continue
            qty = pd.to_numeric(r.get("qty"), errors="coerce")
            px = pd.to_numeric(r.get("watp"), errors="coerce")
            out.setdefault((r.get("symbol") or "").upper(), []).append({
                "kind": kind,
                "side": (r.get("buySell") or "").upper(),
                "client": (r.get("clientName") or "").strip(),
                "qty": int(qty) if pd.notna(qty) else None,
                "price": _f(px, 2),
                "value_cr": _f(qty * px / 1e7, 2) if pd.notna(qty) and pd.notna(px) else None,
            })
    return out


# --------------------------------------------------------------------------
# Per-stock reading
# --------------------------------------------------------------------------

def _qty(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1e7:
        return f"{n / 1e7:.2f} cr"
    if n >= 1e5:
        return f"{n / 1e5:.1f} lakh"
    return f"{n:,}"


def _day_shape(r, up: bool) -> str:
    gap, ret, loc = r["gap"], r["ret"], r["close_loc"]
    word = "up" if up else "down"
    if pd.notna(gap) and abs(gap) >= 0.03 and np.sign(gap) == np.sign(ret):
        if abs(ret) >= abs(gap) * 0.85:
            s = f"Gapped {word} {_pct(gap, sign=False)} at the open and closed {_pct(ret)}"
        else:
            s = f"Gapped {word} {_pct(gap, sign=False)} at the open but gave part of it back, closing {_pct(ret)}"
    elif pd.notna(gap) and abs(gap) < 0.015:
        s = f"Opened near the previous close and {'rose' if up else 'fell'} through the session to {_pct(ret)}"
    elif pd.notna(gap) and np.sign(gap) != np.sign(ret) and abs(gap) >= 0.015:
        s = f"Opened {_pct(gap)} and reversed to close {_pct(ret)}"
    else:
        s = f"Opened {_pct(gap)} and extended the move to close {_pct(ret)}"
    if pd.notna(loc):
        if up and loc < 0.4:
            s += f", well below the day's high of ₹{r['high']:,.2f}"
        elif not up and loc > 0.6:
            s += f", well above the day's low of ₹{r['low']:,.2f}"
        elif up and loc >= 0.95:
            s += ", at the day's high"
        elif not up and loc <= 0.05:
            s += ", at the day's low"
    return s + "."


def _read(r, up: bool, filings: list, deals: list, checked: str = "both",
          driver: dict | None = None) -> dict:
    tags, detail = [], [_day_shape(r, up)]

    # Market / sector / stock decomposition.
    mkt, sec, ret = r["mkt_med"], r["sec_med"], r["ret"]
    peers_ok = r["sec_n"] >= 3
    if pd.isna(sec) or not peers_ok:
        sec = mkt
    parts = {"market": _f(mkt), "sector": _f(sec - mkt), "stock": _f(ret - sec)}
    same_way = peers_ok and np.sign(sec) == np.sign(ret) and abs(sec) >= 0.02
    sector_led = same_way and abs(sec) >= 0.5 * abs(ret)
    if peers_ok:
        detail.append(
            f"Other {r['sector']} stocks moved a median {_pct(sec)} "
            f"({int(r['sec_up'])} of {int(r['sec_n'])} up); the broad market median was {_pct(mkt)}."
        )
    else:
        detail.append(f"Too few listed peers in {r['sector']} to compare; the broad market median was {_pct(mkt)}.")
    if sector_led:
        tags.append("Sector-wide move")

    if pd.notna(r["vol_x"]):
        v = f"Volume was {r['vol_x']:.1f}× its 50-session median"
        if pd.notna(r["deliv_pct"]) and pd.notna(r["deliv_base"]):
            v += f"; delivery {r['deliv_pct']:.0f}% vs its usual {r['deliv_base']:.0f}%"
        detail.append(v + ".")
        if r["vol_x"] >= 3:
            tags.append(f"Volume {r['vol_x']:.1f}×")

    if pd.notna(r["band"]) and r["band"]:
        side = "upper" if up else "lower"
        detail.append(f"Closed at its {side} price band ({r['band'] * 100:.0f}%) — trading was capped there.")
        tags.append(f"{'Upper' if up else 'Lower'} circuit {r['band'] * 100:.0f}%")
    if r["new_52w"]:
        tags.append(f"New 52W {r['new_52w']}")
        detail.append(f"Closed at a new 52-week {r['new_52w']}.")

    for f in filings[:3]:
        when = "after the previous close" if f.get("after_close") else "during the session"
        detail.append(f"Filed {when} ({f['date']} {f['time']}): {f['category']} — {f['text'] or f['title']}")
        if f.get("fact"):
            detail.append(f"From the filing: {f['fact']}.")
    if filings:
        tags.insert(0, filings[0]["category"])
    for d in deals[:3]:
        detail.append(
            f"{d['kind']} deal: {d['client']} {'bought' if d['side'] == 'BUY' else 'sold'} "
            f"{_qty(d['qty'])} shares at ₹{d['price']:,.2f}"
            + (f" (₹{d['value_cr']:.1f} cr)." if d.get("value_cr") is not None else ".")
        )
    if deals:
        tags.append(f"{deals[0]['kind']} deal")

    high_filing = next((f for f in filings if f["importance"] == "high"), None)
    if high_filing:
        headline = f"Coincided with an exchange filing: {high_filing['category']}."
    elif sector_led:
        headline = f"Moved with its sector — {r['sector']} median {_pct(sec)}."
    elif deals:
        d = deals[0]
        headline = f"Coincided with a {d['kind'].lower()} deal ({d['client']} {'buying' if d['side'] == 'BUY' else 'selling'})."
    elif filings:
        headline = f"Coincided with an exchange filing: {filings[0]['category']}."
    else:
        record = {
            "both": "No exchange filing or deal on record for the session.",
            "filings": "No exchange filing on record; deal data isn't available for this session.",
            "none": "Filings and deals couldn't be checked for this session.",
        }[checked]
        if same_way:
            headline = (f"Mostly stock-specific — its sector also {'rose' if up else 'fell'} "
                        f"(peer median {_pct(sec)}), but this moved far more. {record}")
        else:
            headline = f"Stock-specific move. {record}"
    if driver:
        headline = driver["why"]

    return {
        "symbol": r["symbol"],
        "sector": r["sector"],
        "close": _f(r["close"], 2),
        "ret": _f(ret),
        "gap": _f(r["gap"]),
        "high": _f(r["high"], 2),
        "low": _f(r["low"], 2),
        "close_loc": _f(r["close_loc"], 3),
        "vol_x": _f(r["vol_x"], 2),
        "deliv_pct": _f(r["deliv_pct"], 1),
        "deliv_base": _f(r["deliv_base"], 1),
        "turnover_cr": _f(r["turnover"] / 100.0, 2),
        "band": _f(r["band"], 2),
        "new_52w": r["new_52w"],
        "sector_median": _f(sec) if peers_ok else None,
        "sector_up": int(r["sec_up"]) if peers_ok else None,
        "sector_n": int(r["sec_n"]),
        "parts": parts,
        "headline": headline,
        "driver": driver,
        "tags": tags,
        "detail": detail,
        "filings": filings[:3],
        "deals": deals[:3],
    }


def _driver_counts(items: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    for x in items:
        label = (x.get("driver") or {}).get("label")
        if label:
            seen[label] = seen.get(label, 0) + 1
    return [{"label": k, "n": v} for k, v in sorted(seen.items(), key=lambda kv: -kv[1])]


# --------------------------------------------------------------------------
# Session build
# --------------------------------------------------------------------------

def build(frame: pd.DataFrame, day=None, threshold: float = THRESHOLD) -> dict:
    """All 5%+ movers for `day` (default: newest session in `frame`)."""
    dates = pd.Series(sorted(frame["date"].unique()))
    day = pd.Timestamp(day) if day is not None else dates.iloc[-1]
    if day not in set(dates):
        raise ValueError(f"{day.date()} is not a session in the loaded panel")
    prev_day = dates[dates < day].iloc[-1]

    d = frame[frame["date"] == day].copy()
    d = d[d["ret"].notna()]

    # Rolling baselines only for the movers -- a few dozen names, not 2,000.
    syms = set(d.loc[d["ret"].abs() >= threshold, "symbol"])
    h = frame[frame["symbol"].isin(syms) & (frame["date"] <= day)].sort_values(["symbol", "date"])
    g = h.groupby("symbol", sort=False)
    base = pd.DataFrame({
        "symbol": h["symbol"],
        "date": h["date"],
        "vmed": g["volume"].transform(lambda s: s.shift(1).rolling(50, min_periods=20).median()),
        "prior_hi": g["adj_high"].transform(lambda s: s.shift(1).rolling(250, min_periods=120).max()),
        "prior_lo": g["adj_low"].transform(lambda s: s.shift(1).rolling(250, min_periods=120).min()),
    })
    base = base[base["date"] == day].drop(columns="date")
    d = d.merge(base, on="symbol", how="left")
    mkt_med = float(d["ret"].median())
    n_adv, n_dec = int((d["ret"] > 0).sum()), int((d["ret"] < 0).sum())

    # Sector figures are the stock's PEERS, excluding itself -- otherwise a
    # one-stock sector "moves with its sector" by definition.
    d["sec_med"] = np.nan
    d["sec_up"] = np.nan
    d["sec_n"] = 0
    for _, idx in d.groupby("sector").groups.items():
        vals = d.loc[idx, "ret"].to_numpy()
        if len(vals) < 2:
            continue
        for pos, i in enumerate(idx):
            peers = np.delete(vals, pos)
            d.at[i, "sec_med"] = float(np.median(peers))
            d.at[i, "sec_up"] = int((peers > 0).sum())
            d.at[i, "sec_n"] = len(peers)

    d["gap"] = d["open"] / d["prev_close"] - 1.0
    rng = d["high"] - d["low"]
    d["close_loc"] = ((d["close"] - d["low"]) / rng).where(rng > 0)
    d["vol_x"] = d["volume"] / d["vmed"]
    d["deliv_base"] = d["deliv_pct"] / d["deliv_quality"] if "deliv_quality" in d.columns else np.nan
    d["mkt_med"] = mkt_med

    movers = d[d["ret"].abs() >= threshold].copy()
    up = movers["ret"] > 0
    at_high = movers["close"] >= movers["high"] * 0.999
    at_low = movers["close"] <= movers["low"] * 1.001
    snapped = movers["ret"].abs().map(_snap_band)
    movers["band"] = np.where((up & at_high) | (~up & at_low), snapped, np.nan)
    try:
        fno = filing_facts.fno_symbols()
    except Exception:
        fno = None
    if fno:
        # F&O stocks have no price band, so a close at +5% is not a circuit.
        movers.loc[movers["symbol"].isin(fno), "band"] = np.nan
    movers["new_52w"] = np.where(
        up & (movers["adj"] > movers["prior_hi"]), "high",
        np.where(~up & (movers["adj"] < movers["prior_lo"]), "low", None),
    )

    sess = None
    filings = deals = None
    try:
        sess = NSESession(timeout=20, retries=2)
    except Exception as exc:
        print(f"movers: NSE session unavailable: {exc}")
    if sess is not None:
        filings = _filings_by_symbol(sess, prev_day, day)
        deals = _deals_by_symbol(sess, day)

    hist_by = {s: g.reset_index(drop=True) for s, g in h.groupby("symbol", sort=False)}
    try:
        ttm = db.get_ttm_revenue(sorted(syms))
    except Exception:
        ttm = {}
    day_str = day.strftime("%Y-%m-%d")
    checked = "none" if filings is None else ("both" if deals is not None else "filings")

    def driver_for(r, fl: list, dl: list) -> dict | None:
        hs = hist_by.get(r["symbol"])
        if hs is None or len(hs) < 2:
            return None
        f = drivers.features(hs, len(hs) - 1)
        if f is None:
            return None
        peers = {
            "mkt": mkt_med,
            "sec": float(r["sec_med"]) if pd.notna(r["sec_med"]) else None,
            "sec_n": int(r["sec_n"]),
            "sec_up": int(r["sec_up"]) if pd.notna(r["sec_up"]) else 0,
            "sector": r["sector"],
        }
        try:
            facts = drivers.research_filings(r["symbol"], day_str, fl) if fl else {}
        except Exception as exc:
            print(f"movers: filing research failed for {r['symbol']}: {exc}")
            facts = {}
        return drivers.classify(f, peers, fl, dl if deals is not None else None, checked,
                                facts, ttm.get(r["symbol"]))

    def rows(sub: pd.DataFrame, is_up: bool) -> list[dict]:
        sub = sub.sort_values("ret", ascending=not is_up)
        out = []
        for _, r in sub.iterrows():
            fl = (filings or {}).get(r["symbol"], [])
            dl = (deals or {}).get(r["symbol"], [])
            out.append(_read(r, is_up, fl, dl, checked, driver_for(r, fl, dl)))
        return out

    gainers = rows(movers[up], True)
    losers = rows(movers[~up], False)

    def share(items, pred):
        return sum(1 for x in items if pred(x))

    return {
        "session_date": day.strftime("%Y-%m-%d"),
        "prev_session": pd.Timestamp(prev_day).strftime("%Y-%m-%d"),
        "threshold": threshold,
        "market": {"median": _f(mkt_med), "advancing": n_adv, "declining": n_dec, "n": int(len(d))},
        "filings_available": filings is not None,
        "deals_available": deals is not None,
        "counts": {
            "gainers": len(gainers),
            "losers": len(losers),
            "with_filing": share(gainers + losers, lambda x: bool(x["filings"])),
            "sector_wide": share(gainers + losers, lambda x: "Sector-wide move" in x["tags"]),
            "unexplained": share(gainers + losers, lambda x: (x["driver"] or {}).get("key") == "unexplained"
                                 if x["driver"] else not x["filings"] and not x["deals"]
                                 and "Sector-wide move" not in x["tags"]),
            "drivers": _driver_counts(gainers + losers),
        },
        "gainers": gainers,
        "losers": losers,
    }
