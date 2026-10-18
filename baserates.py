"""
Base rates: what usually happened after a move like this one.

Every 5%+ move in the cached bhavcopies is labelled by the same classifier
Insights uses (drivers.classify), with that day's exchange filings. For each
label it then measures the return from the next session's open to the close
5 and 20 sessions after the move, ranked against every stock in the panel
over the same window (percentile 50 = an ordinary outcome).

Rules, so noise does not pass as insight:
  * entry is the next open, never the move day's close;
  * returns are ranked within the same dates, because small-cap returns are
    skewed and "beat the average" is under 50% for almost anything;
  * t-stats are over dates (one mean per day), so a day with forty spikes
    counts once;
  * a pattern needs |t| >= 3 and the same sign in both halves of the window.

Deals are not available historically, so "Large buying/selling deal" never
appears here. Filing facts (results numbers, order sizes) are not read; they
change confidence, not which label wins, in all but rare cases.

    python baserates.py           # label history, print the table
    python baserates.py export    # ... and write data/base_rates.json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import drivers
import fetch
import movers
import panel as pnl
import stocks as stk

warnings.filterwarnings("ignore")

THRESHOLD = movers.THRESHOLD
HORIZONS = (5, 20)
T_REAL = 3.0
MIN_CASES, MIN_DATES = 30, 20
FILINGS_DIR = fetch.CACHE_DIR / "filings_hist"
OUT_FILE = fetch.DATA_DIR / "base_rates.json"


# --------------------------------------------------------------------------
# Panel and history
# --------------------------------------------------------------------------

def load_frame() -> pd.DataFrame:
    """The Insights frame (panel.build + add_indicators) over the whole cache."""
    raw = pd.concat([fetch._read_cache(p) for p in sorted(fetch.RAW_DIR.glob("bhav_*.*"))],
                    ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = fetch.drop_repeat_sessions(raw, verbose=False)
    smap = pd.read_json(fetch.CACHE_DIR / "sector_map.json", orient="index").rename_axis("symbol").reset_index()
    stocks, _ = pnl.build(raw, smap, sector_level="basic_industry")
    return stk.add_indicators(stocks).sort_values(["symbol", "date"]).reset_index(drop=True)


def filings_for(sess: fetch.NSESession | None, prev_day: pd.Timestamp, day: pd.Timestamp) -> dict | None:
    path = FILINGS_DIR / f"{day:%Y-%m-%d}.json"
    if path.exists():
        return json.loads(path.read_text())
    if sess is None:
        return None
    out = movers._filings_by_symbol(sess, prev_day, day)
    if out is not None:
        FILINGS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, default=str))
    return out


def label_history(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per 5%+ move: date, symbol, direction, driver key and label."""
    dates = sorted(frame["date"].unique())
    by_sym = {s: g.reset_index(drop=True) for s, g in frame.groupby("symbol", sort=False)}
    pos = {s: {d: i for i, d in enumerate(g["date"])} for s, g in by_sym.items()}
    try:
        sess = fetch.NSESession(timeout=30, retries=2)
    except Exception as exc:
        print(f"NSE unavailable, using cached filings only: {exc}")
        sess = None

    rows, t0 = [], time.time()
    # The first ~120 sessions lack the history the classifier reads (52-week
    # range, bases, EMAs), so labels there would be systematically different.
    for n, day in enumerate(dates[120:], start=120):
        prev_day = dates[n - 1]
        d = frame.loc[frame["date"] == day, ["symbol", "sector", "ret"]].dropna(subset=["ret"])
        mv = d[d["ret"].abs() >= THRESHOLD]
        if mv.empty:
            continue
        fl = filings_for(sess, pd.Timestamp(prev_day), pd.Timestamp(day))
        checked = "filings" if fl is not None else "none"
        mkt = float(d["ret"].median())
        sec_vals = d.groupby("sector")["ret"].apply(list).to_dict()
        for _, r in mv.iterrows():
            hs = by_sym[r["symbol"]]
            f = drivers.features(hs, pos[r["symbol"]][day])
            if f is None:
                continue
            peers_all = list(sec_vals.get(r["sector"], []))
            peers_all.remove(r["ret"])
            peers = {"mkt": mkt, "sector": r["sector"], "sec_n": len(peers_all),
                     "sec": float(np.median(peers_all)) if peers_all else None,
                     "sec_up": int(sum(v > 0 for v in peers_all))}
            drv = drivers.classify(f, peers, (fl or {}).get(r["symbol"], []), None, checked)
            rows.append({"date": day, "symbol": r["symbol"], "up": r["ret"] > 0,
                         "key": drv["key"], "label": drv["label"]})
        if n % 25 == 0:
            print(f"  {n}/{len(dates)} sessions, {len(rows)} moves labelled, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------

def outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Forward returns from the next open, and their rank among all stocks that day."""
    g = frame.groupby("symbol", sort=False)
    entry = g["open"].shift(-1) * (g["adj"].shift(-1) / g["close"].shift(-1))
    out = frame[["date", "symbol"]].copy()
    for k in HORIZONS:
        out[f"f{k}"] = g["adj"].shift(-k) / entry - 1
        ok = out[f"f{k}"].notna()
        out[f"p{k}"] = np.nan
        out.loc[ok, f"p{k}"] = out[ok].groupby("date")[f"f{k}"].rank(pct=True) * 100
    return out


def _t(s: pd.Series) -> float:
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 2 and s.std() > 0 else float("nan")


def summarise(ev: pd.DataFrame, k: int, mid: pd.Timestamp) -> dict | None:
    ev = ev[ev[f"f{k}"].notna()]
    by_date = ev.groupby("date")[f"p{k}"].mean() - 50
    if len(ev) < MIN_CASES or len(by_date) < MIN_DATES:
        return None
    t, t1, t2 = _t(by_date), _t(by_date[by_date.index < mid]), _t(by_date[by_date.index >= mid])
    stable = bool(abs(t) >= T_REAL and np.sign(t1) == np.sign(t) == np.sign(t2)
                  and min(abs(t1), abs(t2)) >= 1)
    return {
        "n": int(len(ev)),
        "median_ret": round(float(ev[f"f{k}"].median()) * 100, 2),
        "worst10": round(float(ev[f"f{k}"].quantile(0.10)) * 100, 1),
        "best10": round(float(ev[f"f{k}"].quantile(0.90)) * 100, 1),
        "pctile": round(50 + float(by_date.mean()), 1),
        "bottom_half": round(float((ev[f"p{k}"] < 50).mean()) * 100, 0),
        "t": round(t, 1),
        "verdict": ("lagged" if t < 0 else "beat") if stable else "mixed",
    }


def table(labels: pd.DataFrame, out: pd.DataFrame) -> dict:
    ev = labels.merge(out, on=["date", "symbol"], how="left")
    dates = out["date"].drop_duplicates().sort_values()
    mid = dates.iloc[len(dates) // 2]
    rates = {}
    for (label, up), grp in ev.groupby(["label", "up"]):
        h = {f"h{k}": summarise(grp, k, mid) for k in HORIZONS}
        if any(h.values()):
            rates[f"{label}|{'up' if up else 'down'}"] = {
                "label": label, "direction": "up" if up else "down", "key": grp["key"].iloc[0], **h}
    return rates


def print_table(rates: dict) -> None:
    rows = []
    for v in rates.values():
        for k in HORIZONS:
            s = v.get(f"h{k}")
            if s:
                rows.append({"label": v["label"], "dir": v["direction"], "days": k, **s})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).sort_values(["dir", "label", "days"]).to_string(index=False))


def run(export: bool = False) -> dict:
    frame = load_frame()
    dates = frame["date"].drop_duplicates().sort_values()
    print(f"{frame['symbol'].nunique()} stocks, {len(dates)} sessions "
          f"({dates.iloc[0]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d})")
    labels = label_history(frame)
    print(f"{len(labels)} moves labelled\n")
    rates = table(labels, outcomes(frame))
    print_table(rates)
    if export:
        start = labels["date"].min()
        OUT_FILE.write_text(json.dumps({
            "computed_at": datetime.now().strftime("%Y-%m-%d"),
            "window": {"start": f"{start:%Y-%m-%d}", "end": f"{dates.iloc[-1]:%Y-%m-%d}"},
            "horizons": list(HORIZONS),
            "method": ("Every 5%+ move since the window start, labelled by the Insights classifier. "
                       "Return from the next session's open; percentile ranks it among all stocks "
                       "over the same days (50 = ordinary)."),
            "rates": rates,
        }, indent=1))
        print(f"\nwrote {OUT_FILE}")
    return rates


if __name__ == "__main__":
    run(export="export" in sys.argv[1:])
