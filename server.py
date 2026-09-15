#!/usr/bin/env python3
"""
Flask API for the sector scan + coil watchlist.

    cd backend && python server.py

Then, in another terminal:

    cd ui && npm install && npm run dev

The UI proxies /api here (port 5050). Refresh in the browser rebuilds from
the latest cached bhavcopy (and fetches any missing session from NSE).
"""

from __future__ import annotations

import math
import threading
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import analyze
import auth
import breakouts as bo
import db
import fetch
import flagslog
import panel as pnl
import scan as sc
import stocks as stk
import position as posscan
import tom as tomscan

BACKEND = Path(__file__).resolve().parent
REPO = BACKEND.parent
UI_DIST = REPO / "ui" / "dist"

# 290, not 220: Position Trades needs a 12-1 momentum reading, which is a
# 250-session window ending 20 sessions ago = 270 sessions of a symbol's own
# history before it exists at all. 290 leaves a small margin. Everything else
# in the pipeline needs at most 210, so this is the binding constraint.
DEFAULT_DAYS = 290
DEFAULT_LEVEL = "basic_industry"

# Two different questions, so two named sets rather than inline lists that
# drift apart. WATCH is what the Shortlisted count reports — BASE belongs there
# because quiet accumulation is worth following. SETUP is the narrower set a
# coil's sector must be in to reach the Setups list, and BASE is excluded
# because there has been no expansion yet to pull back from.
WATCH_KLASSES = ("CROSSING", "PULLBACK", "BASE")
SETUP_KLASSES = ("CROSSING", "PULLBACK")

# Slack around the scorer's momentum zone when picking which symbols are
# worth a live quote. Wider than tom.MOM_ZONE_* so an intraday move cannot
# push a name into the zone after we have already decided not to quote it.
LIVE_ZONE_ABOVE = 0.20
LIVE_ZONE_BELOW = 0.25
LIVE_MIN_PRICE = 15.0

# Delivery percent is only known at settlement, so a live session has none.
# Leaving it blank fails gate 3, which would erase every PULLBACK sector and
# empty the Setups funnel on each refresh. Carrying the symbol's previous
# value keeps deliv_quality on its 60-day baseline; it is an approximation
# and the UI says so.
CARRY_FORWARD_DELIVERY = True

# NSE continuous trading. Outside this window a quote returns the closing
# price, which is exactly what we want — no special casing needed beyond
# labelling it honestly.
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def market_is_open(now=None) -> bool:
    now = db.now_ist(now)
    return now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_CLOSE


def _py(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (datetime, date)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    x = df.copy()
    for c in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = pd.to_datetime(x[c]).dt.strftime("%Y-%m-%d")
    x = x.replace({np.nan: None, np.inf: None, -np.inf: None})
    return [{k: _py(v) for k, v in row.items()} for row in x.to_dict(orient="records")]


def _live_candidates(stocks: pd.DataFrame) -> list[str]:
    """
    Which symbols are worth a live quote.

    The scorer only keeps uptrend names sitting inside a narrow band around
    the trigger, and that trend gate reads EOD columns only. Applying the
    same shape to yesterday's close, widened, drops the quote list from the
    whole universe to a few hundred — the difference between ~34 Groww
    batches and a handful.
    """
    if stocks is None or stocks.empty:
        return []
    eod = stocks[stocks["date"] == stocks["date"].max()]
    if eod.empty:
        return []

    adj = pd.to_numeric(eod["adj"], errors="coerce")
    trig = pd.to_numeric(eod["trigger"], errors="coerce")
    if "prior_trigger" in eod.columns:
        trig = pd.to_numeric(eod["prior_trigger"], errors="coerce").fillna(trig)
    gap = trig / adj.replace(0, np.nan) - 1.0

    keep = gap.between(-LIVE_ZONE_ABOVE, LIVE_ZONE_BELOW) & (adj >= LIVE_MIN_PRICE)
    if {"ema50", "ema200"}.issubset(eod.columns):
        keep &= (adj > eod["ema50"]) & (eod["ema50"] > eod["ema200"])
    return eod.loc[keep.fillna(False), "symbol"].astype(str).unique().tolist()


def _quote_universe(coil_stocks: pd.DataFrame, raw: pd.DataFrame) -> list[str]:
    """Names worth a live quote: For Tom candidates plus yesterday's list."""
    want = _live_candidates(coil_stocks)
    try:
        dates = db.get_prediction_dates()
        vd = db.verify_scan_date(dates)
        if vd:
            for p in db.get_predictions_on(vd):
                s = p.get("symbol")
                if s:
                    want.append(str(s))
    except Exception as exc:
        print(f"quote universe preds: {exc}")
    seen, out = set(), []
    for s in want:
        s = str(s).strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if out:
        return out
    if raw is None or raw.empty:
        return []
    last = raw[raw["date"] == raw["date"].max()]
    last = last[last["series"].astype(str).str.upper() == "EQ"]
    return sorted(last["symbol"].astype(str).unique().tolist())


def _synthetic_session(raw: pd.DataFrame, quotes: pd.DataFrame,
                       session: pd.Timestamp) -> pd.DataFrame:
    """
    Shape live quotes into one more bhavcopy session.

    Appending this to the raw history and re-running pnl.build is what lets
    the sector panel refresh on live prices — T, T_rel, B and CMF all fall
    out of the existing pipeline, which needs to know nothing about live data.
    """
    if raw is None or raw.empty or quotes is None or quotes.empty:
        return pd.DataFrame()

    prev = raw[raw["date"] == raw["date"].max()]
    prev = prev[prev["series"].astype(str).str.upper() == "EQ"]
    prev = prev.drop_duplicates("symbol").set_index("symbol")

    q = quotes[quotes["symbol"].isin(prev.index)].copy()
    if q.empty:
        return pd.DataFrame()

    close = pd.to_numeric(q["ltp"], errors="coerce")
    prev_close = pd.to_numeric(q["prev_close"], errors="coerce")
    prev_close = prev_close.fillna(q["symbol"].map(prev["close"]))

    out = pd.DataFrame({
        "symbol": q["symbol"].values,
        "series": "EQ",
        "date": session,
        "prev_close": prev_close.values,
        "open": pd.to_numeric(q["open"], errors="coerce").values,
        "high": pd.to_numeric(q["high"], errors="coerce").values,
        "low": pd.to_numeric(q["low"], errors="coerce").values,
        "last": close.values,
        "close": close.values,
        "vwap": pd.to_numeric(q["avg_price"], errors="coerce").values,
        "volume": pd.to_numeric(q["volume"], errors="coerce").values,
        "turnover": pd.to_numeric(q["turnover"], errors="coerce").values,
        "trades": np.nan,
        "deliv_qty": np.nan,
        "deliv_pct": (q["symbol"].map(prev["deliv_pct"]).values
                      if CARRY_FORWARD_DELIVERY and "deliv_pct" in prev.columns
                      else np.nan),
        "source": "live",
    })
    out = out[(out["close"] > 0) & (out["volume"] > 0) & (out["turnover"] > 0)]
    return out.reset_index(drop=True)


def _verify_yesterday(quotes: pd.DataFrame | None = None,
                      stocks: pd.DataFrame | None = None) -> int:
    """
    Score yesterday's For Tom list against today's high.

    Only runs when the market has been open for the verification session —
    otherwise the "today's high" is really yesterday's close and the outcome
    is meaningless.
    """
    # Don't verify with stale prices before the market opens.
    if not market_is_open():
        # After-hours is okay (we have today's real high), pre-market is not.
        now = db.now_ist()
        if now.time() < MARKET_OPEN:
            return 0

    dates = db.get_prediction_dates()
    scan_d = db.verify_scan_date(dates)
    verified = 0
    if scan_d:
        q = quotes
        if q is None or q.empty:
            preds = db.get_predictions_on(scan_d)
            symbols = [p["symbol"] for p in preds if p.get("symbol")]
            if symbols:
                try:
                    q = fetch.live_quotes_full(symbols)
                except Exception as exc:
                    print(f"verify quotes: {exc}")
                    try:
                        q = fetch.live_quotes_groww(symbols)
                    except Exception as exc2:
                        print(f"verify quotes fallback: {exc2}")
                        q = pd.DataFrame()
        if q is not None and not q.empty:
            verified += db.verify_tom_from_quotes(scan_d, q, db.session_date())

    panel = stocks if stocks is not None else engine.coil_stocks
    if panel is not None and not panel.empty:
        hist = sorted(panel["date"].unique())
        for i in range(min(5, max(0, len(hist) - 1))):
            pred_date = hist[-(i + 2)]
            if scan_d and str(pred_date)[:10] == str(scan_d)[:10] and verified:
                continue
            try:
                verified += db.verify_tom_outcomes(pred_date, panel)
            except Exception as exc:
                print(f"verify eod {pred_date}: {exc}")
    return verified


class Engine:
    """One in-memory panel, rebuilt on Refresh."""

    def __init__(self):
        self._lock = threading.Lock()
        self._busy = threading.Lock()
        self.status = "idle"
        self.message = "Waiting to load."
        self.error = None
        self.days = DEFAULT_DAYS
        self.end = date.today()
        self.sector_level = DEFAULT_LEVEL
        self.loaded_at = None
        self.as_of = None
        self.n_stocks = 0
        self.n_sectors = 0
        self.n_sessions = 0
        self.stocks = None
        self.panel = None
        self.coil_stocks = None
        self.scan_rows = pd.DataFrame()
        self.coil_rows = pd.DataFrame()
        self.rest_rows = pd.DataFrame()
        self.near_miss = pd.DataFrame()
        self.buys = pd.DataFrame()
        self.breakouts = pd.DataFrame()
        self.tom = pd.DataFrame()
        self.position_rows = pd.DataFrame()
        self.live_at = None
        self.live_n = 0
        self.live_status = None
        self.live_source = None
        self.live_sectors = pd.DataFrame()
        self.sectors_live = False
        self.raw = None
        self.smap = None
        self.sector_level = DEFAULT_LEVEL
        self.delivery = {"as_of_ok": True, "missing": 0, "missing_dates": []}
        self.verdict_by_sector = {}
        self.tom_preview = []  # Preliminary results during scan

    def snapshot(self) -> dict:
        with self._lock:
            actionable = 0
            bullish_sectors = 0
            total_sectors = 0
            if not self.scan_rows.empty and "klass" in self.scan_rows:
                actionable = int(self.scan_rows["klass"].isin(WATCH_KLASSES).sum())
                total_sectors = len(self.scan_rows)
                bullish_sectors = int(
                    self.scan_rows["klass"].isin(SETUP_KLASSES).sum()
                )
            # Market regime: bullish if > 40% of sectors are in uptrend
            regime_ok = total_sectors == 0 or (bullish_sectors / total_sectors >= 0.4)
            return {
                "status": self.status,
                "message": self.message,
                "error": self.error,
                "days": self.days,
                "end": self.end.isoformat(),
                "as_of": self.as_of,
                "n_stocks": self.n_stocks,
                "n_sectors": self.n_sectors,
                "n_sessions": self.n_sessions,
                "n_coil": int(len(self.coil_rows)),
                "n_buys": int(len(self.buys)),
                "n_tom": int(len(self.tom)),
                "n_position": int(len(self.position_rows)),
                "live_at": self.live_at,
                "live_n": int(self.live_n),
                "live_status": self.live_status,
                "live_source": self.live_source,
                "sectors_live": bool(self.sectors_live),
                "delivery_carried": bool(self.sectors_live and CARRY_FORWARD_DELIVERY),
                "actionable": actionable,
                "loaded_at": self.loaded_at,
                "coil_ready": self.n_sessions >= 210,
                "delivery_ok": bool(self.delivery.get("as_of_ok", True)),
                "delivery_missing": int(self.delivery.get("missing", 0)),
                "delivery_missing_dates": list(self.delivery.get("missing_dates") or []),
                "tom_preview": self.tom_preview,
                "job_refresh_at": _job_state("refresh_at"),
                "job_scan_at": _job_state("scan_at"),
                "regime_ok": regime_ok,
                "bullish_sectors": bullish_sectors,
                "total_sectors": total_sectors,
            }

    def _set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def load(self, days: int, end: date, sector_level: str = DEFAULT_LEVEL):
        if not self._busy.acquire(blocking=False):
            return False
        try:
            self._run(days, end, sector_level)
            return True
        finally:
            self._busy.release()

    def _run(self, days: int, end: date, sector_level: str):
        self._set(status="loading", message="Checking sector map…",
                  error=None, days=days, end=end, sector_level=sector_level)

        smap_path = fetch.CACHE_DIR / "sector_map.json"
        if not smap_path.exists():
            self._set(status="error",
                      error="No sector map. Run `python run.py sectors` first.",
                      message="Sector map missing.")
            return

        smap = pd.read_json(smap_path, orient="index").rename_axis("symbol").reset_index()
        filled = smap[sector_level].notna() & (smap[sector_level].astype(str).str.strip() != "")
        if int(filled.sum()) == 0:
            self._set(status="error",
                      error="Sector map is empty. Re-run `python run.py sectors`.",
                      message="Sector map has no industries.")
            return

        def progress(seen, total, d):
            self._set(message=f"Loading sessions… {seen}/{total}  ({d:%d-%m-%Y})")

        self._set(message=f"Fetching {days} sessions ending {end:%Y-%m-%d}…")
        raw = fetch.load_history(end, days, verbose=False, on_progress=progress)

        self._set(message="Building sector panel…")
        stocks, panel = pnl.build(raw, smap, sector_level=sector_level)
        if stocks.empty:
            self._set(status="error",
                      error="Nothing survived cleaning. Check the sector map and EQ filter.",
                      message="Empty panel.")
            return

        self._set(message="Classifying sectors…")
        scan_rows = sc.classify(panel)

        self._set(message="Computing coil indicators…")
        coil_stocks = stk.add_indicators(stocks)
        # 12-1 momentum lands here once so both the ranked coil list and the
        # position scan read it off the same frame.
        coil_stocks = posscan.add_position_features(coil_stocks)
        coil_all = stk.scan(coil_stocks, top=10_000)
        miss = stk.near_miss(coil_stocks)

        self._set(message="Scoring buy setups…")
        ready = sc.recommend_sectors(scan_rows, panel)
        scan_rows = scan_rows.copy()
        scan_rows["buy_ready"] = scan_rows["sector"].isin(ready)
        
        # Build sector classification lookup
        sector_klass_map = dict(zip(scan_rows["sector"], scan_rows["klass"]))
        
        if coil_all.empty:
            coil_all = coil_all.copy()
            coil_all["recommended"] = pd.Series(dtype=bool)
            coil_all["sector_klass"] = pd.Series(dtype=str)
            buys = coil_all
        else:
            coil_all = coil_all.copy()
            # `recommended` = the sector's shape checks all passed, which is the
            # funnel buytest.py measures. Kept as a column, not a filter, so the
            # UI can rank by confidence without hiding the wider set.
            coil_all["recommended"] = coil_all["sector"].isin(ready)
            coil_all["sector_klass"] = coil_all["sector"].map(sector_klass_map).fillna("")
            setup_sectors = set(
                scan_rows[scan_rows["klass"].isin(SETUP_KLASSES)]["sector"]
            )
            buys = coil_all[coil_all["sector"].isin(setup_sectors)].copy()
        # No cap: the seven gates already did the filtering, and ranking by
        # `coil` measured no relationship with forward returns, so cutting
        # the list at 40 by that score was discarding names arbitrarily.
        coil_rows = coil_all
        # The shown list: same pool, ordered by 12-1 momentum and cut to 20.
        # See eval_listsize.py for why 20 rather than the full 63 or a
        # single-digit shortlist.
        rest_rows = posscan.leaders_at_rest(coil_rows, coil_stocks)

        as_of = panel["date"].max()
        as_of_s = pd.Timestamp(as_of).strftime("%Y-%m-%d")

        # Delivery coverage. UDiFF fallback leaves deliv_pct all-NaN, which
        # blanks delivery quality, B_deliv and every delivery gate. Surface it.
        delivery = fetch.delivery_coverage(stocks)

        # Shape report per actionable / buy-ready sector — verdict for the
        # log, and the plain-language "why" on each buy setup.
        shapes = {}
        verdict_by_sector = {}
        watch = scan_rows[
            scan_rows["klass"].isin(["CROSSING", "PULLBACK", "CROSSING_UNVERIFIED"])
            | scan_rows["buy_ready"].fillna(False)
        ]
        for sec in watch["sector"]:
            _, shp = sc.shape_report(panel[panel["sector"] == sec])
            shapes[sec] = shp
            verdict_by_sector[sec] = shp.get("verdict", "")

        if not buys.empty:
            buys = buys.copy()
            buys["why"] = buys.apply(
                lambda r: sc.explain_setup(shapes.get(r["sector"], {}), r),
                axis=1,
            )

        self._set(message="Checking recent breakouts…")
        broke = pd.DataFrame()
        try:
            broke = bo.find_breakouts(panel, coil_stocks, as_of)
        except Exception as exc:
            print(f"breakout scan failed: {exc}")

        with self._lock:
            self.raw = raw
            self.smap = smap
            self.stocks = stocks
            self.panel = panel
            self.coil_stocks = coil_stocks
            self.scan_rows = scan_rows
            # A fresh EOD load supersedes any earlier live reclassification.
            self.live_sectors = pd.DataFrame()
            self.sectors_live = False
            self.coil_rows = coil_rows
            self.rest_rows = rest_rows
            self.near_miss = miss
            self.buys = buys
            self.breakouts = broke
            self.delivery = delivery
            self.verdict_by_sector = verdict_by_sector
            self.as_of = as_of_s
            self.n_stocks = int(stocks["symbol"].nunique())
            self.n_sectors = int(panel["sector"].nunique())
            self.n_sessions = int(panel["date"].nunique())
            self.loaded_at = datetime.now().isoformat(timespec="seconds")
            self.status = "ready"
            self.message = f"Ready · {as_of_s}"
            self.error = None

        self._restore_tom()

        # Persist today's setups outside the lock (file IO). Idempotent per
        # as_of, so a repeated Refresh does not duplicate rows.
        try:
            flagslog.append_day(as_of_s, buys, scan_rows, verdict_by_sector)
        except Exception as exc:  # logging must never break a load
            print(f"flags log append failed: {exc}")

    def _restore_tom(self):
        """Load the last saved For Tom list so a restart does not require a rescan."""
        try:
            scan_d = db.latest_tom_scan_date()
            if not scan_d:
                return
            preds = db.get_predictions_on(scan_d)
            rows = db.tom_frame_from_predictions(preds)
            if rows.empty:
                return
            created = [p.get("created_at") for p in preds if p.get("created_at")]
            live_at = max(created) if created else f"{scan_d}T00:00:00"
            with self._lock:
                if self.live_status == "loading":
                    return
                self.tom = rows
                self.live_at = str(live_at)[:19]
                self.live_n = int(len(rows))
                self.live_status = "ready"
                if not self.live_source:
                    self.live_source = "saved"
        except Exception as exc:
            print(f"restore tom failed: {exc}")

    def dashboard(self) -> dict:
        snap = self.snapshot()
        if snap["status"] != "ready":
            return snap
        with self._lock:
            scan_cols = [c for c in [
                "sector", "klass", "T", "T_rel", "B", "B_deliv",
                "deliv_quality_rel", "cmf", "cmf_rel", "rs", "rs_chg_5",
                "n_stocks", "n_adv", "top_share", "note", "date", "buy_ready",
            ] if c in self.scan_rows.columns]
            return {
                **snap,
                "scan": records(self.scan_rows[scan_cols] if scan_cols else self.scan_rows),
                "coil": records(self.coil_rows),
                "rest": records(self.rest_rows),
                "near_miss": records(self.near_miss),
                "buys": records(self.buys),
                "tom": records(self.tom),
                "delivery_missing_dates": list(self.delivery.get("missing_dates") or []),
                "delivery_missing_detail": list(self.delivery.get("missing_detail") or []),
            }

    def sector(self, name: str) -> dict | None:
        with self._lock:
            if self.status != "ready" or self.panel is None:
                return None
            hist = self.panel[self.panel["sector"].str.lower() == name.lower()].sort_values("date")
            if hist.empty:
                near = sorted({s for s in self.panel["sector"].unique()
                               if name.lower()[:6] in s.lower()})
                return {"found": False, "near": near}

            last = self.stocks[
                (self.stocks["sector"].str.lower() == name.lower())
                & (self.stocks["date"] == self.stocks["date"].max())
            ].sort_values("turnover", ascending=False).head(20)

            annotated, shape = sc.shape_report(hist)
            hcols = [c for c in [
                "date", "T", "T_rel", "B", "B_deliv", "deliv_quality_rel",
                "cmf", "cmf_rel", "rs", "rs_chg_5", "n_adv", "n_stocks",
                "top_share", "ret", "mark",
            ] if c in annotated.columns]
            scols = [c for c in [
                "symbol", "close", "ret", "turnover", "deliv_pct",
                "deliv_quality", "cmf",
            ] if c in last.columns]

            klass, note = "", ""
            if not self.scan_rows.empty:
                hit = self.scan_rows[self.scan_rows["sector"].str.lower() == name.lower()]
                if not hit.empty:
                    klass = hit.iloc[0].get("klass", "")
                    note = hit.iloc[0].get("note", "")

            recommend = bool(shape.get("buy_ready") and klass == "PULLBACK")
            shape = {**shape, "recommend": recommend}
            buys = pd.DataFrame()
            if self.buys is not None and not self.buys.empty:
                buys = self.buys[self.buys["sector"].str.lower() == name.lower()]

            return {
                "found": True,
                "sector": hist["sector"].iloc[0],
                "klass": klass,
                "note": note,
                "shape": shape,
                "buys": records(buys),
                "history": records(annotated[hcols]),
                "constituents": records(last[scols]),
            }

    def symbols(self) -> list[dict]:
        with self._lock:
            if self.status != "ready" or self.stocks is None:
                return []
            last = self.stocks[self.stocks["date"] == self.stocks["date"].max()]
            cols = [c for c in ["symbol", "sector"] if c in last.columns]
            out = last[cols].drop_duplicates("symbol").sort_values("symbol")
            return records(out)

    def scan_all(self) -> bool:
        """
        Unified refresh: sectors, coils, and For Tom.

        - Market open / no bhavcopy yet: quote the For Tom neighborhood and
          overlay last prices on the last EOD panel (sectors stay EOD).
        - Official bhavcopy for today and market closed: rebuild everything.
        """
        if not self._busy.acquire(blocking=False):
            return False
        try:
            with self._lock:
                if self.status != "ready" or self.coil_stocks is None:
                    return False
                raw = self.raw
                smap = self.smap
                sector_level = getattr(self, "sector_level", DEFAULT_LEVEL)
                coil_eod = self.coil_stocks
                scan_eod = self.scan_rows
                self.live_status = "loading"
                self.message = "Starting scan…"
                self.tom_preview = []  # Clear preview at start

            session = pd.Timestamp(db.session_date())
            open_now = market_is_open()
            have_session_eod = (raw is not None and not raw.empty
                                and bool((raw["date"] == session).any()))
            quotes = None
            rebuild = False

            # Determine what data source to use
            if have_session_eod and not open_now:
                # Official bhavcopy loaded and market closed — best data
                source = "eod"
                rebuild = True
                self._set(message=f"Using official bhavcopy ({session:%d %b})…")
                raw_final = raw
            else:
                # Live overlay: quote the For Tom neighborhood, not the universe.
                # A partial synthetic session would corrupt sector breadth, so
                # sectors/coils stay on last EOD and only For Tom updates.
                universe = _quote_universe(coil_eod, raw)

                label = "LIVE" if open_now else "closing"
                self._set(message=f"[1/2] Fetching {label} prices… 0 of {len(universe):,}")

                # Preliminary ranking callback - updates tom_preview as data comes in
                def on_batch(partial_df):
                    if partial_df is None or partial_df.empty:
                        return
                    try:
                        # Simple preliminary score: price change % + volume boost
                        df = partial_df.copy()
                        df["pchange"] = (
                            (df["ltp"] - df["prev_close"]) / df["prev_close"] * 100
                        ).fillna(0)
                        # Filter: only positive movers with decent volume
                        df = df[(df["pchange"] > 0) & (df["volume"] > 10000)]
                        # Sort by pchange descending, take top 20
                        df = df.nlargest(20, "pchange")
                        preview = []
                        for _, r in df.iterrows():
                            preview.append({
                                "symbol": r["symbol"],
                                "pchange": round(r["pchange"], 2),
                                "volume": int(r["volume"]),
                                "ltp": round(r["ltp"], 2),
                            })
                        with self._lock:
                            self.tom_preview = preview
                    except Exception as e:
                        print(f"preview error: {e}")

                quotes = pd.DataFrame()
                try:
                    quotes = fetch.live_quotes_full(
                        universe,
                        on_progress=lambda done, total, lbl=label: self._set(
                            message=f"[1/2] Fetching {lbl} prices… {done:,} of {total:,}"
                        ),
                        on_batch=on_batch,
                    )
                except Exception as exc:
                    self._set(message=f"Quote fetch failed — {str(exc)[:40]}")
                    print(f"groww quotes failed: {exc}")

                if quotes is None or quotes.empty:
                    if have_session_eod:
                        source = "eod"
                        rebuild = True
                        raw_final = raw
                        self._set(message="Quotes failed, falling back to bhavcopy…")
                    else:
                        source = "eod_stale"
                        rebuild = True
                        raw_final = raw
                        self._set(message="No quotes or bhavcopy — using stale data")
                else:
                    source = "live" if open_now else "close"
                    rebuild = False

            if rebuild:
                self._set(message="Building sector panel…")
                stocks, panel = pnl.build(raw_final, smap, sector_level=sector_level)

                self._set(message="Classifying sectors…")
                scan_rows = sc.classify(panel)

                self._set(message="Computing coil indicators…")
                coil_stocks = stk.add_indicators(stocks)
                coil_all = stk.scan(coil_stocks, top=10_000)
                miss = stk.near_miss(coil_stocks)

                self._set(message="Scoring buy setups…")
                ready = sc.recommend_sectors(scan_rows, panel)
                scan_rows = scan_rows.copy()
                scan_rows["buy_ready"] = scan_rows["sector"].isin(ready)

                sector_klass_map = dict(zip(scan_rows["sector"], scan_rows["klass"]))
                if coil_all.empty:
                    coil_all = coil_all.copy()
                    coil_all["recommended"] = pd.Series(dtype=bool)
                    coil_all["sector_klass"] = pd.Series(dtype=str)
                    buys = coil_all
                else:
                    coil_all = coil_all.copy()
                    coil_all["recommended"] = coil_all["sector"].isin(ready)
                    coil_all["sector_klass"] = coil_all["sector"].map(sector_klass_map).fillna("")
                    setup_sectors = set(
                        scan_rows[scan_rows["klass"].isin(SETUP_KLASSES)]["sector"]
                    )
                    buys = coil_all[coil_all["sector"].isin(setup_sectors)].copy()
                # No cap: the seven gates already did the filtering, and ranking by
                # `coil` measured no relationship with forward returns, so cutting
                # the list at 40 by that score was discarding names arbitrarily.
                coil_rows = coil_all
                rest_rows = posscan.leaders_at_rest(coil_rows, coil_stocks)

                # Same-session close: LTP is today's adj, so tom uses prior_trigger.
                as_of = coil_stocks["date"].max()
                eod = coil_stocks[coil_stocks["date"] == as_of].copy()
                live = eod[["symbol"]].copy()
                live["ltp"] = eod["adj"]
                live["volume"] = eod["volume"]
                live["pchange"] = eod["ret"] * 100 if "ret" in eod.columns else 0
                live["high"] = eod["adj_high"] if "adj_high" in eod.columns else eod["adj"]

                self._set(message="Scoring For Tom…")
                tom_rows = tomscan.for_tomorrow_momentum(
                    coil_stocks, live, scan_rows=scan_rows
                )
            else:
                self._set(message="[2/2] Scoring For Tom…")
                coil_stocks = coil_eod
                scan_rows = scan_eod
                stocks = None
                panel = None
                coil_rows = None
                rest_rows = None
                miss = None
                buys = None
                as_of = coil_stocks["date"].max()
                live = quotes
                tom_rows = tomscan.for_tomorrow_momentum(
                    coil_stocks, quotes, scan_rows=scan_rows
                )

            # Position Trades: EOD only. The 7-10 session horizon means an
            # intraday quote adds nothing, so this always runs off closes and
            # is unaffected by which live source the tom scan used.
            position_rows = pd.DataFrame()
            try:
                if coil_stocks is not None and not coil_stocks.empty:
                    position_rows = posscan.scan(
                        posscan.add_position_features(coil_stocks), as_of=as_of
                    )
            except Exception as e:
                print(f"position scan failed: {e}")

            # Save today's picks, and score yesterday's against today's high
            saved = 0
            verified = 0
            if source != "eod_stale":
                try:
                    if not tom_rows.empty:
                        saved = db.save_tom_predictions(session, tom_rows)
                except Exception as e:
                    print(f"Failed to save tom predictions: {e}")
                try:
                    live_quotes = quotes if source in ("live", "close") else None
                    verified = _verify_yesterday(live_quotes, coil_stocks)
                except Exception as e:
                    print(f"Failed to verify yesterday: {e}")

            delivery = fetch.delivery_coverage(stocks) if stocks is not None else None

            as_of_s = pd.Timestamp(as_of).strftime("%Y-%m-%d")
            with self._lock:
                if rebuild and stocks is not None:
                    self.stocks = stocks
                    self.panel = panel
                    self.coil_stocks = coil_stocks
                    self.scan_rows = scan_rows
                    self.coil_rows = coil_rows
                    self.rest_rows = rest_rows
                    self.near_miss = miss
                    self.buys = buys
                    self.as_of = as_of_s
                    self.n_stocks = int(stocks["symbol"].nunique())
                    self.n_sectors = (
                        int(scan_rows["sector"].nunique()) if not scan_rows.empty else 0
                    )
                    if delivery is not None:
                        self.delivery = delivery
                self.tom = tom_rows
                if not position_rows.empty:
                    self.position_rows = position_rows
                self.live_n = int(len(live) if live is not None else 0)
                self.live_at = datetime.now().isoformat(timespec="seconds")
                self.live_status = "ready"
                self.live_source = source
                # Overlay quotes do not rebuild the sector panel.
                self.sectors_live = False
                self.error = None
                self.tom_preview = []

                db_msg = f" · {saved} saved" if saved else ""
                if verified:
                    db_msg += f" · {verified} verified"
                n_coil = int(len(self.coil_rows) if self.coil_rows is not None else 0)
                n_tom = int(len(tom_rows))
                n_quotes = int(len(quotes)) if quotes is not None and not quotes.empty else 0

                if source == "eod_stale":
                    self.message = (
                        f"Stale data · no quotes and no bhavcopy · "
                        f"{n_coil} coils · {n_tom} for tom · not saved"
                    )
                elif source == "eod":
                    self.message = (
                        f"Official close {session:%d-%m} · "
                        f"{n_coil} coils · {n_tom} for tom{db_msg}"
                    )
                else:
                    label = "Live" if source == "live" else "Close"
                    self.message = (
                        f"Done — {label} overlay · {n_quotes} quotes · "
                        f"{n_tom} for tom{db_msg}"
                    )

            return True
        except Exception as exc:
            import traceback
            traceback.print_exc()
            with self._lock:
                self.live_status = "error"
            self._set(message=f"Scan failed: {str(exc)[:50]}", error=str(exc))
            return False
        finally:
            self._busy.release()

    def stock(self, query: str, entry=None) -> dict:
        with self._lock:
            if self.status != "ready" or self.stocks is None:
                return {"found": False, "query": query, "error": "not ready"}
            raw = analyze.build(
                query,
                stocks=self.coil_stocks if self.coil_stocks is not None else self.stocks,
                panel=self.panel,
                scan_rows=self.scan_rows,
                buys=self.buys,
                breakouts=self.breakouts,
                as_of=self.as_of,
                entry=entry,
            )
            return _clean(raw)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return _py(obj)


engine = Engine()
app = Flask(__name__, static_folder=None)
application = app  # gunicorn / Elastic Beanstalk: application:application
CORS(app, resources={r"/api/*": {"origins": "*"}})
# Auth middleware — requires sign-in for protected routes
app.before_request(auth.before_request)


def _parse_end(raw) -> date:
    if not raw:
        return date.today()
    return datetime.strptime(str(raw), "%Y-%m-%d").date()


@app.get("/api/auth/config")
def api_auth_config():
    """Public: URL + anon key the UI needs to start Google sign-in."""
    return jsonify(auth.public_config())


@app.get("/api/me")
def api_me():
    return jsonify(auth.me_payload())


@app.get("/api/status")
def api_status():
    return jsonify(engine.snapshot())


@app.get("/api/dashboard")
def api_dashboard():
    snap = engine.snapshot()
    if snap["status"] != "ready":
        return jsonify(snap), 202
    return jsonify(engine.dashboard())


@app.post("/api/refresh")
def api_refresh():
    body = request.get_json(silent=True) or {}
    days = int(body.get("days") or request.args.get("days") or engine.days or DEFAULT_DAYS)
    days = max(60, min(days, 400))
    try:
        end = _parse_end(body.get("end") or request.args.get("end"))
    except ValueError:
        return jsonify({"error": "end must be YYYY-MM-DD"}), 400

    def work():
        try:
            engine.load(days, end)
        except Exception as exc:
            engine._set(status="error", error=str(exc), message="Load failed.")

    if engine._busy.locked():
        return jsonify(engine.snapshot()), 409

    threading.Thread(target=work, daemon=True).start()
    # Give the worker a moment to flip status so the client does not
    # immediately think it is still idle.
    return jsonify({**engine.snapshot(), "status": "loading",
                    "message": f"Refresh started · {days} sessions"}), 202


@app.get("/api/flags")
def api_flags():
    """The persistent flags log — buy setups and sector verdicts."""
    flags = flagslog.load_flags()
    shapes = flagslog.load_shapes()
    return jsonify({
        "n": int(len(flags)),
        "n_shapes": int(len(shapes)),
        "flags": records(flags),
        "shapes": records(shapes),
    })


@app.post("/api/scan")
@app.post("/api/tom")  # backward compatibility
def api_scan():
    """
    Unified refresh: sectors, coils, and For Tom.
    Uses live quotes if market is open, closing quotes after hours,
    or official bhavcopy if already loaded.
    """
    snap = engine.snapshot()
    if snap["status"] != "ready":
        return jsonify(snap), 202
    if engine._busy.locked():
        return jsonify(engine.snapshot()), 409

    engine._set(message="Starting scan…")
    with engine._lock:
        engine.live_status = "loading"

    def work():
        engine.scan_all()

    threading.Thread(target=work, daemon=True).start()
    return jsonify({**engine.snapshot(),
                    "message": "Scan started"}), 202


@app.get("/api/symbols")
def api_symbols():
    snap = engine.snapshot()
    if snap["status"] != "ready":
        return jsonify(snap), 202
    return jsonify({"symbols": engine.symbols()})


@app.get("/api/quote")
def api_quote():
    """One-symbol Groww price check. Public so you can curl it on the server."""
    symbol = (request.args.get("symbol") or request.args.get("q") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required, e.g. /api/quote?symbol=RELIANCE"}), 400
    try:
        quotes = fetch.live_quotes_groww([symbol])
    except Exception as exc:
        return jsonify({"ok": False, "source": "groww", "symbol": symbol, "error": str(exc)}), 502
    if quotes is None or quotes.empty:
        return jsonify({
            "ok": False,
            "source": "groww",
            "symbol": symbol,
            "error": "Groww returned no quote for this symbol",
        }), 404
    row = quotes.iloc[0].to_dict()
    return jsonify({
        "ok": True,
        "source": "groww",
        "symbol": row.get("symbol") or symbol,
        "ltp": _py(row.get("ltp")),
        "open": _py(row.get("open")),
        "high": _py(row.get("high")),
        "low": _py(row.get("low")),
        "prev_close": _py(row.get("prev_close")),
        "pchange": _py(row.get("pchange")),
    })


@app.get("/api/stock")
def api_stock():
    q = (request.args.get("q") or request.args.get("symbol") or "").strip()
    if not q:
        return jsonify({"error": "q is required"}), 400
    snap = engine.snapshot()
    if snap["status"] != "ready":
        return jsonify(snap), 202
    raw_entry = request.args.get("entry")
    entry = None
    if raw_entry not in (None, ""):
        try:
            entry = float(raw_entry)
        except ValueError:
            return jsonify({"error": "entry must be a number"}), 400
    return jsonify(engine.stock(q, entry=entry))


@app.get("/api/sector")
def api_sector():
    name = (request.args.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    snap = engine.snapshot()
    if snap["status"] != "ready":
        return jsonify(snap), 202
    out = engine.sector(name)
    if out is None:
        return jsonify(snap), 202
    return jsonify(out)


# ---------------------------------------------------------------------------
# Sector Lookouts (Section 1: Post-Market Sector Analysis)
# ---------------------------------------------------------------------------

@app.get("/api/sector-lookouts")
def api_sector_lookouts():
    """
    Get sector scan data for the cross-sectional heatmap.
    Returns all sectors for a given date (default: latest available).
    """
    try:
        scan_date = request.args.get("date")
        if scan_date:
            rows = db.get_sector_scans_on(scan_date)
        else:
            rows = db.get_all_sectors_latest()
        
        if not rows:
            return jsonify({"rows": [], "scan_date": None, "dates": []})
        
        dates = db.get_sector_scan_dates(limit=30)
        actual_date = rows[0].get("scan_date") if rows else None
        
        return jsonify({
            "rows": rows,
            "scan_date": actual_date,
            "dates": dates,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/sector-lookouts/history")
def api_sector_lookouts_history():
    """
    Get historical scan data for a single sector (time-series heatmap).
    """
    sector = (request.args.get("sector") or "").strip()
    if not sector:
        return jsonify({"error": "sector is required"}), 400
    
    try:
        days = int(request.args.get("days", 30))
        end_date = request.args.get("end_date")
        rows = db.get_sector_history(sector, days=days, end_date=end_date)
        return jsonify({"sector": sector, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/sector-lookouts/constituents")
def api_sector_lookouts_constituents():
    """
    Get top stocks for a sector on a given date.
    """
    sector = (request.args.get("sector") or "").strip()
    if not sector:
        return jsonify({"error": "sector is required"}), 400
    
    try:
        scan_date = request.args.get("date")
        top = int(request.args.get("top", 15))
        
        with engine._lock:
            stocks_df = engine.stocks if engine.status == "ready" else None
        
        rows = db.get_sector_constituents(sector, scan_date, stocks_df, top)
        return jsonify({"sector": sector, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/sector-lookouts/save")
def api_sector_lookouts_save():
    """
    Manually trigger saving current sector scans and setups to DB.
    Normally runs automatically at 19:30 IST.
    
    Saves both:
    - Sector classifications (scan_rows)
    - Setups (buys) - coiled stocks in actionable sectors
    """
    snap = engine.snapshot()
    if snap["status"] != "ready":
        return jsonify(snap), 202
    
    with engine._lock:
        scan_rows = engine.scan_rows
        panel = engine.panel
        buys = engine.buys
        verdict_by_sector = engine.verdict_by_sector
        as_of = engine.as_of
    
    if scan_rows is None or scan_rows.empty:
        return jsonify({"error": "No sector data available"}), 400
    
    try:
        n_sectors = db.save_sector_scans(as_of, scan_rows, panel)
        n_setups = db.save_setups(as_of, buys, verdict_by_sector)
        return jsonify({
            "saved_sectors": n_sectors,
            "saved_setups": n_setups,
            "scan_date": as_of,
            "message": f"Saved {n_sectors} sector scans and {n_setups} setups for {as_of}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/coiled-bases/save")
def api_coiled_bases_save():
    """
    Save current coiled bases (stock-level pre-breakout candidates) to DB.
    Uses closing prices — call this after market close.
    
    Coiled bases are stock-level only, independent of sector classifications.
    """
    snap = engine.snapshot()
    if snap["status"] != "ready":
        return jsonify(snap), 202
    
    with engine._lock:
        coil_rows = engine.coil_rows
        as_of = engine.as_of
    
    if coil_rows is None or coil_rows.empty:
        return jsonify({"error": "No coiled bases data available"}), 400
    
    try:
        n = db.save_coiled_bases(as_of, coil_rows)
        return jsonify({
            "saved": n,
            "scan_date": as_of,
            "message": f"Saved {n} coiled bases for {as_of}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/position-trades")
def api_position_trades_get():
    """
    Position Trades for a date. Query params: scan_date (default: latest).

    Reads from the DB rather than the live engine so the list a user sees is
    the one that was actually recorded at scan time.
    """
    try:
        scan_date = request.args.get("scan_date")
        if scan_date:
            rows = db.get_position_trades(scan_date)
        else:
            scan_date, rows = db.latest_position_trades()
        return jsonify({"scan_date": scan_date, "rows": rows or []})
    except Exception as e:
        return jsonify({"error": str(e), "rows": []}), 500


@app.get("/api/position-trades/dates")
def api_position_trades_dates():
    """Dates with saved Position Trades."""
    try:
        return jsonify({"dates": db.get_position_trades_dates()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/coiled-bases/dates")
def api_coiled_bases_dates():
    """Get list of dates with saved coiled bases."""
    try:
        dates = db.get_coiled_bases_dates()
        return jsonify({"dates": dates})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/coiled-bases")
def api_coiled_bases_get():
    """
    Get coiled bases for a specific date.
    Query params: scan_date (defaults to latest available)
    """
    try:
        scan_date = request.args.get("scan_date")
        if not scan_date:
            dates = db.get_coiled_bases_dates(limit=1)
            if not dates:
                return jsonify({"error": "No coiled bases data available"}), 404
            scan_date = dates[0]
        
        rows = db.get_coiled_bases(scan_date)
        return jsonify({
            "scan_date": scan_date,
            "count": len(rows),
            "rows": rows,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/setups/dates")
def api_setups_dates():
    """Get list of dates with saved setups."""
    try:
        dates = db.get_setups_dates()
        return jsonify({"dates": dates})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/setups")
def api_setups_get():
    """
    Get setups for a specific date.
    Query params: scan_date (defaults to latest available)
    
    Setups = coiled stocks in actionable sectors (CROSSING, PULLBACK, etc.)
    """
    try:
        scan_date = request.args.get("scan_date")
        if not scan_date:
            dates = db.get_setups_dates(limit=1)
            if not dates:
                return jsonify({"error": "No setups data available"}), 404
            scan_date = dates[0]
        
        rows = db.get_setups(scan_date)
        return jsonify({
            "scan_date": scan_date,
            "count": len(rows),
            "rows": rows,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Cron-triggered endpoint (public, no auth)
@app.post("/api/cron/sector-lookouts")
def api_cron_sector_lookouts():
    """
    Cron trigger for sector lookouts and setups.
    POST /api/cron/sector-lookouts
    
    Saves both sector classifications and setups (coiled stocks in actionable sectors).
    """
    snap = engine.snapshot()
    if snap["status"] != "ready":
        return jsonify({"status": "not_ready", "message": snap.get("message", "Engine loading")}), 202
    
    with engine._lock:
        scan_rows = engine.scan_rows
        panel = engine.panel
        buys = engine.buys
        verdict_by_sector = engine.verdict_by_sector
        as_of = engine.as_of
    
    if scan_rows is None or scan_rows.empty:
        return jsonify({"error": "No sector data available"}), 400
    
    try:
        n_sectors = db.save_sector_scans(as_of, scan_rows, panel)
        n_setups = db.save_setups(as_of, buys, verdict_by_sector)
        print(f"[cron] saved {n_sectors} sector scans and {n_setups} setups for {as_of}")
        return jsonify({
            "saved_sectors": n_sectors,
            "saved_setups": n_setups,
            "scan_date": str(as_of),
            "message": f"Saved {n_sectors} sector scans and {n_setups} setups for {as_of}",
        })
    except Exception as e:
        print(f"[cron] sector lookout error: {e}")
        return jsonify({"error": str(e)}), 500


@app.post("/api/cron/coiled-bases")
def api_cron_coiled_bases():
    """
    Cron trigger for coiled bases.
    POST /api/cron/coiled-bases
    """
    snap = engine.snapshot()
    if snap["status"] != "ready":
        return jsonify({"status": "not_ready", "message": snap.get("message", "Engine loading")}), 202
    
    with engine._lock:
        coil_rows = engine.coil_rows
        as_of = engine.as_of
    
    if coil_rows is None or coil_rows.empty:
        return jsonify({"error": "No coiled bases data available"}), 400
    
    try:
        n = db.save_coiled_bases(as_of, coil_rows)
        print(f"[cron] saved {n} coiled bases for {as_of}")
        return jsonify({
            "saved": n,
            "scan_date": str(as_of),
            "message": f"Saved {n} coiled bases for {as_of}",
        })
    except Exception as e:
        print(f"[cron] coiled bases error: {e}")
        return jsonify({"error": str(e)}), 500


@app.post("/api/cron/position-trades")
def api_cron_position_trades():
    """
    Cron trigger for Position Trades.
    POST /api/cron/position-trades

    Recomputes from the loaded panel rather than reusing engine.position_rows,
    so a cron hit is correct even if no scan has run since the last refresh.
    """
    snap = engine.snapshot()
    if snap["status"] != "ready":
        return jsonify({"status": "not_ready",
                        "message": snap.get("message", "Engine loading")}), 202

    with engine._lock:
        coil_stocks = engine.coil_stocks
        as_of = engine.as_of

    if coil_stocks is None or getattr(coil_stocks, "empty", True):
        return jsonify({"error": "No stock data available"}), 400

    try:
        rows = posscan.scan(posscan.add_position_features(coil_stocks), as_of=as_of)
        if rows.empty:
            return jsonify({
                "saved": 0,
                "scan_date": str(as_of),
                "message": "No names qualified — needs 270 sessions of history",
            })
        n = db.save_position_trades(as_of, rows)
        print(f"[cron] saved {n} position trades for {as_of}")
        return jsonify({
            "saved": n,
            "scan_date": str(as_of),
            "message": f"Saved {n} position trades for {as_of}",
        })
    except Exception as e:
        print(f"[cron] position trades error: {e}")
        return jsonify({"error": str(e)}), 500


def _known_sessions() -> tuple[set, str | None]:
    """Trading days from the loaded panel, plus as_of."""
    found: set = set()
    as_of = None
    with engine._lock:
        as_of = engine.as_of
        for df in (engine.panel, engine.raw, engine.stocks, engine.coil_stocks):
            if df is None or getattr(df, "empty", True) or "date" not in df.columns:
                continue
            for v in df["date"].dropna().unique():
                d = pd.Timestamp(v).date()
                found.add(d)
    return found, as_of


def _outcomes_ready(scan_date, sessions: set, calendar: dict, now) -> bool:
    """True only after the next *trading* session after the scan has closed."""
    scan_d = db._as_date(scan_date)
    if not scan_d:
        return False
    today = db._as_date(calendar.get("today")) or db.now_ist(now).date()
    nxt = db.next_trading_date(scan_d, sessions or None, today)
    if nxt is None:
        return False
    if today < nxt:
        return False
    if today > nxt:
        return True
    if calendar.get("today_holiday"):
        return False
    return (not market_is_open(now)) and now.time() >= MARKET_CLOSE


@app.get("/api/track-record")
def api_track_record():
    """One day's For Tom list (default: last session before today)."""
    try:
        kind = request.args.get("kind")
        dates = db.get_prediction_dates()
        sessions, as_of = _known_sessions()
        for d in dates:
            parsed = db._as_date(d)
            if parsed:
                sessions.add(parsed)
        now = db.now_ist()
        calendar = db.session_calendar(now, sessions, as_of)

        requested = (request.args.get("scan_date") or "").strip()[:10]
        scan_date = requested if requested in dates else None
        if not scan_date:
            scan_date = db.verify_scan_date(dates) or (dates[0] if dates else None)

        preds = db.get_predictions_with_outcomes(kind=kind, scan_date=scan_date)
        stats = db.get_track_record_stats(kind=kind, scan_date=scan_date, preds=preds)

        with engine._lock:
            coil = engine.coil_stocks if engine.status == "ready" else None
        base = tomscan.momentum_base_rates(coil, scan_date)
        stats["base_n"] = base.get("n")
        stats["base_hit_rate"] = base.get("hit_rate")
        stats["base_avg_gain"] = base.get("avg_gain")
        stats["base_scope"] = base.get("scope")
        pooled = base.get("pooled") or {}
        stats["base_pooled_n"] = pooled.get("n")
        stats["base_pooled_hit_rate"] = pooled.get("hit_rate")
        stats["base_pooled_avg_gain"] = pooled.get("avg_gain")
        if stats.get("hit_rate") is not None and stats.get("base_hit_rate") is not None:
            stats["edge_hit"] = stats["hit_rate"] - stats["base_hit_rate"]
        if stats.get("avg_gain") is not None and stats.get("base_avg_gain") is not None:
            stats["edge_gain"] = stats["avg_gain"] - stats["base_avg_gain"]

        next_session_done = _outcomes_ready(scan_date, sessions, calendar, now)
        nxt = db.next_trading_date(
            db._as_date(scan_date), sessions or None,
            db._as_date(calendar["today"]),
        ) if scan_date else None

        return jsonify({
            "dates": dates,
            "scan_date": scan_date,
            "stats": stats,
            "predictions": preds,
            "outcomes_ready": next_session_done,
            "calendar": {
                **calendar,
                "verify_session": nxt.isoformat() if nxt else None,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/verify-outcomes")
def api_verify_outcomes():
    """
    Yesterday's For Tom list vs today's high.

    Fetches live/closing quotes for those symbols only (one Groww batch),
    then backfills older days from the EOD panel if it is loaded.
    """
    try:
        n = _verify_yesterday()
        return jsonify({
            "verified": n,
            "message": (
                f"Verified {n} picks — today's high vs yesterday's scan price"
                if n else "Nothing to verify yet"
            ),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/subscription")
def api_subscription():
    """Get current user's subscription status."""
    user = request.environ.get("auth_user") or {}
    email = user.get("email") or ""
    if not email:
        return jsonify({"is_premium": False, "plan": None})
    
    try:
        from subscription import check_subscription
        return jsonify(check_subscription(email))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/subscription/checkout")
def api_subscription_checkout():
    """Create a Dodo checkout session for subscription."""
    user = request.environ.get("auth_user") or {}
    email = user.get("email") or ""
    name = user.get("name") or user.get("full_name") or ""
    if not email:
        return jsonify({"error": "Sign in required"}), 401
    
    body = request.get_json(silent=True) or {}
    plan = body.get("plan", "monthly")
    
    try:
        from subscription import create_checkout_url
        url = create_checkout_url(email, plan, customer_name=name)
        if url:
            return jsonify({"url": url})
        return jsonify({"error": "Could not create checkout session"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/subscription/webhook")
def api_subscription_webhook():
    """Dodo webhook for subscription events. Updates Supabase."""
    from subscription import verify_webhook_signature, handle_webhook
    
    # Verify signature
    signature = request.headers.get("X-Dodo-Signature", "") or request.headers.get("Webhook-Signature", "")
    if signature and not verify_webhook_signature(request.data, signature):
        return jsonify({"error": "Invalid signature"}), 401
    
    body = request.get_json(silent=True) or {}
    event_type = body.get("type") or body.get("event_type") or ""
    data = body.get("data", body)  # Data might be nested or at root
    
    if event_type:
        handle_webhook(event_type, data)
    
    return jsonify({"received": True})


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/")
def index():
    if (UI_DIST / "index.html").exists():
        return send_from_directory(UI_DIST, "index.html")
    return (
        "API is up. Start the UI with <code>cd ui && npm run dev</code> "
        "or run <code>npm run build</code> in ui/ to serve it from Flask.",
        200,
        {"Content-Type": "text/html"},
    )


@app.get("/<path:path>")
def ui_assets(path: str):
    if path.startswith("api/"):
        return jsonify({"error": "not found"}), 404
    target = UI_DIST / path
    if target.exists() and target.is_file():
        return send_from_directory(UI_DIST, path)
    if (UI_DIST / "index.html").exists():
        return send_from_directory(UI_DIST, "index.html")
    return jsonify({"error": "not found"}), 404


def _job_state(key: str):
    try:
        import jobs
        return jobs.state.get(key)
    except Exception:
        return None


def _boot():
    try:
        engine.load(DEFAULT_DAYS, date.today())
    except Exception as exc:
        engine._set(status="error", error=str(exc), message="Startup load failed.")
    try:
        import jobs
        jobs.start(engine, DEFAULT_DAYS)
    except Exception as exc:
        print(f"[jobs] failed to start: {exc}")


if __name__ == "__main__":
    threading.Thread(target=_boot, daemon=True).start()
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)
