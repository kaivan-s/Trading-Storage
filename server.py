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
import json
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
import episodes as eps
import fetch
import flagslog
import panel as pnl
import scan as sc
import stocks as stk
import position as posscan

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
#
# REGIME_KLASSES drives the market-regime flag and the Telegram headline:
# only fully verified sectors. SETUP_KLASSES adds CROSSING_UNVERIFIED so
# the Setups tab shows "Emerging" watchlist names when no verified sector
# qualifies, without pretending the regime is bullish.
WATCH_KLASSES = ("CROSSING", "PULLBACK", "BASE")
REGIME_KLASSES = ("CROSSING", "PULLBACK")
SETUP_KLASSES = ("CROSSING", "PULLBACK", "CROSSING_UNVERIFIED")

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


def _rank_at_rest(coil_rows: pd.DataFrame, coil_stocks: pd.DataFrame):
    """
    `(pool carrying mom12_1 and rest_rank, the cut-to-20 list)`.

    Ranked once and stamped back onto the pool, because both columns are
    stored and the UI renders them as given. Doing it the other way round --
    letting leaders_at_rest keep the reading on its own 20 rows -- leaves
    mom12_1 null on every row of the saved pool, which is what made Leaders
    at rest carry a rank with nothing behind it.

    Annotated over the whole pool rather than the top 20, so the momentum
    shown against a coil is present whether or not it made the cut.
    """
    if coil_rows is None or coil_rows.empty:
        return coil_rows, pd.DataFrame()

    out = posscan.annotate_momentum(coil_rows, coil_stocks)
    rest = posscan.leaders_at_rest(coil_rows, coil_stocks)
    order = {s: i + 1 for i, s in enumerate(rest["symbol"])} if not rest.empty else {}
    out["rest_rank"] = out["symbol"].map(order)
    return out, rest


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
        self.episodes = pd.DataFrame()
        self.near_miss = pd.DataFrame()
        self.buys = pd.DataFrame()
        self.breakouts = pd.DataFrame()
        self.tom = pd.DataFrame()
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
                    self.scan_rows["klass"].isin(REGIME_KLASSES).sum()
                )
            # Market regime: bullish if > 40% of sectors are in uptrend.
            # Uses REGIME_KLASSES (CROSSING + PULLBACK only), not the wider
            # SETUP_KLASSES, so unverified expansions don't inflate the count.
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
            coil_all["shape_confirmed"] = pd.Series(dtype=bool)
            coil_all["sector_klass"] = pd.Series(dtype=str)
            buys = coil_all
        else:
            coil_all = coil_all.copy()
            # `shape_confirmed` = the sector's shape checks all passed, which is
            # the funnel buytest.py measures. Kept as a column, not a filter, so
            # the UI can rank by confidence without hiding the wider set.
            coil_all["shape_confirmed"] = coil_all["sector"].isin(ready)
            coil_all["sector_klass"] = coil_all["sector"].map(sector_klass_map).fillna("")
            setup_sectors = set(
                scan_rows[scan_rows["klass"].isin(SETUP_KLASSES)]["sector"]
            )
            buys = coil_all[coil_all["sector"].isin(setup_sectors)].copy()
        # No cap: the seven gates already did the filtering, and ranking by
        # `coil` measured no relationship with forward returns, so cutting
        # the list at 40 by that score was discarding names arbitrarily.
        # The shown list: same pool, ordered by 12-1 momentum and cut to 20.
        # See eval_listsize.py for why 20 rather than the full 63 or a
        # single-digit shortlist.
        coil_rows, rest_rows = _rank_at_rest(coil_all, coil_stocks)

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

        # Episodes track the names that were actually published, so they are
        # seeded from the saved coil lists rather than from a replay of the
        # raw gates -- the gates would open an episode for every base they
        # would have found over the whole panel, which is history nobody saw.
        #
        # Today's list is passed in from memory because it is not saved until
        # _save_post_market runs, which is after this. The state is a full
        # replay rather than a step forward from yesterday, so age, gap and
        # the bridged streak are all derived and a missed or repeated run
        # repairs itself instead of drifting.
        self._set(message="Tracking base episodes…")
        episode_rows = pd.DataFrame()
        try:
            published = {}
            try:
                published = db.get_published_coils()
            except Exception as exc:
                print(f"published coil history unavailable: {exc}")
            if not coil_rows.empty:
                published[as_of_s] = list(coil_rows["symbol"])
            episode_rows = eps.build(coil_stocks, published=published)
            # The shown lists carry the bridged age instead of `coil_days`,
            # which resets on a single missed session.
            coil_rows = eps.tag(coil_rows, episode_rows, as_of)
            rest_rows = eps.tag(rest_rows, episode_rows, as_of)
            buys = eps.tag(buys, episode_rows, as_of)
        except Exception as exc:
            print(f"episode tracking failed: {exc}")

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
            self.episodes = episode_rows
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

        # Persist today's setups outside the lock (file IO). Idempotent per
        # as_of, so a repeated Refresh does not duplicate rows.
        try:
            flagslog.append_day(as_of_s, buys, scan_rows, verdict_by_sector)
        except Exception as exc:  # logging must never break a load
            print(f"flags log append failed: {exc}")

    # Restoring the saved For Tom list used to happen here. It read
    # tom_predictions, which no longer exists in this project, to populate
    # state no screen reads -- Expected Movers was removed once its score
    # measured no return edge. So it logged a failure on every load and
    # bought nothing. The live scan path still fills self.tom for the
    # intraday endpoints.

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
                members = self.stocks[self.stocks["sector"].str.lower() == name.lower()]
                if not members.empty:
                    n = int(members["symbol"].nunique())
                    return {"found": False, "near": [], "reason":
                            f"{name} has {n} liquid stocks. Sectors need at least 8 "
                            "before breadth and width mean anything, so it is not scanned."}
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


@app.get("/api/health")
def api_health():
    """Quick data-freshness check — public, no auth."""
    snap = engine.snapshot()
    try:
        lists = db.get_lists()
        db_as_of = lists.get("as_of")
        db_dates = lists.get("dates", {})
        db_counts = {
            "sectors": len(lists.get("scan") or []),
            "setups": len(lists.get("buys") or []),
            "coils": len(lists.get("coil") or []),
        }
        stale = lists.get("stale", [])
    except Exception as exc:
        db_as_of = None
        db_dates = {}
        db_counts = {}
        stale = [f"db error: {exc}"]

    return jsonify({
        "engine_status": snap.get("status"),
        "engine_as_of": snap.get("as_of"),
        "engine_loaded_at": snap.get("loaded_at"),
        "engine_message": snap.get("message"),
        "db_as_of": db_as_of,
        "db_dates": db_dates,
        "db_counts": db_counts,
        "stale": stale,
        "n_buys": snap.get("n_buys", 0),
        "n_coil": snap.get("n_coil", 0),
    })


@app.get("/api/dashboard")
def api_dashboard():
    """
    The lists, read from Supabase rather than from the in-memory panel.

    The post-market cron is the only writer, so this is a SELECT of what it
    last stored. Nothing here depends on the engine having loaded, which is
    what lets a cold process serve the current lists immediately instead of
    showing an empty app until it has re-fetched a year of bhavcopies.

    `snapshot` is still merged in for the status fields the header reads, but
    the rows themselves come from the database.
    """
    snap = engine.snapshot()
    try:
        lists = db.get_lists()
    except Exception as exc:
        print(f"dashboard read failed: {exc}")
        return jsonify({**snap, "error": f"Could not read saved lists: {exc}"}), 503

    if not (lists.get("scan") or lists.get("coil") or lists.get("buys")):
        return jsonify({
            **snap,
            "status": "empty",
            "message": "No saved scan yet. The post-market job writes one "
                       "after the close.",
        }), 202

    return jsonify({
        **snap,
        "status": "ready",
        "as_of": lists["as_of"] or snap.get("as_of"),
        "scan": lists["scan"],
        "coil": lists["coil"],
        "rest": lists["rest"],
        "buys": lists["buys"],
        "n_coil": len(lists["coil"]),
        "n_buys": len(lists["buys"]),
        "stale": lists.get("stale", []),
        "dates": lists.get("dates", {}),
        "session": lists.get("session"),
    })


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


FREE_STOCK_LOOKUPS_PER_DAY = 1


def _stock_lookup_gate(symbol: str) -> tuple[dict | None, dict | None]:
    """(lookups_info, denial). Free users get N distinct stocks per IST day;
    reopening one already counted today is free. Pro and local dev are
    unlimited. If the lookups table can't be read the gate stays open, so a
    Supabase hiccup never locks paying-tier pages for everyone."""
    user = request.environ.get("auth_user") or {}
    email = (user.get("email") or "").strip().lower()
    if not email:
        return None, None
    try:
        from subscription import check_subscription
        if check_subscription(email).get("is_premium"):
            return {"unlimited": True}, None
    except Exception as exc:
        print(f"stock gate subscription check: {exc}")

    day = db.now_ist().date().isoformat()
    used = db.get_stock_lookups(email, day)
    if used is None:
        return None, None
    limit = FREE_STOCK_LOOKUPS_PER_DAY
    if symbol in used:
        return {"unlimited": False, "limit": limit, "used": used}, None
    if len(used) >= limit:
        return None, {
            "error": "Daily free stock lookup used",
            "limit_reached": True,
            "limit": limit,
            "used": used,
            "symbol": symbol,
        }
    db.add_stock_lookup(email, day, symbol)
    return {"unlimited": False, "limit": limit, "used": used + [symbol]}, None


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
    out = engine.stock(q, entry=entry)
    if out.get("found") and out.get("symbol"):
        info, denial = _stock_lookup_gate(str(out["symbol"]).upper())
        if denial:
            return jsonify(denial), 403
        if info:
            out["lookups"] = info
    return jsonify(out)


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


@app.get("/api/episodes")
def api_episodes():
    """
    Base episodes: what happened to each base since it appeared.

    Read from Supabase like every other list, and scoped to open bases plus
    the last few sessions of resolutions. The table itself keeps every
    episode ever tracked, but almost all of them are closed and old: serving
    the full retention window put several hundred rows on screen, none of
    them news. The digest is computed from the returned rows rather than
    stored, since it is just a count of what changed on the latest date.
    """
    try:
        rows = db.get_open_episodes()
        if not rows:
            return jsonify({
                "as_of": None, "digest": None, "count": 0, "rows": [],
                "message": "No episodes saved yet. The post-market job writes "
                           "them after the close.",
            })

        df = pd.DataFrame(rows)
        as_of = max(str(r.get("state_since") or "") for r in rows) or None
        return jsonify({
            "as_of": as_of,
            "digest": eps.digest(df, as_of) if as_of else None,
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


def _save_post_market(force: bool = False) -> dict:
    """
    Persist everything derived from the session now in memory.

    Every save is keyed on `engine.as_of` -- the newest session actually in
    the panel -- and never on today's date. On a holiday the panel does not
    advance, so the run finds nothing new and writes nothing, rather than
    publishing the previous session's numbers under a date the market never
    traded. `force` re-writes the current session anyway, for backfilling a
    table after a failure.

    One failure does not abort the rest: a sector save that fails should not
    cost us the coil list.
    """
    with engine._lock:
        as_of = engine.as_of
        scan_rows = engine.scan_rows
        panel = engine.panel
        buys = engine.buys
        coil_rows = engine.coil_rows
        episode_rows = engine.episodes
        verdict = engine.verdict_by_sector

    out: dict = {"scan_date": str(as_of)}

    # Nothing new to publish. Reached on a holiday, on a weekend, and on any
    # evening the bhavcopy has not landed yet -- in all three the panel still
    # ends at the last real session, which is already stored.
    saved = None
    try:
        saved = db.latest_saved_session()
    except Exception as exc:
        print(f"[cron] could not read the last saved session: {exc}")
    if not force and saved and as_of and str(as_of) <= saved:
        out.update(skipped="no new session", latest_saved=saved)
        print(f"[cron] nothing to do: panel ends {as_of}, "
              f"already saved through {saved}")
        return out

    def attempt(label, fn):
        try:
            out[label] = fn()
        except Exception as exc:
            out[label] = f"error: {exc}"
            print(f"[cron] {label} save failed: {exc}")

    if scan_rows is not None and not scan_rows.empty:
        attempt("sectors", lambda: db.save_sector_scans(as_of, scan_rows, panel))
        n_buys = len(buys) if buys is not None and not buys.empty else 0
        setup_klasses = list(scan_rows[scan_rows["klass"].isin(SETUP_KLASSES)]["sector"]) if not scan_rows.empty else []
        print(f"[cron] setups: {n_buys} buys in {len(setup_klasses)} setup sectors ({setup_klasses[:5]})")
        attempt("setups", lambda: db.save_setups(as_of, buys, verdict))
    else:
        out["sectors"] = out["setups"] = 0

    # The coil pool carries mom12_1, so Leaders at rest is the top 20 of this
    # table by that column rather than a second table holding the same rows.
    if coil_rows is not None and not coil_rows.empty:
        attempt("coiled_bases", lambda: db.save_coiled_bases(as_of, coil_rows))
    else:
        out["coiled_bases"] = 0

    # Upsert, not replace: episodes are spans that outlive the panel window,
    # so the rows that have aged out of the replay must survive untouched.
    if episode_rows is not None and not episode_rows.empty:
        attempt("episodes", lambda: db.save_episodes(episode_rows))
    else:
        out["episodes"] = 0

    print(f"[cron] post-market saved for {as_of}: {out}")
    return out


@app.post("/api/cron/post-market")
def api_cron_post_market():
    """
    The one scheduled call. Run it once after the bhavcopy lands (~19:30 IST).
    POST /api/cron/post-market

    Loads the latest session and then saves the sectors, the setups, the coil
    pool and the base episodes derived from it. Returns 202 straight away and
    finishes in a background thread, because a full load runs for minutes and
    cron services time out long before that -- so a 202 here means "accepted
    and running", not "finished". Poll /api/status to watch it.

    Safe to call every day including holidays. The save is keyed on the newest
    session in the panel, so a day the market was shut writes nothing instead
    of republishing yesterday's numbers under today's date. Pass `?force=1`
    to re-write the current session, for filling a table in after a failure.

    This replaces jobs.py. Loading is the part that matters: the previous
    per-list cron endpoints only saved whatever was already in memory, so
    without an internal scheduler they would have re-saved the same stale
    snapshot every night.
    """
    if engine._busy.locked():
        return jsonify({**engine.snapshot(),
                        "message": "Already running."}), 409

    force = str(request.args.get("force") or "").lower() in ("1", "true", "yes")

    def work():
        try:
            ok = engine.load(DEFAULT_DAYS, date.today())
        except Exception as exc:
            engine._set(status="error", error=str(exc), message="Cron load failed.")
            print(f"[cron] load failed: {exc}")
            return
        if not ok:
            print("[cron] load returned no data — nothing saved")
            return
        _save_post_market(force=force)
        try:
            import carry
            if engine.as_of:
                n = carry.ensure_session(pd.Timestamp(engine.as_of).date())
                if n:
                    print(f"[cron] carry: no live snapshot, rebuilt {n} rows from bhavcopy")
            filled, pending = carry.score()
            print(f"[cron] carry scored {filled}, {pending} pending")
        except Exception as exc:
            print(f"[cron] carry score failed: {exc}")
        # Pre-compute bot cache so Telegram replies are instant
        try:
            import telegram_interactive as tgi
            tgi.precompute_bot_cache(engine)
        except Exception as exc:
            print(f"[cron] bot cache failed: {exc}")

    threading.Thread(target=work, daemon=True).start()
    return jsonify({
        "status": "started",
        "message": "Loading the latest session, then saving sectors, setups, "
                   "the coil pool and base episodes. Poll /api/status for "
                   "progress.",
    }), 202


_carry_busy = threading.Lock()


@app.post("/api/cron/carry-snap")
def api_cron_carry_snap():
    """
    Upper-circuit carry snapshot. Schedule at 15:22 IST, weekdays.
    POST /api/cron/carry-snap

    Must run while the market is open: Groww rolls circuit limits to the next
    session after the close, so a later call records nothing useful. Scoring
    happens inside /api/cron/post-market. Returns 202 and runs in a thread.
    """
    if not _carry_busy.acquire(blocking=False):
        return jsonify({"status": "busy", "message": "Snapshot already running."}), 409

    def work():
        try:
            import carry
            rows = carry.snapshot()
            n = carry.append(rows)
            at = int(rows["at_circuit"].astype(bool).sum()) if n else 0
            print(f"[cron] carry snap: {at} at circuit, {n} logged")
        except Exception as exc:
            print(f"[cron] carry snap failed: {exc}")
        finally:
            _carry_busy.release()

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"status": "started",
                    "message": "Reading at-circuit names and order books."}), 202


@app.get("/api/carry")
@app.get("/api/carry/report")
def api_carry():
    """
    The Circuit carry screen: the latest session's list, the track record
    summary, a per-session history and the scored rows behind it.
    """
    import carry
    try:
        return jsonify(carry.payload())
    except Exception as exc:
        return jsonify({"error": f"Could not read carry log: {exc}"}), 500


_intraday_busy = threading.Lock()


@app.post("/api/cron/carry-scan")
def api_cron_carry_scan():
    """
    Intraday circuit scanner. Schedule every 30 min during market hours.
    POST /api/cron/carry-scan
    
    Scans for stocks approaching their upper circuit while still buyable.
    Call at 10:00, 10:30, 11:00, ... 15:00 IST on trading days.
    Returns 202 and runs in a thread.
    """
    if not _intraday_busy.acquire(blocking=False):
        return jsonify({"status": "busy", "message": "Scan already running."}), 409

    def work():
        try:
            import carry
            rows = carry.intraday_scan()
            n = carry.save_intraday_scan(rows)
            summary = rows["status"].value_counts().to_dict() if not rows.empty else {}
            print(f"[cron] intraday scan: {n} rows - {summary}")
            # Circuit flash to channels — disabled (bot-only for now)
            # send_paid is a no-op, but skip the work entirely
            print(f"[cron] circuit flash skipped (channels disabled)")
        except Exception as exc:
            print(f"[cron] intraday scan failed: {exc}")
        finally:
            _intraday_busy.release()

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"status": "started",
                    "message": "Scanning for stocks approaching circuit."}), 202


@app.get("/api/carry/intraday")
def api_carry_intraday():
    """
    Intraday scan results for today. Shows stocks approaching or at circuit
    with their progression through the day.
    
    Query params:
        as_of: date in YYYY-MM-DD format (default: today)
    """
    import carry
    try:
        as_of = request.args.get("as_of")
        return jsonify(carry.intraday_payload(as_of))
    except Exception as exc:
        return jsonify({"error": f"Could not read intraday scan: {exc}"}), 500


# --------------------------------------------------------------------------
# Broader market scanners
# --------------------------------------------------------------------------

_scanner_busy = threading.Lock()
_scanner_cache: dict = {}


@app.post("/api/cron/scanner-scan")
def api_cron_scanner_scan():
    """
    Run all intraday scanners. Schedule alongside carry-scan (every 30 min).
    POST /api/cron/scanner-scan
    """
    if not _scanner_busy.acquire(blocking=False):
        return jsonify({"status": "busy"}), 409

    def work():
        global _scanner_cache
        try:
            import scanners
            data = scanners.intraday_all()
            _scanner_cache["intraday"] = data
            print(f"[cron] scanners: {len(data.get('big_movers', []))} movers, "
                  f"{len(data.get('unusual_volume', []))} unusual vol, "
                  f"{len(data.get('sector_pulse', []))} sectors")
            # No Telegram here — midday pulse is sent once at 12:30,
            # not on every 30-min scan. See /api/cron/telegram-midday.
        except Exception as exc:
            print(f"[cron] scanners failed: {exc}")
        finally:
            _scanner_busy.release()

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"status": "started"}), 202


@app.post("/api/cron/scanner-eod")
def api_cron_scanner_eod():
    """
    Run EOD scanners (momentum streaks, 52w breakouts). Schedule after market close.
    POST /api/cron/scanner-eod
    """
    def work():
        global _scanner_cache
        try:
            import scanners
            data = scanners.eod_all()
            _scanner_cache["eod"] = data
            print(f"[cron] EOD scanners: {len(data.get('breakouts_52w', []))} breakouts, "
                  f"{len(data.get('momentum_streaks', []))} streaks")
            # Channel sending disabled — bot-only for now
            print(f"[cron] scanner EOD channel send skipped (channels disabled)")
        except Exception as exc:
            print(f"[cron] EOD scanners failed: {exc}")

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"status": "started"}), 202


# --- Channel cron endpoints disabled — bot-only for now ---
# These endpoints still exist so external cron jobs don't 404,
# but they no-op since send_both / send_paid are disabled.

@app.post("/api/cron/telegram-eod")
def api_cron_telegram_eod():
    """EOD wrap — disabled (channels off)."""
    return jsonify({"status": "skipped", "reason": "channels disabled"}), 202

@app.post("/api/cron/telegram-morning")
def api_cron_telegram_morning():
    """Morning scorecard — disabled (channels off)."""
    return jsonify({"status": "skipped", "reason": "channels disabled"}), 202

@app.post("/api/cron/telegram-midday")
def api_cron_telegram_midday():
    """Midday pulse — disabled (channels off)."""
    return jsonify({"status": "skipped", "reason": "channels disabled"}), 202

@app.post("/api/cron/telegram-weekly")
def api_cron_telegram_weekly():
    """Weekly digest — disabled (channels off)."""
    return jsonify({"status": "skipped", "reason": "channels disabled"}), 202


# --------------------------------------------------------------------------
# Fundamentals
# --------------------------------------------------------------------------

@app.get("/api/fundamentals")
def api_fundamentals():
    """
    Fundamental data for one symbol. Cache-first; fetches from NSE on miss.

    GET /api/fundamentals?symbol=RELIANCE
    """
    import fundamentals as fun
    symbol = (request.args.get("symbol") or request.args.get("q") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    try:
        data = fun.get_fundamentals_or_fetch(symbol)
        if not data:
            return jsonify({"found": False, "symbol": symbol}), 404
        return jsonify({"found": True, **data})
    except Exception as e:
        return jsonify({"error": str(e), "symbol": symbol}), 500


@app.get("/api/fundamentals/screen")
def api_fundamentals_screen():
    """
    All cached fundamentals, with optional filters. For the Fundamentals view.

    GET /api/fundamentals/screen?min_mcap=500&max_pe=40&sort=pe_ttm
    """
    import fundamentals as fun
    try:
        rows = fun.get_screen_data(
            min_mcap_cr=_float_or_none(request.args.get("min_mcap")),
            max_pe=_float_or_none(request.args.get("max_pe")),
            min_growth=_float_or_none(request.args.get("min_growth")),
            sector=request.args.get("sector"),
            sort_by=request.args.get("sort", "market_cap_cr"),
            sort_desc=request.args.get("order", "desc").lower() != "asc",
            limit=min(int(request.args.get("limit", 500)), 2000),
        )
        return jsonify({"count": len(rows), "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _float_or_none(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_fundamentals_busy = threading.Lock()


@app.post("/api/cron/fundamentals-refresh")
def api_cron_fundamentals_refresh():
    """
    Bulk-refresh fundamentals from NSE. Run quarterly after results season.

    POST /api/cron/fundamentals-refresh

    Uses the symbol list from the loaded panel. Takes ~50 min for 2,000
    stocks (3 NSE calls per symbol, 0.5s pause). Skips symbols already
    refreshed within the staleness window. Returns 202 immediately.
    """
    import fundamentals as fun

    if not _fundamentals_busy.acquire(blocking=False):
        return jsonify({"status": "busy",
                        "message": "Refresh already running."}), 409

    # Get the symbol list from the engine or from the equity list
    symbols = []
    prices = {}
    with engine._lock:
        if engine.coil_stocks is not None and not engine.coil_stocks.empty:
            last = engine.coil_stocks[
                engine.coil_stocks["date"] == engine.coil_stocks["date"].max()
            ]
            symbols = list(last["symbol"].unique())
            if "adj" in last.columns:
                prices = dict(zip(last["symbol"], last["adj"]))
    if not symbols:
        try:
            eq = fetch.fetch_equity_list()
            symbols = list(eq["symbol"].unique())
        except Exception:
            pass
    if not symbols:
        _fundamentals_busy.release()
        return jsonify({"error": "No symbols available"}), 400

    def work():
        try:
            n = fun.refresh_bulk(symbols, prices=prices)
            print(f"[cron] fundamentals: refreshed {n} symbols")
        except Exception as exc:
            print(f"[cron] fundamentals refresh failed: {exc}")
        finally:
            _fundamentals_busy.release()

    threading.Thread(target=work, daemon=True).start()
    return jsonify({
        "status": "started",
        "symbols": len(symbols),
        "message": f"Refreshing fundamentals for {len(symbols)} symbols. "
                   "This takes ~50 min. Poll /api/fundamentals/screen to "
                   "see progress.",
    }), 202


# --------------------------------------------------------------------------
# Chart data (OHLCV via yfinance)
# --------------------------------------------------------------------------

@app.get("/api/chart-data")
def api_chart_data():
    """
    Historical OHLCV for a symbol. Used by the in-app candlestick chart.
    Query params: symbol (required), period (default 1y: 6mo|1y|2y|5y)
    """
    symbol = (request.args.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    period = request.args.get("period", "1y")
    if period not in ("6mo", "1y", "2y", "5y"):
        period = "1y"

    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period=period)

        if hist.empty:
            return jsonify({"symbol": symbol, "data": [], "volume": []})

        import math

        candles = []
        volumes = []
        for idx, row in hist.iterrows():
            o, h, l, c, v = row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]
            # Skip rows with NaN (Yahoo Finance gaps)
            if any(math.isnan(x) for x in (o, h, l, c)):
                continue
            t = idx.strftime("%Y-%m-%d")
            o, c = round(o, 2), round(c, 2)
            candles.append({
                "time": t,
                "open": o,
                "high": round(h, 2),
                "low": round(l, 2),
                "close": c,
            })
            volumes.append({
                "time": t,
                "value": int(v) if not math.isnan(v) else 0,
                "color": "rgba(125,186,150,0.3)" if c >= o else "rgba(229,115,115,0.3)",
            })

        return jsonify({"symbol": symbol, "data": candles, "volume": volumes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------

@app.get("/api/watchlist")
def api_watchlist_get():
    """
    Get user's watchlist, enriched with fundamentals and latest data.
    """
    user = request.environ.get("auth_user") or {}
    email = user.get("email") or ""
    if not email:
        return jsonify({"error": "Sign in required"}), 401

    rows = db.get_watchlist(email)
    if not rows:
        return jsonify({"count": 0, "rows": []})

    # Enrich with fundamentals
    symbols = [r["symbol"] for r in rows]
    fund_lookup = {}
    try:
        client = db.get_client()
        resp = (client.table("company_fundamentals")
                .select("symbol,name,industry,earnings_trend,market_cap_cr,pe_ttm,"
                        "promoter_pct,latest_pat_growth_yoy,latest_opm_pct")
                .in_("symbol", symbols)
                .execute())
        fund_lookup = {r["symbol"]: r for r in (resp.data or [])}
    except Exception:
        pass

    # Enrich with latest coil/setup data if available
    coil_lookup = {}
    try:
        client = db.get_client()
        coil_date = db._latest_date("coiled_bases")
        if coil_date:
            resp = (client.table("coiled_bases")
                    .select("symbol,adj,coil,pos_hi,to_trigger,rsi,vol_ratio,cmf")
                    .eq("scan_date", coil_date)
                    .in_("symbol", symbols)
                    .execute())
            coil_lookup = {r["symbol"]: r for r in (resp.data or [])}
    except Exception:
        pass

    enriched = []
    for row in rows:
        sym = row["symbol"]
        f = fund_lookup.get(sym, {})
        c = coil_lookup.get(sym, {})
        enriched.append({
            "symbol": sym,
            "notes": row.get("notes"),
            "added_at": row.get("added_at"),
            "name": f.get("name"),
            "industry": f.get("industry"),
            "earnings_trend": f.get("earnings_trend"),
            "market_cap_cr": f.get("market_cap_cr"),
            "pe_ttm": f.get("pe_ttm"),
            "promoter_pct": f.get("promoter_pct"),
            "latest_pat_growth_yoy": f.get("latest_pat_growth_yoy"),
            "latest_opm_pct": f.get("latest_opm_pct"),
            "adj": c.get("adj"),
            "coil": c.get("coil"),
            "pos_hi": c.get("pos_hi"),
            "to_trigger": c.get("to_trigger"),
            "rsi": c.get("rsi"),
            "vol_ratio": c.get("vol_ratio"),
            "cmf": c.get("cmf"),
            "is_coiled": bool(c),
        })

    return jsonify({"count": len(enriched), "rows": enriched})


@app.post("/api/watchlist")
def api_watchlist_add():
    """Add a symbol to watchlist. Body: {symbol, notes?}"""
    user = request.environ.get("auth_user") or {}
    email = user.get("email") or ""
    if not email:
        return jsonify({"error": "Sign in required"}), 401

    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    row = db.add_to_watchlist(email, symbol, body.get("notes"))
    if row:
        return jsonify({"ok": True, "symbol": symbol})
    return jsonify({"error": "Failed to add"}), 500


@app.delete("/api/watchlist")
def api_watchlist_remove():
    """Remove a symbol from watchlist. Body: {symbol}"""
    user = request.environ.get("auth_user") or {}
    email = user.get("email") or ""
    if not email:
        return jsonify({"error": "Sign in required"}), 401

    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    ok = db.remove_from_watchlist(email, symbol)
    return jsonify({"ok": ok, "symbol": symbol})


@app.get("/api/results-calendar")
def api_results_calendar():
    """
    Upcoming quarterly results dates, enriched with fundamentals.

    GET /api/results-calendar?days=45
    """
    import fundamentals as fun
    try:
        days = int(request.args.get("days", 45))
        rows = fun.get_results_calendar(days_ahead=days)
        return jsonify({"count": len(rows), "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_calendar_busy = threading.Lock()


@app.post("/api/cron/results-calendar")
def api_cron_results_calendar():
    """
    Refresh results calendar from Yahoo Finance.

    POST /api/cron/results-calendar

    Fetches upcoming earnings dates for all symbols. Takes ~15 min
    for 1,700 stocks. Returns 202 immediately.
    """
    import fundamentals as fun

    if not _calendar_busy.acquire(blocking=False):
        return jsonify({"status": "busy",
                        "message": "Calendar refresh already running."}), 409

    symbols = []
    with engine._lock:
        if engine.coil_stocks is not None and not engine.coil_stocks.empty:
            last = engine.coil_stocks[
                engine.coil_stocks["date"] == engine.coil_stocks["date"].max()
            ]
            symbols = list(last["symbol"].unique())
    if not symbols:
        try:
            eq = fetch.fetch_equity_list()
            symbols = list(eq["symbol"].unique())
        except Exception:
            pass
    if not symbols:
        _calendar_busy.release()
        return jsonify({"error": "No symbols available"}), 400

    def work():
        try:
            n = fun.refresh_results_calendar(symbols)
            print(f"[cron] results calendar: {n} entries saved")
        except Exception as exc:
            print(f"[cron] results calendar failed: {exc}")
        finally:
            _calendar_busy.release()

    threading.Thread(target=work, daemon=True).start()
    return jsonify({
        "status": "started",
        "symbols": len(symbols),
        "message": f"Fetching results calendar for {len(symbols)} symbols.",
    }), 202


# --------------------------------------------------------------------------
# Interactive Telegram bot (webhook)
# --------------------------------------------------------------------------

@app.post("/api/telegram/webhook")
def api_telegram_webhook():
    """
    Telegram sends message updates here. The bot replies with stock reports.
    Must return 200 immediately — Telegram retries if the response is slow,
    causing duplicate messages. All processing happens in a background thread.
    """
    import telegram_interactive as tgi
    update = request.get_json(silent=True) or {}

    def work():
        try:
            tgi.process_update(update, engine)
        except Exception as exc:
            print(f"[telegram] webhook error: {exc}")

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/telegram/setup-webhook")
def api_telegram_setup_webhook():
    """Register our webhook URL with Telegram."""
    import telegram_interactive as tgi
    body = request.get_json(silent=True) or {}
    base = body.get("url", "").rstrip("/")
    if not base:
        base = request.host_url.rstrip("/")
    webhook_url = f"{base}/api/telegram/webhook"
    ok = tgi.set_webhook(webhook_url)
    cmd_ok = tgi.set_commands()
    return jsonify({"ok": ok, "webhook_url": webhook_url, "commands_registered": cmd_ok})


@app.post("/api/telegram/delete-webhook")
def api_telegram_delete_webhook():
    """Remove the webhook (switch to polling mode)."""
    import telegram_interactive as tgi
    ok = tgi.delete_webhook()
    return jsonify({"ok": ok})


@app.get("/api/scanners")
def api_scanners():
    """Latest scanner results (cached from the most recent scan)."""
    intraday = _scanner_cache.get("intraday", {})
    eod = _scanner_cache.get("eod", {})
    return jsonify({**intraday, **eod})


@app.get("/api/scanners/intraday")
def api_scanners_intraday():
    """Latest intraday scanner results."""
    return jsonify(_scanner_cache.get("intraday", {}))


@app.get("/api/scanners/eod")
def api_scanners_eod():
    """Latest EOD scanner results (streaks, breakouts)."""
    return jsonify(_scanner_cache.get("eod", {}))


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


@app.post("/api/subscription/cancel")
def api_subscription_cancel():
    """Cancel the current user's subscription. User keeps access until period ends."""
    user = request.environ.get("auth_user") or {}
    email = user.get("email") or ""
    if not email:
        return jsonify({"error": "Sign in required"}), 401
    
    body = request.get_json(silent=True) or {}
    subscription_id = body.get("subscription_id")  # Optional, will look up from DB if not provided
    
    try:
        from subscription import cancel_subscription
        result = cancel_subscription(email, subscription_id)
        if result.get("success"):
            return jsonify({
                "success": True, 
                "message": "Subscription cancelled",
                "expires_at": result.get("expires_at"),  # When access ends
            })
        return jsonify({"success": False, "error": result.get("error", "Cancel failed")}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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


def _boot():
    """
    One load at startup, now only for the things that genuinely need a panel.

    The lists no longer depend on this: /api/dashboard and /api/episodes read
    Supabase, so the app is usable the moment the process is up, whether or
    not this has finished. What still needs the in-memory panel is the sector
    and stock drawers, the symbol list behind the header lookup, and the
    post-market cron itself, which has to compute before it can save.

    Everything after this is driven externally by POST
    /api/cron/post-market -- there is no internal scheduler.
    """
    try:
        engine.load(DEFAULT_DAYS, date.today())
        # Pre-compute bot cache after startup load
        try:
            import telegram_interactive as tgi
            tgi.precompute_bot_cache(engine)
        except Exception:
            pass
    except Exception as exc:
        engine._set(status="error", error=str(exc), message="Startup load failed.")


if __name__ == "__main__":
    threading.Thread(target=_boot, daemon=True).start()
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)
