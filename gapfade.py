"""
Gap-ups in downtrends: the 09:00-09:30 screen.

Lists non-F&O cash stocks that open 2-9% above yesterday's close while
yesterday's close sat below the 50-day average. The rule was re-priced on
5-minute bars over three years in attic/gap_5m.py (see BACKTEST).

Prices come from NSE's pre-open endpoint: the indicative equilibrium price
(IEP) from 09:00, final once the order-entry window shuts at ~09:08. After
09:15 the current price is added so the move since the open is visible.
"""

from __future__ import annotations

import io
import threading
import time
from datetime import datetime, time as dtime, timedelta, timezone

import numpy as np
import pandas as pd

import fetch
import filing_facts
import news

PREOPEN_URL = fetch.BASE + "/api/market-data-pre-open?key=ALL"
PREOPEN_REFERER = fetch.BASE + "/market-data/pre-open-market-cm-and-emerge-market"
BAND_URL = fetch.ARCHIVES + "/content/equities/sec_list.csv"
ETF_URL = fetch.ARCHIVES + "/content/equities/eq_etfseclist.csv"
ETF_NAME = r"ETF|BEES|^SETF|GOLD|SILVER|LIQUID|CASE$|NIFTY|SENSEX|IETF|MODEFENCE"  # attic/tab_audit.py
ANNOUNCEMENTS_ALL = (fetch.BASE + "/api/corporate-announcements?index=equities"
                     "&from_date={frm}&to_date={to}")

GAP_MIN, GAP_MAX = 0.02, 0.09
MIN_PRICE = 50
MIN_TURNOVER_CR = 10
TOP_N = 5
NEAR_BAND = 0.02          # flag when the open sits within 2% of the upper band
CACHE_TTL = 30
LIVE_CAP = 25             # current-price lookups per refresh
BROAD_OPEN = 0.0075       # median liquid stock opening this far up = a broad gap-up morning
FILINGS_TTL = 5 * 60
GLOBAL_MIN = 0.01         # driver rise since the previous Indian close that earns the tag
GLOBAL_TTL = 15 * 60
# attic/gap_global.py: industry (sector_map basic_industry) -> overnight global driver.
GLOBAL_DRIVERS = {
    "Copper": ("HG=F", {"Aluminium", "Aluminium Copper & Zinc Products", "Copper", "Zinc",
                        "Diversified Metals", "Trading - Metals", "Iron & Steel",
                        "Iron & Steel Products", "Sponge Iron", "Pig Iron"}),
    "Nasdaq": ("^IXIC", {"IT Enabled Services", "Computers - Software & Consulting",
                         "Software Products", "Computers Hardware & Equipments"}),
    "Crude": ("CL=F", {"Oil Exploration & Production", "Oil Equipment & Services"}),
    "Gold": ("GC=F", {"Gems Jewellery And Watches", "Precious Metals"}),
}

# attic/gap_5m.py: every gap-up matching this screen, priced on Groww 5-minute bars.
# Positive = the stock fell from the entry (a short gained), after costs.
BACKTEST = {
    "window": "Dec 2023 - Nov 2026 (5-minute bars)",
    "trades": 5307,
    "median_pct": -0.37,
    "avg_pct": -0.53,
    "win_pct": 41.0,
    "entry_0925_median_pct": -0.38,
    "open_to_0920_pct": -0.63,
    "cost_bps": 25,
    "notes": [
        "Measured from 09:20 to 15:15 on every qualifying stock, after 0.25% round-trip costs.",
        "Most of the fall happened before 09:20: the median stock was already 0.63% below its "
        "opening price by then, and the remainder of the day tended to recover.",
        "Stocks with a company filing between the previous close and the open did better "
        "(median +0.06%, 912 instances) but were still roughly flat after costs.",
        "The live record above is the running check on this list as it is published.",
    ],
    # attic/gap_traits.py: the top 5 by gap, split by the median opening gap of all liquid stocks.
    "broad_open": {
        "threshold_pct": BROAD_OPEN * 100,
        "trades": 173, "sessions": 36, "of_sessions": 602,
        "median_pct": -0.96, "win_pct": 33.0,
        "by_year": {"2024": -0.99, "2025": -0.66, "2026": -1.00},
        "other_trades": 1734, "other_median_pct": 0.09, "other_win_pct": 51.0,
    },
    # attic/gap_global.py: every downtrend gap-up in a driver-linked industry.
    "global_driver": {
        "threshold_pct": GLOBAL_MIN * 100,
        "trades": 233, "median_pct": -0.96, "win_pct": 33.0,
        "other_trades": 414, "other_median_pct": -0.25, "other_win_pct": 44.0,
    },
}

IST = timezone(timedelta(hours=5, minutes=30))

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "payload": None}
_band_cache: dict = {"day": None, "bands": {}}
_etf_cache: dict = {"day": None, "symbols": set()}
_base_cache: dict = {"key": None, "frame": None, "days": []}
_filings_cache: dict = {"at": 0.0, "key": None, "by": {}}
_global_cache: dict = {"at": 0.0, "key": None, "moves": {}}
_industry_map: dict | None = None


def _industry(sym: str) -> str | None:
    global _industry_map
    if _industry_map is None:
        try:
            smap = pd.read_json(fetch.CACHE_DIR / "sector_map.json", orient="index")
            _industry_map = smap["basic_industry"].dropna().to_dict()
        except Exception:
            _industry_map = {}
    return _industry_map.get(sym)


def _now_ist() -> datetime:
    return datetime.now(IST)


def phase(now: datetime | None = None) -> str:
    now = now or _now_ist()
    if now.weekday() >= 5:
        return "closed"
    m = now.hour * 60 + now.minute
    if m < 9 * 60:
        return "before"
    if m < 9 * 60 + 8:
        return "preopen"
    if m < 9 * 60 + 15:
        return "open_set"
    if m <= 9 * 60 + 30:
        return "window"
    return "after"


def _bands(sess) -> dict[str, str]:
    day = _now_ist().date().isoformat()
    if _band_cache["day"] == day and _band_cache["bands"]:
        return _band_cache["bands"]
    try:
        r = sess.get(BAND_URL)
        d = pd.read_csv(io.StringIO(r.text))
        d = d[d["Series"].astype(str).str.strip() == "EQ"]
        bands = dict(zip(d["Symbol"].astype(str).str.strip(),
                         d["Band"].astype(str).str.strip()))
        _band_cache.update(day=day, bands=bands)
    except Exception as exc:
        print(f"gapfade bands: {exc}")
    return _band_cache["bands"]


def _etfs(sess) -> set[str]:
    """NSE's listed ETFs. The rule was tested on company shares only; an ETF's gap is its underlying's overnight move."""
    day = _now_ist().date().isoformat()
    if _etf_cache["day"] == day and _etf_cache["symbols"]:
        return _etf_cache["symbols"]
    try:
        r = sess.get(ETF_URL)
        d = pd.read_csv(io.StringIO(r.text))
        syms = set(d["Symbol"].astype(str).str.strip().str.upper())
        if syms:
            _etf_cache.update(day=day, symbols=syms)
    except Exception as exc:
        print(f"gapfade etf list: {exc}")
    return _etf_cache["symbols"]


def _base(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol history facts: 50-day average close, 20-day median turnover."""
    key = (len(raw), str(raw["date"].max()))
    if _base_cache["key"] == key:
        return _base_cache["frame"]
    d = raw[raw["series"].astype(str).str.upper() == "EQ"]
    d = d[(d["close"] > 0)].sort_values(["symbol", "date"])
    g = d.groupby("symbol")
    out = pd.DataFrame({
        "sma50": g["close"].apply(lambda s: s.tail(50).mean() if len(s) >= 50 else np.nan),
        "med_turnover_cr": g["turnover"].apply(lambda s: s.tail(20).median() / 100),
        "avg_vol20": g["volume"].apply(lambda s: s.tail(20).mean()),
        "hist_close": g["close"].last(),
        "hist_date": g["date"].last(),
    }).reset_index()
    days = sorted(pd.to_datetime(pd.Series(d["date"].unique())).dt.date)[-5:]
    _base_cache.update(key=key, frame=out, days=days)
    return out


def _overnight_filings(sess) -> dict[str, dict]:
    """Medium/high-importance filings after the previous session's close and before 09:15 today."""
    today = _now_ist().date()
    prior = [d for d in _base_cache["days"] if d < today]
    if not prior:
        return {}
    prev_day = prior[-1]
    key = (prev_day, today)
    if _filings_cache["key"] == key and time.time() - _filings_cache["at"] < FILINGS_TTL:
        return _filings_cache["by"]
    url = ANNOUNCEMENTS_ALL.format(frm=prev_day.strftime("%d-%m-%Y"), to=today.strftime("%d-%m-%Y"))
    try:
        rows = sess.get_json(url, referer=news.REFERER)
    except Exception as exc:
        print(f"gapfade filings: {exc}")
        return _filings_cache["by"] if _filings_cache["key"] == key else {}
    if isinstance(rows, dict):
        rows = rows.get("data") or []
    lo = datetime.combine(prev_day, dtime(15, 30))
    hi = datetime.combine(today, dtime(9, 15))
    rank = {"high": 0, "normal": 1}
    by: dict[str, dict] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        when = news._parse_dt(r.get("an_dt")) or news._parse_dt(r.get("sort_date"))
        if when is None or not (lo < when <= hi):
            continue
        f = news._shape(r)
        if f is None or f["importance"] == "low":
            continue
        sym = (r.get("symbol") or "").strip().upper()
        cur = by.get(sym)
        if cur is None or rank.get(f["importance"], 2) < rank.get(cur["importance"], 2):
            by[sym] = {k: f[k] for k in ("date", "time", "category", "importance", "title", "url")}
    _filings_cache.update(at=time.time(), key=key, by=by)
    return by


def _preopen(sess) -> tuple[pd.DataFrame, str | None]:
    d = sess.get_json(PREOPEN_URL, referer=PREOPEN_REFERER)
    rows, stamp = [], None
    for item in d.get("data") or []:
        m = item.get("metadata") or {}
        po = (item.get("detail") or {}).get("preOpenMarket") or {}
        if str(m.get("series", "")).upper() != "EQ":
            continue
        stamp = stamp or po.get("lastUpdateTime")
        price = po.get("finalPrice") or m.get("iep") or m.get("lastPrice")
        rows.append({
            "symbol": str(m.get("symbol", "")).strip().upper(),
            "prev_close": pd.to_numeric(m.get("previousClose"), errors="coerce"),
            "open": pd.to_numeric(price, errors="coerce"),
            "preopen_qty": pd.to_numeric(m.get("finalQuantity"), errors="coerce"),
        })
    return pd.DataFrame(rows), stamp


def _live_prices(symbols: list[str]) -> dict[str, float]:
    """Groww first; Yahoo 1-minute bars when Groww is unavailable. NSE's
    per-symbol quote endpoint rejects server-side requests, so it is skipped."""
    symbols = symbols[:LIVE_CAP]
    if not symbols:
        return {}
    try:
        q = fetch.live_quotes_groww(symbols)
        got = {r.symbol: float(r.ltp) for r in q.itertuples() if pd.notna(r.ltp)}
        if got:
            return got
    except Exception as exc:
        print(f"gapfade groww prices: {exc}")
    try:
        import yfinance as yf
        bars = yf.download([f"{s}.NS" for s in symbols], period="1d", interval="1m",
                           group_by="ticker", progress=False, auto_adjust=False)
        out = {}
        for s in symbols:
            t = f"{s}.NS"
            if t in bars.columns.get_level_values(0):
                c = bars[t]["Close"].dropna()
                if len(c):
                    out[s] = float(c.iloc[-1])
        return out
    except Exception as exc:
        print(f"gapfade yahoo prices: {exc}")
        return {}


def _global_moves() -> dict[str, float]:
    """Each driver's move from the previous Indian close (15:30 IST) to its latest price.
    The reference is the last hourly bar at or before that close; the latest price comes from
    the daily feed, because Yahoo's hourly futures bars can stall for a session or more."""
    today = _now_ist().date()
    prior = [d for d in _base_cache["days"] if d < today]
    if not prior:
        return {}
    key = (prior[-1], today)
    if _global_cache["key"] == key and time.time() - _global_cache["at"] < GLOBAL_TTL:
        return _global_cache["moves"]
    ref_at = pd.Timestamp(datetime.combine(prior[-1], dtime(15, 30)), tz=IST)
    moves: dict[str, float] = {}
    try:
        import yfinance as yf
        tickers = [t for t, _ in GLOBAL_DRIVERS.values()]
        hourly = yf.download(tickers, period="7d", interval="60m", progress=False, auto_adjust=False)["Close"]
        hourly.index = pd.to_datetime(hourly.index).tz_convert(IST)
        daily = yf.download(tickers, period="10d", interval="1d", progress=False, auto_adjust=False)["Close"]
        for name, (t, _) in GLOBAL_DRIVERS.items():
            h = hourly[t].dropna() if t in hourly else pd.Series(dtype=float)
            d = daily[t].dropna() if t in daily else pd.Series(dtype=float)
            before = h[h.index <= ref_at]
            if len(before) and len(d):
                moves[name] = float(d.iloc[-1] / before.iloc[-1] - 1)
    except Exception as exc:
        print(f"gapfade global moves: {exc}")
        return _global_cache["moves"] if _global_cache["key"] == key else {}
    _global_cache.update(at=time.time(), key=key, moves=moves)
    return moves


def _global_tag(sym: str, moves: dict[str, float]) -> dict | None:
    ind = _industry(sym)
    for name, (_, industries) in GLOBAL_DRIVERS.items():
        if ind in industries and moves.get(name, 0) >= GLOBAL_MIN:
            return {"driver": name, "move_pct": round(moves[name] * 100, 2), "industry": ind}
    return None


def _first_bar_volumes(symbols: list[str]) -> dict[str, float]:
    """Volume of today's 09:15 five-minute bar (Groww), the early-volume measure in attic/gap_rank.py."""
    if not symbols:
        return {}
    try:
        client = fetch._get_groww_client()
    except Exception as exc:
        print(f"gapfade first bar: {exc}")
        return {}
    day = _now_ist().strftime("%Y-%m-%d")
    out = {}
    for sym in symbols:
        try:
            c = client.get_historical_candles(
                exchange="NSE", segment="CASH", groww_symbol=f"NSE-{sym}",
                start_time=f"{day} 09:15:00", end_time=f"{day} 09:20:00",
                candle_interval="5minute")
            rows = (c.get("candles") if isinstance(c, dict) else c) or []
            if rows and rows[0][5] is not None:
                out[sym] = float(rows[0][5])
        except Exception as exc:
            print(f"gapfade first bar {sym}: {exc}")
        time.sleep(0.2)
    return out


def _fno() -> set[str]:
    s = filing_facts.fno_symbols()
    if s:
        return s
    files = sorted((fetch.RAW_DIR / "fo").glob("fo_*.csv.gz"))
    if files:
        return set(pd.read_csv(files[-1])["symbol"].astype(str).str.strip().str.upper())
    return set()


def scan(raw: pd.DataFrame) -> dict:
    sess = fetch.NSESession()
    pre, stamp = _preopen(sess)
    if pre.empty:
        return {"error": "NSE pre-open data is not available yet."}

    base = _base(raw)
    bands = _bands(sess)
    fno = _fno()

    df = pre.merge(base, on="symbol", how="inner")
    df = df[(df["open"] > 0) & (df["prev_close"] > 0)]
    etfs = _etfs(sess)
    df = df[~df["symbol"].isin(etfs) & ~df["symbol"].str.contains(ETF_NAME, regex=True)]
    df["gap"] = df["open"] / df["prev_close"] - 1
    df["band"] = df["symbol"].map(bands)

    liquid = df[(df["prev_close"] >= MIN_PRICE) & (df["med_turnover_cr"] >= MIN_TURNOVER_CR)]
    market_gap = float(liquid["gap"].median()) if len(liquid) else None

    universe = df[(df["prev_close"] >= MIN_PRICE)
                  & (df["med_turnover_cr"] >= MIN_TURNOVER_CR)
                  & ~df["symbol"].isin(fno)
                  & (df["prev_close"] < df["sma50"])]
    gapped = universe[(universe["gap"] >= GAP_MIN) & (universe["gap"] < GAP_MAX)]
    band5 = gapped["band"].isin(["5", "2"])
    hits = gapped[~band5].copy()

    band_frac = pd.to_numeric(hits["band"], errors="coerce") / 100
    upper = hits["prev_close"] * (1 + band_frac)
    hits["to_band_pct"] = (upper / hits["open"] - 1) * 100
    hits["near_band"] = hits["to_band_pct"] < NEAR_BAND * 100
    hits = hits.sort_values("gap", ascending=False).reset_index(drop=True)
    hits["rank"] = np.arange(1, len(hits) + 1)

    ph = phase()
    live = _live_prices(list(hits["symbol"])) if ph in ("window", "after") else {}
    hits["ltp"] = hits["symbol"].map(live)
    hits["from_open_pct"] = (hits["ltp"] / hits["open"] - 1) * 100
    filings = _overnight_filings(sess) if len(hits) else {}
    moves = _global_moves() if len(hits) else {}

    def num(v, nd=2):
        return None if v is None or pd.isna(v) else round(float(v), nd)

    rows = [{
        "symbol": r.symbol,
        "rank": int(r.rank),
        "top": bool(r.rank <= TOP_N),
        "prev_close": num(r.prev_close),
        "open": num(r.open),
        "gap_pct": num(r.gap * 100),
        "sma50": num(r.sma50),
        "below_sma50_pct": num((r.prev_close / r.sma50 - 1) * 100),
        "med_turnover_cr": num(r.med_turnover_cr, 1),
        "band": None if pd.isna(r.band) else str(r.band),
        "to_band_pct": num(r.to_band_pct),
        "near_band": bool(r.near_band) if pd.notna(r.to_band_pct) else False,
        "preopen_qty": None if pd.isna(r.preopen_qty) else int(r.preopen_qty),
        "ltp": num(r.ltp),
        "from_open_pct": num(r.from_open_pct),
        "filing": filings.get(r.symbol),
        "global_move": _global_tag(r.symbol, moves),
    } for r in hits.itertuples()]

    return {
        "phase": ph,
        "preopen_time": stamp,
        "fetched_at": _now_ist().strftime("%Y-%m-%d %H:%M:%S"),
        "history_as_of": str(pd.to_datetime(base["hist_date"].max()).date()),
        "universe": int(len(universe)),
        "excluded_small_band": int(band5.sum()),
        "market_open": None if market_gap is None else {
            "median_gap_pct": num(market_gap * 100),
            "stocks": int(len(liquid)),
            "broad": bool(market_gap >= BROAD_OPEN),
        },
        "rules": {
            "gap_min_pct": GAP_MIN * 100, "gap_max_pct": GAP_MAX * 100,
            "min_price": MIN_PRICE, "min_turnover_cr": MIN_TURNOVER_CR,
            "top_n": TOP_N,
        },
        "backtest": BACKTEST,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Live track record: picks logged at ~09:20, priced again at ~15:15
# --------------------------------------------------------------------------

RECORD_TTL = 10 * 60
_record_cache: dict = {"at": 0.0, "record": None}


def _session_of(stamp: str | None) -> str | None:
    """Date of NSE's pre-open stamp (e.g. '23-Oct-2026 09:07:59')."""
    if not stamp:
        return None
    try:
        return pd.to_datetime(stamp, format="%d-%b-%Y %H:%M:%S").date().isoformat()
    except (TypeError, ValueError):
        return None


def log_entries(raw: pd.DataFrame) -> dict:
    """Record today's list with the current price as the short entry."""
    import db
    today = _now_ist().date().isoformat()
    out = payload(raw, force=True)
    if "error" in out:
        return {"status": "error", "error": out["error"]}
    if _session_of(out.get("preopen_time")) not in (None, today):
        return {"status": "skipped", "reason": f"no session today (pre-open is from {out['preopen_time']})"}
    # The live table only quotes the first 25 names. The log prices every pick;
    # if Groww/Yahoo miss one, the opening price is the fallback.
    missing = [r["symbol"] for r in out["rows"] if not r.get("ltp")]
    extra: dict[str, float] = {}
    for i in range(0, len(missing), LIVE_CAP):
        extra.update(_live_prices(missing[i:i + LIVE_CAP]))
    # Shadow ranking for the Feb 2027 checkpoint: 09:15-bar volume / 20-day average volume.
    # Logged only; the published top five stays ranked by gap.
    avg = _base_cache["frame"].set_index("symbol")["avg_vol20"] if _base_cache["frame"] is not None else pd.Series(dtype=float)
    vols = _first_bar_volumes([r["symbol"] for r in out["rows"]])
    early = {s: v / avg[s] for s, v in vols.items() if s in avg.index and avg[s] > 0}
    vol_rank = {s: i + 1 for i, s in enumerate(sorted(early, key=early.get, reverse=True))}
    now = _now_ist().isoformat()
    keep = ("symbol", "rank", "top", "prev_close", "open", "gap_pct",
            "below_sma50_pct", "band", "near_band")
    rows = []
    for r in out["rows"]:
        px = r.get("ltp") or extra.get(r["symbol"]) or r.get("open")
        ev = early.get(r["symbol"])
        rows.append({**{k: r[k] for k in keep}, "session_date": today,
                     "entry_price": px, "entry_at": now if px else None,
                     "early_vol": None if ev is None else round(float(ev), 4),
                     "vol_rank": vol_rank.get(r["symbol"])})
    sent = db.save_gap_fade_entries(rows)
    _record_cache["at"] = 0.0
    return {"status": "ok" if sent is not None else "save failed", "session": today,
            "picks": len(rows), "priced": sum(1 for r in rows if r["entry_price"]),
            "early_vol": len(early)}


def log_exits() -> dict:
    """Price today's logged picks and store the short-side return."""
    import db
    today = _now_ist().date().isoformat()
    rows = db.get_gap_fade_log(session_date=today)
    if rows is None:
        return {"status": "error", "error": "could not read gap_fade_log"}
    open_rows = [r for r in rows if r.get("entry_price") and r.get("exit_price") is None]
    if not open_rows:
        return {"status": "ok", "session": today, "closed": 0}
    symbols = [r["symbol"] for r in open_rows]
    prices: dict[str, float] = {}
    for i in range(0, len(symbols), LIVE_CAP):
        prices.update(_live_prices(symbols[i:i + LIVE_CAP]))
    now, closed = _now_ist().isoformat(), 0
    for r in open_rows:
        px = prices.get(r["symbol"])
        if not px:
            continue
        entry = float(r["entry_price"])
        ret = (entry - px) / entry * 100
        if db.set_gap_fade_exit(today, r["symbol"], {
                "exit_price": round(px, 2), "exit_at": now, "short_ret_pct": round(ret, 3)}):
            closed += 1
    _record_cache["at"] = 0.0
    return {"status": "ok", "session": today, "closed": closed, "unpriced": len(open_rows) - closed}


def _summary(rets: list[float]) -> dict | None:
    if not rets:
        return None
    a = np.array(rets)
    return {"trades": len(a), "avg_pct": round(float(a.mean()), 2),
            "median_pct": round(float(np.median(a)), 2),
            "win_pct": round(float((a > 0).mean() * 100), 1)}


def record() -> dict | None:
    """Net-of-cost results of logged picks, overall and for the top five."""
    if _record_cache["record"] is not None and time.time() - _record_cache["at"] < RECORD_TTL:
        return _record_cache["record"]
    import db
    rows = db.get_gap_fade_log()
    if rows is None:
        return _record_cache["record"]
    cost = BACKTEST["cost_bps"] / 100
    done = [r for r in rows if r.get("short_ret_pct") is not None]
    net = lambda rs: [float(r["short_ret_pct"]) - cost for r in rs]
    sessions: dict[str, list] = {}
    for r in done:
        sessions.setdefault(r["session_date"], []).append({
            "symbol": r["symbol"], "rank": r["rank"], "top": r["top"],
            "gap_pct": r["gap_pct"], "entry": r["entry_price"], "exit": r["exit_price"],
            "net_pct": round(float(r["short_ret_pct"]) - cost, 2)})
    out = {
        "since": min(sessions) if sessions else None,
        "sessions": len(sessions),
        "cost_bps": BACKTEST["cost_bps"],
        "top": _summary(net([r for r in done if r.get("top")])),
        "vol_top": _summary(net([r for r in done if (r.get("vol_rank") or 99) <= TOP_N])),
        "all": _summary(net(done)),
        "recent": [{"date": d, "picks": sessions[d]} for d in sorted(sessions, reverse=True)[:10]],
    }
    _record_cache.update(at=time.time(), record=out)
    return out


def payload(raw: pd.DataFrame | None, force: bool = False) -> dict:
    if raw is None or raw.empty:
        return {"error": "Price history is still loading. Try again in a minute."}
    with _lock:
        fresh = time.time() - _cache["at"] < CACHE_TTL
        if _cache["payload"] is not None and fresh and not force:
            return _cache["payload"]
        try:
            out = scan(raw)
            if "error" not in out:
                try:
                    out["live_record"] = record()
                except Exception as exc:
                    print(f"gapfade record: {exc}")
        except Exception as exc:
            print(f"gapfade scan failed: {exc}")
            if _cache["payload"] is not None:
                stale = dict(_cache["payload"], stale=True)
                return stale
            return {"error": f"Could not read NSE pre-open data: {exc}"}
        _cache.update(at=time.time(), payload=out)
        return out
