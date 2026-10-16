"""
Live side of the commodities view: prices, news, and per-stock drivers.

Prices come from yfinance (cached 30 min). Stock links are read from
data/commodity_links.json, which `commodities.py export` writes offline.

News comes from GDELT, which allows commercial use but limits callers to one
request every few seconds and often answers 429 regardless. So news is never
fetched in the request path: a single background thread refreshes stale
drivers one at a time, and requests serve whatever is cached. A driver with no
news yet says so instead of blocking the page.
"""

from __future__ import annotations

import json
import re
import threading
import time
import warnings

import numpy as np
import pandas as pd
import requests

import commodities as cm

warnings.filterwarnings("ignore")

PRICE_TTL = 30 * 60
NEWS_TTL = 3 * 60 * 60
NEWS_RETRY = 15 * 60
NEWS_GAP = 7.0
NEWS_MAX = 6
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"

_price_lock = threading.Lock()
_prices: tuple[float, dict] | None = None

_news_lock = threading.Lock()
_news: dict[str, dict] = {}
_worker: threading.Thread | None = None

_links: dict | None = None


# --------------------------------------------------------------------------
# Links
# --------------------------------------------------------------------------

def links() -> dict:
    global _links
    if _links is None:
        try:
            _links = json.loads(cm.LINKS_FILE.read_text())
        except FileNotFoundError:
            _links = {"links": {}, "window": None, "method": None}
    return _links


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------

def _chg(s: pd.Series, n: int, points: bool) -> float | None:
    if len(s) <= n:
        return None
    a, b = float(s.iloc[-1]), float(s.iloc[-1 - n])
    if points:
        return round(a - b, 3)
    return round((a / b - 1.0) * 100, 2) if b else None


def prices() -> dict:
    """{key: {price, chg_1d, chg_1w, chg_1m, chg_3m, spark, as_of}}."""
    global _prices
    now = time.monotonic()
    with _price_lock:
        if _prices and _prices[0] > now:
            return _prices[1]
        import yfinance as yf
        tickers = [cm.TICKERS[k] for k, *_ in cm.DISPLAY]
        out: dict = {}
        try:
            px = yf.download(tickers, period="6mo", interval="1d",
                             progress=False, auto_adjust=True)["Close"]
        except Exception as exc:
            print(f"[commodities] price fetch failed: {exc}")
            px = pd.DataFrame()
        for key, *_ in cm.DISPLAY:
            t = cm.TICKERS[key]
            s = px[t].dropna() if t in px else pd.Series(dtype=float)
            if s.empty:
                continue
            points = key == "US10Y"
            out[key] = {
                "price": round(float(s.iloc[-1]), 3),
                "as_of": f"{s.index[-1]:%Y-%m-%d}",
                "chg_1d": _chg(s, 1, points),
                "chg_1w": _chg(s, 5, points),
                "chg_1m": _chg(s, 21, points),
                "chg_3m": _chg(s, 63, points),
                "points": points,
                "spark": [round(float(v), 3) for v in s.tail(66)],
            }
        # A failed fetch is retried in 2 minutes rather than held for 30.
        _prices = (now + (PRICE_TTL if out else 120), out)
        return out


# --------------------------------------------------------------------------
# News
# --------------------------------------------------------------------------

def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _fetch_news(query: str) -> tuple[list[dict] | None, str]:
    try:
        r = requests.get(GDELT, params={
            "query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
            "maxrecords": 40, "timespan": "3d", "sort": "hybridrel"}, timeout=20)
    except requests.RequestException as exc:
        return None, f"unreachable: {exc.__class__.__name__}"
    if r.status_code == 429:
        return None, "rate limited"
    try:
        arts = r.json().get("articles", [])
    except ValueError:
        return None, f"bad response ({r.status_code})"
    seen, items = set(), []
    for a in arts:
        key = _norm_title(a.get("title", ""))[:80]
        if not key or key in seen:
            continue
        seen.add(key)
        d = a.get("seendate", "")
        items.append({
            "title": a.get("title", "").strip(),
            "url": a.get("url"),
            "source": a.get("domain"),
            "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) >= 8 else None,
        })
        if len(items) >= NEWS_MAX:
            break
    return items, "ok"


def _stale_keys() -> list[str]:
    now = time.monotonic()
    with _news_lock:
        return [k for k, *_ in cm.DISPLAY
                if _news.get(k, {}).get("next", 0) <= now]


def _work() -> None:
    queries = {k: q for k, _, _, q in cm.DISPLAY}
    while True:
        keys = _stale_keys()
        if not keys:
            time.sleep(60)
            continue
        for key in keys:
            items, status = _fetch_news(queries[key])
            now = time.monotonic()
            with _news_lock:
                prev = _news.get(key, {})
                if items is not None:
                    _news[key] = {"items": items, "status": "ok",
                                  "fetched": time.time(), "next": now + NEWS_TTL}
                else:
                    # Keep serving older headlines when a refresh fails.
                    _news[key] = {"items": prev.get("items", []), "status": status,
                                  "fetched": prev.get("fetched"), "next": now + NEWS_RETRY}
            # A 429 usually means GDELT wants a longer pause than its stated limit.
            time.sleep(NEWS_GAP * (8 if status == "rate limited" else 1))


def _ensure_worker() -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_work, daemon=True, name="commodity-news")
        _worker.start()


def news(key: str) -> dict:
    _ensure_worker()
    with _news_lock:
        n = _news.get(key)
    if not n:
        return {"items": [], "status": "loading", "fetched": None}
    return {"items": n["items"], "status": n["status"], "fetched": n["fetched"]}


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------

def _driver_meta(key: str) -> dict:
    group, unit, _ = next((g, u, q) for k, g, u, q in cm.DISPLAY if k == key)
    return {"key": key, "label": cm.LABELS[key], "group": group, "unit": unit}


def overview() -> dict:
    px = prices()
    lk = links()
    drivers = []
    for key, *_ in cm.DISPLAY:
        rows = lk["links"].get(key, [])
        drivers.append({
            **_driver_meta(key),
            "market": px.get(key),
            "with": [r for r in rows if r["beta"] > 0],
            "against": [r for r in rows if r["beta"] < 0],
            "news": news(key),
        })
    return {"drivers": drivers, "window": lk.get("window"), "method": lk.get("method")}


_industries: dict | None = None


def _industry(sym: str) -> str | None:
    global _industries
    if _industries is None:
        try:
            smap = pd.read_json(cm.fetch.CACHE_DIR / "sector_map.json", orient="index")
            _industries = smap["basic_industry"].dropna().to_dict()
        except Exception:
            _industries = {}
    return _industries.get(sym)


def for_stock(symbol: str) -> dict:
    """
    Drivers for one stock: links measured on this stock first, then drivers
    its industry is expected to be exposed to but that did not show up in this
    stock's own price history.
    """
    sym = symbol.upper()
    industry = _industry(sym)
    px = prices()
    lk = links()["links"]
    measured, seen = [], set()
    for key, *_ in cm.DISPLAY:
        hit = next((r for r in lk.get(key, []) if r["symbol"] == sym), None)
        if hit:
            seen.add(key)
            measured.append({**_driver_meta(key), **hit, "market": px.get(key), "news": news(key)})
    expected = [{**_driver_meta(key), "market": px.get(key), "news": news(key)}
                for key, *_ in cm.DISPLAY
                if key not in seen and industry and industry in cm.PRIORS.get(key, set())]
    measured.sort(key=lambda r: -abs(r["t"]))
    return {"symbol": sym, "industry": industry, "measured": measured, "expected": expected,
            "window": links().get("window")}
