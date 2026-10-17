"""
Live side of the commodities view: prices, news, and per-stock drivers.

Prices come from yfinance (cached 30 min). Stock links are read from
data/commodity_links.json, which `commodities.py export` writes offline.

News comes from Finlight (FINLIGHT_API_KEY). It is never fetched in the
request path: POST /api/cron/commodity-news calls refresh_news(), which stores
articles in the commodity_news table, and pages read the last 14 days from
there. One cron run is 10 Finlight calls; every 3 hours that is ~2,400 a
month, inside the free plan's 5,000.
"""

from __future__ import annotations

import json
import os
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
NEWS_CACHE = 5 * 60
NEWS_GAP = 1.0
NEWS_MAX = 6
FINLIGHT = "https://api.finlight.me/v2/articles"
# Press-release wires: company announcements, not market news.
NEWS_EXCLUDE = " AND ".join(f"-source:{s}" for s in (
    "www.prnewswire.com", "www.accessnewswire.com", "www.globenewswire.com", "www.businesswire.com"))

_price_lock = threading.Lock()
_prices: tuple[float, dict] | None = None

_news_lock = threading.Lock()
_news: tuple[float, dict] | None = None

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
    key = os.environ.get("FINLIGHT_API_KEY", "").strip()
    if not key:
        return None, "FINLIGHT_API_KEY not set"
    try:
        r = requests.post(FINLIGHT, json={
            "query": f"({query}) AND {NEWS_EXCLUDE}", "language": "en", "pageSize": 20},
            headers={"X-API-KEY": key}, timeout=20)
    except requests.RequestException as exc:
        return None, f"unreachable: {exc.__class__.__name__}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    try:
        arts = r.json().get("articles", [])
    except ValueError:
        return None, "bad response"
    return [{
        "title": a["title"].strip(),
        "url": a["link"],
        "source": (a.get("source") or "").removeprefix("www."),
        "published_at": a.get("publishDate"),
    } for a in arts if a.get("title") and a.get("link")], "ok"


def refresh_news() -> dict:
    """Fetch every driver once and store new articles. Called by the cron."""
    import db
    out = {}
    for key, _, _, query in cm.DISPLAY:
        items, status = _fetch_news(query)
        if items is None:
            out[key] = status
        else:
            n = db.save_commodity_news([{**i, "driver": key} for i in items])
            out[key] = "save failed" if n is None else n
        time.sleep(NEWS_GAP)
    global _news
    _news = None
    return out


def _load_news() -> dict[str, dict]:
    import db
    rows = db.get_commodity_news()
    if rows is None:
        return {}
    by: dict[str, dict] = {}
    for r in rows:
        n = by.setdefault(r["driver"], {"items": [], "seen": set(), "fetched": None})
        n["fetched"] = max(filter(None, [n["fetched"], r.get("fetched_at")]), default=None)
        # Syndicated stories run under several outlets with the same headline.
        norm = _norm_title(r["title"])[:80]
        if norm in n["seen"] or len(n["items"]) >= NEWS_MAX:
            continue
        n["seen"].add(norm)
        n["items"].append({"title": r["title"], "url": r["url"], "source": r["source"],
                           "date": (r.get("published_at") or "")[:10] or None})
    return by


def news(key: str) -> dict:
    global _news
    now = time.monotonic()
    with _news_lock:
        if _news is None or _news[0] <= now:
            _news = (now + NEWS_CACHE, _load_news())
        n = _news[1].get(key)
    if not n:
        return {"items": [], "status": "ok", "fetched": None}
    return {"items": n["items"], "status": "ok", "fetched": n["fetched"]}


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
