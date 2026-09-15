"""
NSE data fetching.

Three things live here:
  1. A requests session that NSE will actually talk to (cookie warm-up + headers).
  2. Daily bhavcopy download, cached to disk so you fetch each date once.
  3. Symbol -> basic-industry mapping, cached and resumable.

NSE rejects bare HTTP requests. You must hit the homepage first to pick up
cookies, then reuse that session. This is the single most common reason these
scripts fail.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

_HERE = Path(__file__).resolve().parent
_env_path = _HERE / ".env"
if not _env_path.exists():
    _env_path = _HERE.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

DATA_DIR = _HERE / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"

BASE = "https://www.nseindia.com"
ARCHIVES = "https://nsearchives.nseindia.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


class NSESession:
    """A requests session pre-warmed with NSE cookies."""

    def __init__(self, timeout: int = 30, retries: int = 3, pause: float = 0.4):
        self.timeout = timeout
        self.retries = retries
        self.pause = pause
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self._warm()

    def _warm(self) -> None:
        """Pick up cookies. Without this every archive request 401s."""
        try:
            self.s.get(BASE, timeout=self.timeout)
            time.sleep(self.pause)
            self.s.get(f"{BASE}/all-reports", timeout=self.timeout)
            time.sleep(self.pause)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not reach {BASE}. Check connectivity / VPN / rate limits."
            ) from exc

    def get_json(self, url: str, referer: str = BASE) -> dict:
        r = self.get(url, referer=referer, headers={"Accept": "application/json"})
        return r.json()

    def get(self, url: str, referer: str = BASE, **kw) -> requests.Response:
        last = None
        for attempt in range(self.retries):
            try:
                hdrs = {"Referer": referer}
                extra = kw.pop("headers", None)
                if extra:
                    hdrs.update(extra)
                r = self.s.get(url, timeout=self.timeout, headers=hdrs, **kw)
                if r.status_code == 200:
                    return r
                # 401/403 usually means the cookie went stale mid-run.
                if r.status_code in (401, 403):
                    self._warm()
                last = requests.HTTPError(f"{r.status_code} for {url}")
            except requests.RequestException as exc:
                last = exc
            time.sleep(self.pause * (attempt + 1) * 2)
        raise RuntimeError(f"Failed after {self.retries} attempts: {url}") from last


# --------------------------------------------------------------------------
# Bhavcopy
# --------------------------------------------------------------------------

# sec_bhavdata_full is the file we want: it is the only one carrying delivery.
SEC_BHAV = ARCHIVES + "/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
# UDiFF is the modern replacement but has no delivery columns. Fallback only.
UDIFF = ARCHIVES + "/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"

# Columns in sec_bhavdata_full arrive padded with leading spaces (" SERIES").
SEC_BHAV_RENAME = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "DATE1": "date",
    "PREV_CLOSE": "prev_close",
    "OPEN_PRICE": "open",
    "HIGH_PRICE": "high",
    "LOW_PRICE": "low",
    "LAST_PRICE": "last",
    "CLOSE_PRICE": "close",
    "AVG_PRICE": "vwap",
    "TTL_TRD_QNTY": "volume",
    "TURNOVER_LACS": "turnover",
    "NO_OF_TRADES": "trades",
    "DELIV_QTY": "deliv_qty",
    "DELIV_PER": "deliv_pct",
}

NUMERIC = [
    "prev_close", "open", "high", "low", "last", "close", "vwap",
    "volume", "turnover", "trades", "deliv_qty", "deliv_pct",
]


def _have_parquet() -> bool:
    """
    Parquet is faster but pulls in pyarrow, which is the usual source of
    NumPy 1.x/2.x binary mismatches ("_ARRAY_API not found"). Fall back to
    gzipped CSV so the cache never becomes a dependency problem.
    """
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


_PARQUET = _have_parquet()


def _cache_path(d: date) -> Path:
    ext = "parquet" if _PARQUET else "csv.gz"
    return RAW_DIR / f"bhav_{d:%Y%m%d}.{ext}"


def _ensure_source(df: pd.DataFrame) -> pd.DataFrame:
    """Tag how the day was fetched. Old cache files have no `source` column —
    infer from delivery coverage so a UDiFF day cannot hide as a full bhavcopy."""
    if df.empty:
        return df
    if "source" in df.columns and df["source"].notna().any():
        return df
    df = df.copy()
    cov = df["deliv_pct"].notna().mean() if "deliv_pct" in df.columns else 0.0
    df["source"] = "sec_bhav" if cov >= 0.5 else "udiff"
    return df


def _read_cache(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="gzip")
        df["date"] = pd.to_datetime(df["date"])
    return _ensure_source(df)


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, compression="gzip")


_NO_SESSION_FILE = CACHE_DIR / "no_session.json"
_no_session: set[str] | None = None

# A date is only recorded as a non-session once it is old enough that "the
# file has not landed yet" is off the table. Asking for today's bhavcopy
# before it publishes is answered with yesterday's, which is indistinguishable
# from a holiday -- recording that would permanently blind us to a real
# session.
NO_SESSION_MIN_AGE = timedelta(days=3)


def _no_session_set() -> set[str]:
    """Dates known to have no bhavcopy of their own, remembered across runs."""
    global _no_session
    if _no_session is None:
        try:
            _no_session = {str(x) for x in json.loads(_NO_SESSION_FILE.read_text())}
        except Exception:
            _no_session = set()
    return _no_session


def _mark_no_session(d: date) -> None:
    """
    Remember that `d` is not a trading session.

    Holidays are otherwise re-requested on every single load, forever: a
    rejection caches nothing, so the fourteen holidays in a year of history
    cost fourteen round trips plus rate-limit pauses every time the panel is
    rebuilt. Recording them turns that into a dictionary lookup.
    """
    if date.today() - d < NO_SESSION_MIN_AGE:
        return
    known = _no_session_set()
    if d.isoformat() in known:
        return
    known.add(d.isoformat())
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _NO_SESSION_FILE.write_text(json.dumps(sorted(known)))
    except Exception as exc:
        print(f"  could not record non-session {d}: {exc}")


class StaleBhavcopy(RuntimeError):
    """
    The file NSE returned is for a different session than the one requested.

    Asking for a holiday does not 404 -- NSE serves the previous session's
    file. Accepting it stamps a duplicate session into the panel under the
    wrong date: every return on that day is exactly zero, which drags down
    realised range and volatility, corrupts every session-counting indicator,
    and publishes a scan under a date the market never traded.

    A subclass of RuntimeError so the existing fallback chain in
    fetch_bhavcopy handles it like any other failed source.
    """

    def __init__(self, asked: date, got) -> None:
        super().__init__(f"asked for {asked}, file is dated {got}")
        self.asked = asked
        self.got = got


def _check_date(stamped: pd.Series, d: date) -> None:
    """Reject a file whose own trade date is not `d`."""
    real = pd.to_datetime(stamped, errors="coerce").dropna().unique()
    if len(real) == 0:
        return  # nothing to check against; the caller's other guards apply
    got = pd.Timestamp(real[0]).date()
    if len(real) == 1 and got != d:
        raise StaleBhavcopy(d, got)


def _parse_sec_bhav(text: str, d: date) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=SEC_BHAV_RENAME)

    # DATE1 carries the session the file is actually for. Checked before the
    # blind restamp below, which would otherwise hide a holiday's stale file.
    if "date" in df:
        _check_date(df["date"].astype(str).str.strip(), d)

    for col in ("symbol", "series"):
        if col in df:
            df[col] = df[col].astype(str).str.strip()

    # Delivery columns carry "-" for rows where delivery doesn't apply.
    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().replace({"-": None, "": None}),
                errors="coerce",
            )

    df["date"] = pd.Timestamp(d)
    keep = ["symbol", "series", "date"] + [c for c in NUMERIC if c in df]
    return df[keep]


def _parse_udiff(content: bytes, d: date) -> pd.DataFrame:
    """Fallback. Has OHLC and turnover but NO delivery — those come back NaN."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = z.namelist()[0]
        df = pd.read_csv(z.open(name))
    df.columns = [c.strip() for c in df.columns]

    # Same holiday trap as sec_bhavdata: TradDt is the session the file is
    # really for, so it is checked before the requested date is stamped on.
    if "TradDt" in df.columns:
        _check_date(df["TradDt"], d)

    out = pd.DataFrame({
        "symbol": df["TckrSymb"].astype(str).str.strip(),
        "series": df["SctySrs"].astype(str).str.strip(),
        "date": pd.Timestamp(d),
        "prev_close": pd.to_numeric(df["PrvsClsgPric"], errors="coerce"),
        "open": pd.to_numeric(df["OpnPric"], errors="coerce"),
        "high": pd.to_numeric(df["HghPric"], errors="coerce"),
        "low": pd.to_numeric(df["LwPric"], errors="coerce"),
        "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
        "volume": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        # UDiFF turnover is in rupees; sec_bhavdata is in lakhs. Normalise.
        "turnover": pd.to_numeric(df["TtlTrfVal"], errors="coerce") / 1e5,
        "trades": pd.to_numeric(df["TtlNbOfTxsExctd"], errors="coerce"),
    })
    out["last"] = out["close"]
    out["vwap"] = pd.NA
    out["deliv_qty"] = pd.NA
    out["deliv_pct"] = pd.NA
    return out


def _is_udiff(df: pd.DataFrame) -> bool:
    """True when this day has no usable delivery (UDiFF fallback or empty)."""
    if df is None or df.empty:
        return True
    if "source" in df.columns and df["source"].astype(str).str.startswith("udiff").any():
        return True
    if "deliv_pct" in df.columns:
        return float(df["deliv_pct"].notna().mean()) < 0.5
    return True


def fetch_bhavcopy(d: date, sess: NSESession | None = None,
                   use_cache: bool = True) -> pd.DataFrame | None:
    """
    One day's bhavcopy. Returns None for holidays / dates with no file.

    Cached as parquet after the first pull. A UDiFF cache is not final —
    sec_bhavdata_full (with delivery) often lands hours later, so those
    days are retried on the next load.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(d)

    # Known holiday. Checked before the cache read and before any request,
    # because this is the only one of the three that costs nothing.
    if use_cache and d.isoformat() in _no_session_set():
        return None

    cached = None
    if use_cache and path.exists():
        cached = _read_cache(path)
        if cached is not None and not cached.empty and not _is_udiff(cached):
            return cached

    sess = sess or NSESession()

    try:
        r = sess.get(SEC_BHAV.format(ddmmyyyy=f"{d:%d%m%Y}"),
                     referer=f"{BASE}/all-reports")
        df = _parse_sec_bhav(r.text, d)
        df["source"] = "sec_bhav"
    except StaleBhavcopy as exc:
        # Not a trading session. Any cache entry for this date was written
        # before the check existed and holds the wrong session, so it is
        # removed rather than served for the rest of the install's life.
        print(f"  {d}: no session — {exc}")
        path.unlink(missing_ok=True)
        _mark_no_session(d)
        return None
    except RuntimeError:
        if cached is not None and not cached.empty:
            return cached
        try:
            r = sess.get(UDIFF.format(yyyymmdd=f"{d:%Y%m%d}"),
                         referer=f"{BASE}/all-reports")
            df = _parse_udiff(r.content, d)
            df["source"] = "udiff"
            print(f"  {d}: fell back to UDiFF (no delivery data for this date)")
        except RuntimeError:
            return None  # holiday, weekend, or not published yet

    if df.empty:
        return cached if cached is not None and not cached.empty else None
    _write_cache(df, path)
    return df


def candidate_dates(end: date, n_sessions: int, lookback_slack: float = 1.6):
    """
    Weekdays walking back from `end`. Holidays just fail the fetch and get
    skipped, so we over-generate rather than maintaining a holiday calendar.
    """
    out, d = [], end
    limit = int(n_sessions * lookback_slack) + 20
    while len(out) < limit:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out


def load_history(end: date, n_sessions: int, sess: NSESession | None = None,
                 verbose: bool = True, on_progress=None) -> pd.DataFrame:
    """Fetch and concatenate `n_sessions` trading days ending at `end`.

    A session is opened only when a date is missing from cache, so a fully
    cached load never talks to NSE. `on_progress(seen, total, date)` is
    optional — used by the UI to show fetch progress.
    """
    frames, seen = [], 0
    for d in candidate_dates(end, n_sessions):
        if seen >= n_sessions:
            break
        if sess is None and not _cache_path(d).exists():
            sess = NSESession()
        df = fetch_bhavcopy(d, sess)
        if df is None:
            continue
        frames.append(df)
        seen += 1
        if verbose:
            print(f"  {d} … {len(df):,} rows ({seen}/{n_sessions})")
        if on_progress is not None:
            on_progress(seen, n_sessions, d)
        if sess is not None and not _cache_path(d).exists():
            time.sleep(sess.pause)
    if not frames:
        raise RuntimeError("No bhavcopy data fetched. Check connectivity.")
    out = pd.concat(frames, ignore_index=True)
    # Cached CSV and freshly parsed frames can carry different datetime units
    # (ns vs us). Normalise so groupby/merge on date behaves consistently.
    out["date"] = pd.to_datetime(out["date"])
    out = _ensure_source(out)
    out = drop_repeat_sessions(out, verbose=verbose)
    return out.sort_values(["symbol", "date"])


def drop_repeat_sessions(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove a session that is an exact copy of the session before it.

    The date check in the parsers stops new stale files getting in, but a
    cache written before that check holds the wrong session already stamped
    with the requested date, and nothing inside the row can reveal it. What
    does reveal it is the copy itself: two sessions where every symbol closed
    at exactly the same price cannot both be real, and the later one is the
    holiday that served the earlier one's file.

    Compared on the full (symbol, close) vector rather than a sample, so a
    genuine session that merely resembles its predecessor is never dropped.
    """
    if df.empty or "close" not in df.columns:
        return df

    sig = {
        d: tuple(g.sort_values("symbol")[["symbol", "close"]]
                 .itertuples(index=False, name=None))
        for d, g in df.groupby("date", sort=True)
    }

    drop, prev_date, prev = [], None, None
    for d in sorted(sig):
        if prev is not None and sig[d] == prev:
            drop.append(d)
            if verbose:
                print(f"  dropping {pd.Timestamp(d).date()}: identical to "
                      f"{pd.Timestamp(prev_date).date()} — not a session")
            continue
        prev_date, prev = d, sig[d]

    if not drop:
        return df
    kept = df[~df["date"].isin(drop)]
    _purge_cache(drop)
    return kept


def _purge_cache(dates) -> None:
    """Delete cache files for dates that turned out not to be sessions."""
    for d in dates:
        day = pd.Timestamp(d).date()
        p = _cache_path(day)
        if p.exists():
            p.unlink(missing_ok=True)
            print(f"  removed poisoned cache {p.name}")
        _mark_no_session(day)


def delivery_coverage(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    Which sessions have usable delivery data.

    UDiFF fallback (and a genuinely missing delivery file) leave `deliv_pct`
    all-NaN. Delivery quality, B_deliv and every delivery gate then go blank
    for that day. Call this after load so the UI can show it instead of
    silently degrading.
    """
    empty = {
        "as_of_ok": True, "as_of": None, "missing": 0,
        "missing_dates": [], "missing_detail": [], "n_sessions": 0,
    }
    if df is None or df.empty or "deliv_pct" not in df.columns:
        return empty
    g = df.groupby("date", sort=True)
    cov = g["deliv_pct"].apply(lambda s: float(s.notna().mean()))
    src = None
    if "source" in df.columns:
        src = g["source"].agg(
            lambda s: str(s.dropna().mode().iloc[0]) if s.notna().any() else ""
        )
    detail = []
    for d, v in cov.items():
        if v >= threshold:
            continue
        reason = "udiff" if src is not None and str(src.get(d, "")).startswith("udiff") \
            else "no_delivery"
        detail.append({
            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "reason": reason,
            "coverage": round(float(v), 3),
        })
    as_of = cov.index.max()
    return {
        "as_of_ok": bool(float(cov.loc[as_of]) >= threshold),
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        "missing": len(detail),
        "missing_dates": [x["date"] for x in detail[-10:]],
        "missing_detail": detail[-10:],
        "n_sessions": int(len(cov)),
    }


# --------------------------------------------------------------------------
# Sector mapping
# --------------------------------------------------------------------------

EQUITY_LIST = ARCHIVES + "/content/equities/EQUITY_L.csv"
# The old /api/quote-equity endpoint 401s now, and when it does answer,
# industryInfo is empty strings. Classification lives on NextApi.
QUOTE_NEXT = BASE + "/api/NextApi/apiClient/GetQuoteApi"


def _nz(v):
    """Treat NSE blanks ('', '-', None) as missing."""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "-", "null", "None", "nan"):
        return None
    return s


def _has_industry(rec: dict | None) -> bool:
    return _nz((rec or {}).get("basic_industry")) is not None


def fetch_equity_list(sess: NSESession | None = None) -> pd.DataFrame:
    sess = sess or NSESession()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "equity_list.csv"
    if path.exists():
        return pd.read_csv(path)
    r = sess.get(EQUITY_LIST, referer=f"{BASE}/market-data/securities-available-for-trading")
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"SYMBOL": "symbol", "NAME OF COMPANY": "name",
                            "ISIN NUMBER": "isin", "SERIES": "series"})
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df.to_csv(path, index=False)
    return df


def build_sector_map(symbols=None, sess: NSESession | None = None,
                     pause: float = 0.35, verbose: bool = True) -> pd.DataFrame:
    """
    symbol -> {macro, sector, industry, basic_industry} via NSE's quote endpoint.

    ~2000 symbols at ~0.35s each is roughly 12 minutes, but it is a one-time
    cost: results are written to cache after every 25 lookups, so an interrupted
    run resumes where it stopped.

    `basic_industry` is the level that matches a granular sector screener
    ("Paper & Paper Products", "Commodity Chemicals", and so on).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "sector_map.json"
    known: dict[str, dict] = {}
    if path.exists():
        known = json.loads(path.read_text())

    sess = sess or NSESession()
    if symbols is None:
        symbols = fetch_equity_list(sess)["symbol"].tolist()

    # Re-fetch blanks. A completed run against the dead quote-equity
    # endpoint cached every symbol as all-null; those must not count as done.
    already = sum(1 for s in symbols if _has_industry(known.get(s)))
    todo = [s for s in symbols if not _has_industry(known.get(s))]
    if verbose and todo:
        print(f"Resolving industry for {len(todo):,} symbols "
              f"({already:,} already cached)…")

    n_fail = 0
    for i, sym in enumerate(todo, 1):
        try:
            qsym = requests.utils.quote(sym, safe="")
            r = sess.get(
                f"{QUOTE_NEXT}?functionName=getSymbolData"
                f"&marketType=N&series=EQ&symbol={qsym}",
                referer=f"{BASE}/get-quotes/equity?symbol={qsym}",
            )
            eq = (r.json().get("equityResponse") or [None])[0] or {}
            sec = eq.get("secInfo") or {}
            rec = {
                "macro": _nz(sec.get("macro")),
                "sector": _nz(sec.get("sector")),
                "industry": _nz(sec.get("industryInfo")),
                "basic_industry": _nz(sec.get("basicIndustry")),
            }
            if not _has_industry(rec):
                n_fail += 1
            known[sym] = rec
        except Exception:
            n_fail += 1
            known[sym] = {"macro": None, "sector": None,
                          "industry": None, "basic_industry": None}
        if i % 25 == 0:
            path.write_text(json.dumps(known))
            if verbose:
                filled = sum(1 for v in known.values() if _has_industry(v))
                print(f"  {i}/{len(todo)}  ({filled:,} with industry)")
        time.sleep(pause)

    path.write_text(json.dumps(known))
    filled = sum(1 for v in known.values() if _has_industry(v))
    if verbose:
        print(f"\nSector map: {filled:,}/{len(known):,} symbols have a "
              f"basic industry ({n_fail:,} misses this pass).")
        if filled == 0:
            print("That is not usable — every stock will be dropped at "
                  "clean(). Check connectivity / NSE cookies and re-run.")
    return (pd.DataFrame.from_dict(known, orient="index")
              .rename_axis("symbol").reset_index())


# --------------------------------------------------------------------------
# Live snapshot (last prices at Refresh time)
# --------------------------------------------------------------------------

LIVE_INDICES = [
    "NIFTY TOTAL MARKET",
    "NIFTY 500",
    "NIFTY MIDSMALLCAP 400",
    "NIFTY SMALLCAP 250",
    "NIFTY MICROCAP 250",
    "SECURITIES IN F&O",
]
LIVE_INDEX_URL = BASE + "/api/equity-stockIndices?index={idx}"
LIVE_MARKET = BASE + "/market-data/live-equity-market"


def _index_rows(payload: dict) -> list[dict]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    rows = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        sym = str(rec.get("symbol") or "").strip()
        if not sym or sym.upper().startswith("NIFTY") or sym.upper() == "INDIA VIX":
            continue
        ltp = rec.get("lastPrice")
        if ltp is None:
            continue
        rows.append({
            "symbol": sym,
            "ltp": float(ltp),
            "open": rec.get("open"),
            "high": rec.get("dayHigh") or rec.get("high"),
            "low": rec.get("dayLow") or rec.get("low"),
            "prev_close": rec.get("previousClose"),
            "volume": rec.get("totalTradedVolume"),
            "turnover": rec.get("totalTradedValue"),
            "pchange": rec.get("pChange"),
            "time": rec.get("lastUpdateTime") or payload.get("timestamp"),
        })
    return rows


def live_snapshot(sess: NSESession | None = None,
                  indices: list[str] | None = None) -> pd.DataFrame:
    """
    Current last prices for as much of the cash market as NSE will give
    in a few index dumps. Taken at Refresh time, not a replacement for
    the EOD bhavcopy.
    """
    sess = sess or NSESession()
    try:
        sess.get(LIVE_MARKET, referer=BASE)
    except Exception:
        pass
    chunks = []
    for name in (indices or LIVE_INDICES):
        url = LIVE_INDEX_URL.format(idx=requests.utils.quote(name, safe=""))
        try:
            payload = sess.get_json(url, referer=LIVE_MARKET)
            part = _index_rows(payload)
            if part:
                chunks.append(pd.DataFrame(part))
        except Exception as exc:
            print(f"live_snapshot {name}: {exc}")
            continue
        time.sleep(sess.pause)
    if not chunks:
        return pd.DataFrame(columns=[
            "symbol", "ltp", "open", "high", "low", "prev_close",
            "volume", "turnover", "pchange", "time",
        ])
    out = pd.concat(chunks, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out = out.drop_duplicates("symbol", keep="last")
    for c in ("ltp", "open", "high", "low", "prev_close", "volume", "turnover", "pchange"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.reset_index(drop=True)


QUOTE_URL = BASE + "/api/quote-equity?symbol={sym}"


def _quote_row(payload: dict, symbol: str) -> dict | None:
    """Parse /api/quote-equity into the same shape as live_snapshot rows."""
    if not isinstance(payload, dict):
        return None
    info = payload.get("priceInfo") or {}
    ltp = info.get("lastPrice")
    if ltp is None:
        return None
    band = info.get("intraDayHighLow") or {}
    return {
        "symbol": str(symbol).strip().upper(),
        "ltp": float(ltp),
        "open": info.get("open"),
        "high": band.get("max"),
        "low": band.get("min"),
        "prev_close": info.get("previousClose"),
        "volume": (payload.get("securityWiseDP") or {}).get("quantityTraded"),
        "turnover": None,
        "pchange": info.get("pChange"),
        "time": payload.get("metadata", {}).get("lastUpdateTime"),
    }


GROWW_QUOTE_WORKERS = 6

FULL_QUOTE_COLUMNS = [
    "symbol", "ltp", "open", "high", "low", "prev_close",
    "volume", "avg_price", "turnover", "pchange", "time",
]


def _quote_one(groww, symbol: str) -> dict | None:
    """
    Per-symbol quote. Slower than the batch endpoints but the only place
    Groww exposes traded volume and the day's average price, which is what
    the sector panel needs to rebuild turnover.
    """
    q = groww.get_quote(
        exchange=groww.EXCHANGE_NSE, segment=groww.SEGMENT_CASH,
        trading_symbol=symbol,
    )
    if not isinstance(q, dict):
        return None
    ltp = q.get("last_price")
    if ltp is None:
        return None
    bar = q.get("ohlc") or {}
    volume = q.get("volume")
    avg = q.get("average_price")
    # Bhavcopy turnover ships in lakhs; match it so the panel's 9-day
    # turnover baseline stays on one scale. If average_price (VWAP) is
    # missing, approximate with the last traded price.
    turnover = None
    price_for_turnover = avg or ltp
    if volume and price_for_turnover:
        turnover = float(volume) * float(price_for_turnover) / 1e5
    return {
        "symbol": symbol,
        "ltp": ltp,
        "open": bar.get("open"),
        "high": bar.get("high"),
        "low": bar.get("low"),
        "prev_close": bar.get("close"),
        "volume": volume,
        "avg_price": avg,
        "turnover": turnover,
        "pchange": q.get("day_change_perc"),
        "time": q.get("last_trade_time"),
    }


def live_quotes_full(symbols: list[str], on_progress=None, on_batch=None,
                     workers: int = GROWW_QUOTE_WORKERS) -> pd.DataFrame:
    """
    Live quotes WITH volume and turnover, for the whole universe.

    get_quote is one call per symbol, so this is threaded. Roughly a minute
    for ~1,700 names. Use this when the sector panel has to be rebuilt;
    `live_quotes_groww` is the cheap price-only path.

    on_progress(done, total) - called every 50 symbols
    on_batch(partial_df) - called every 50 symbols with current data for preview
    """
    empty = pd.DataFrame(columns=FULL_QUOTE_COLUMNS)
    want = [str(s).strip().upper() for s in symbols if s]
    want = list(dict.fromkeys(want))
    if not want:
        return empty

    groww = _get_groww_client()
    rows: list[dict] = []
    done = 0
    lock = threading.Lock()

    def fetch_one(sym: str):
        for attempt in range(GROWW_RETRIES):
            try:
                return _quote_one(groww, sym)
            except Exception:
                if attempt == GROWW_RETRIES - 1:
                    return None
                time.sleep(GROWW_PAUSE * (attempt + 1))
        return None

    def _make_partial_df():
        if not rows:
            return empty
        df = pd.DataFrame(rows)
        df["symbol"] = df["symbol"].astype(str).str.strip()
        return df.drop_duplicates("symbol", keep="last")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(fetch_one, want):
            with lock:
                done += 1
                if row:
                    rows.append(row)
                if done % 50 == 0:
                    if on_progress is not None:
                        on_progress(done, len(want))
                    if on_batch is not None:
                        on_batch(_make_partial_df())

    if on_progress is not None:
        on_progress(len(want), len(want))
    if on_batch is not None:
        on_batch(_make_partial_df())
    missing = len(want) - len(rows)
    if missing:
        print(f"groww full quotes: {missing}/{len(want)} symbols returned nothing")
    if not rows:
        return empty

    out = pd.DataFrame(rows)
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out = out.drop_duplicates("symbol", keep="last")
    for c in ("ltp", "open", "high", "low", "prev_close",
              "volume", "avg_price", "turnover", "pchange"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.reset_index(drop=True)


def live_quotes(symbols: list[str], sess: NSESession | None = None) -> pd.DataFrame:
    """
    Live prices for a small named list using NSE quote-equity endpoint.
    Falls back to Groww API if NSE fails.
    """
    # Try Groww first (more reliable)
    try:
        df = live_quotes_groww(symbols)
        if not df.empty:
            return df
    except Exception as exc:
        print(f"Groww API failed, trying NSE: {exc}")

    # Fallback to NSE
    empty = pd.DataFrame(columns=[
        "symbol", "ltp", "open", "high", "low", "prev_close",
        "volume", "turnover", "pchange", "time",
    ])
    want = [str(s).strip().upper() for s in symbols if s]
    want = list(dict.fromkeys(want))
    if not want:
        return empty

    sess = sess or NSESession()
    have: dict[str, dict] = {}

    for sym in want:
        url = QUOTE_URL.format(sym=requests.utils.quote(sym, safe=""))
        try:
            payload = sess.get_json(url, referer=BASE)
            row = _quote_row(payload, sym)
            if row:
                have[sym] = row
        except Exception as exc:
            print(f"nse quote {sym}: {exc}")
        time.sleep(sess.pause)

    if not have:
        return empty
    out = pd.DataFrame(list(have.values()))
    for c in ("ltp", "open", "high", "low", "prev_close", "volume", "turnover", "pchange"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Groww API for live data
# --------------------------------------------------------------------------

GROWW_BATCH = 50          # hard limit on symbols per get_ltp / get_ohlc call
GROWW_PAUSE = 0.25
GROWW_RETRIES = 3
GROWW_TOKEN_TTL = 6 * 3600

_groww_client = None
_groww_issued_at = 0.0


def _get_groww_client(force: bool = False):
    """
    Authenticated Groww client, reused across calls.

    Each login is a TOTP round trip, so a scan that walks 30 batches must not
    re-authenticate per batch. Pass force=True to re-issue after a call fails
    on a stale token.
    """
    global _groww_client, _groww_issued_at
    if (not force and _groww_client is not None
            and time.time() - _groww_issued_at < GROWW_TOKEN_TTL):
        return _groww_client

    try:
        from growwapi import GrowwAPI
        import pyotp
    except ImportError:
        raise ImportError("Install growwapi and pyotp: pip install growwapi pyotp")

    totp_token = os.environ.get("GROWW_TOTP_TOKEN")
    totp_secret = os.environ.get("GROWW_TOTP_SECRET")

    if not totp_token or not totp_secret:
        raise ValueError("GROWW_TOTP_TOKEN and GROWW_TOTP_SECRET must be set in .env")

    totp = pyotp.TOTP(totp_secret).now()
    access_token = GrowwAPI.get_access_token(api_key=totp_token, totp=totp)
    _groww_client = GrowwAPI(access_token)
    _groww_issued_at = time.time()
    return _groww_client


def _groww_batch(groww, batch: tuple[str, ...]) -> list[dict]:
    """
    One batch of at most GROWW_BATCH symbols.

    get_ltp carries the traded price. get_ohlc's `close` is the PREVIOUS
    close, not the running price — reading it as the last price backdates
    every trigger comparison by a day. Take the price from get_ltp and use
    the OHLC close only as prev_close, which is what gives us pchange.
    """
    ltp = groww.get_ltp(
        segment=groww.SEGMENT_CASH, exchange_trading_symbols=batch,
    ) or {}
    try:
        bars = groww.get_ohlc(
            segment=groww.SEGMENT_CASH, exchange_trading_symbols=batch,
        ) or {}
    except Exception as exc:
        print(f"groww get_ohlc: {exc}")
        bars = {}

    rows = []
    for key, price in ltp.items():
        if price is None:
            continue
        bar = bars.get(key) or {}
        prev = bar.get("close")
        try:
            pchange = (float(price) / float(prev) - 1.0) * 100 if prev else None
        except (TypeError, ValueError, ZeroDivisionError):
            pchange = None
        rows.append({
            "symbol": str(key).replace("NSE_", ""),
            "ltp": price,
            "open": bar.get("open"),
            "high": bar.get("high"),
            "low": bar.get("low"),
            "prev_close": prev,
            # Groww exposes traded volume only on the per-symbol quote
            # endpoint, which is one call per name. Left unknown.
            "volume": None,
            "turnover": None,
            "pchange": pchange,
            "time": None,
        })
    return rows


def live_quotes_groww(symbols: list[str], on_progress=None) -> pd.DataFrame:
    """
    Live prices for a named list via the Groww Trading API.

    Batched at GROWW_BATCH per call. A batch that fails every retry is
    reported rather than silently dropping those symbols from the scan.
    `on_progress(done, total)` is optional — used by the UI for the message.
    """
    empty = pd.DataFrame(columns=[
        "symbol", "ltp", "open", "high", "low", "prev_close",
        "volume", "turnover", "pchange", "time",
    ])

    want = [str(s).strip().upper() for s in symbols if s]
    want = list(dict.fromkeys(want))
    if not want:
        return empty

    groww = _get_groww_client()
    exchange_symbols = [f"NSE_{sym}" for sym in want]

    rows: list[dict] = []
    dropped = 0
    for i in range(0, len(exchange_symbols), GROWW_BATCH):
        batch = tuple(exchange_symbols[i:i + GROWW_BATCH])
        for attempt in range(GROWW_RETRIES):
            try:
                rows.extend(_groww_batch(groww, batch))
                break
            except Exception as exc:
                if attempt == GROWW_RETRIES - 1:
                    dropped += len(batch)
                    print(f"groww batch {i // GROWW_BATCH}: {exc}")
                    break
                # An expired token is the usual cause; re-issue once.
                if attempt == 0:
                    try:
                        groww = _get_groww_client(force=True)
                    except Exception:
                        pass
                time.sleep(GROWW_PAUSE * (attempt + 1) * 2)
        if on_progress is not None:
            on_progress(min(i + GROWW_BATCH, len(want)), len(want))
        time.sleep(GROWW_PAUSE)

    if dropped:
        print(f"groww: {dropped}/{len(want)} symbols dropped by failed batches")
    if not rows:
        return empty

    out = pd.DataFrame(rows)
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out = out.drop_duplicates("symbol", keep="last")
    for c in ("ltp", "open", "high", "low", "prev_close", "pchange"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.reset_index(drop=True)
