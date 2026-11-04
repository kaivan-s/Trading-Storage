"""
Results already declared, and how each stock traded on and after them.

Declarations come from NSE corporate announcements: the first results filing
("Outcome of Board Meeting", results press release, integrated financial
filing...) per company. Later copies of the same results — newspaper notices,
clarifications, corrigenda — are skipped.

The reaction session is the first session that traded with the results known:
  before 09:15            that day's session
  09:15-14:30             that same session (flagged "during market")
  after 14:30 / holiday   the next session
Prices are the daily bhavcopy history (engine.raw); a reaction session that
isn't in the history yet (today, during market hours) is priced from Groww.

Revenue / profit numbers (filing_facts.results_numbers, XBRL) are slow — a few
NSE calls per company — so a background worker fills them in, largest
companies first, and keeps them on disk.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

import fetch
import filing_facts
import movers
import news

IST = timezone(timedelta(hours=5, minutes=30))
SAME_SESSION_CUTOFF = "14:30"
SKIP_DESC = ("clarification", "delayed", "non-submission", "corrigendum", "newspaper")
NEW_EVENT_GAP_DAYS = 25           # two results filings further apart are two quarters
CACHE_TTL = 10 * 60
LIVE_TTL = 60
NUMBERS_FILE = fetch.CACHE_DIR / "results_numbers.json"

_lock = threading.Lock()
_ann_cache: dict = {"key": None, "at": 0.0, "events": None}
_live_cache: dict = {"at": 0.0, "quotes": {}}
_numbers: dict[str, dict | None] | None = None
_numbers_lock = threading.Lock()
_worker: dict = {"thread": None, "queue": []}


def _now() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)


# --------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------

def _announcements(frm: datetime, to: datetime) -> list[dict]:
    sess = fetch.NSESession(timeout=60, retries=2)
    url = movers.ANNOUNCEMENTS_ALL.format(frm=frm.strftime("%d-%m-%Y"), to=to.strftime("%d-%m-%Y"))
    rows = sess.get_json(url, referer=news.REFERER)
    return (rows.get("data") or []) if isinstance(rows, dict) else (rows or [])


def declarations(days_back: int) -> list[dict]:
    """One row per company per results event, earliest filing first."""
    now = _now()
    key = (days_back, now.date())
    with _lock:
        if _ann_cache["key"] == key and time.time() - _ann_cache["at"] < CACHE_TTL:
            return _ann_cache["events"]
    raw = _announcements(now - timedelta(days=days_back), now)
    hits = []
    for r in raw:
        desc = (r.get("desc") or "").lower()
        if any(s in desc for s in SKIP_DESC):
            continue
        shaped = news._shape(r)
        if not shaped or shaped["category"] != "Results":
            continue
        when = news._parse_dt(r.get("an_dt")) or news._parse_dt(r.get("sort_date"))
        sym = (r.get("symbol") or "").strip().upper()
        if when and sym:
            hits.append({"symbol": sym, "when": when, "title": r.get("desc") or "Results",
                         "url": r.get("attchmntFile") or None, "name": r.get("sm_name")})
    hits.sort(key=lambda h: (h["symbol"], h["when"]))
    events, last = [], {}
    for h in hits:
        prev = last.get(h["symbol"])
        if prev and (h["when"] - prev).days < NEW_EVENT_GAP_DAYS:
            continue
        last[h["symbol"]] = h["when"]
        events.append(h)
    with _lock:
        _ann_cache.update(key=key, at=time.time(), events=events)
    return events


# --------------------------------------------------------------------------
# Price reaction
# --------------------------------------------------------------------------

def _timing(when: datetime) -> str:
    hm = when.strftime("%H:%M")
    if hm < "09:15":
        return "before_open"
    if hm < "15:30":
        return "during_market"
    return "after_close"


def _reaction_date(when: datetime, sessions: list[pd.Timestamp], today_open: bool) -> pd.Timestamp | None:
    """The session that first traded with the results known (None = not yet)."""
    day = pd.Timestamp(when.date())
    same_day_ok = when.strftime("%H:%M") < SAME_SESSION_CUTOFF
    if same_day_ok and day in set(sessions):
        return day
    nxt = [s for s in sessions if s > day]
    if nxt:
        return nxt[0]
    # Past the price history: only today's live session can stand in, and only
    # when the history stops at the previous session (a weekend at most).
    today, last = pd.Timestamp(_now().date()), sessions[-1]
    if not today_open or (today - last).days > 4:
        return None
    if day < today or same_day_ok:
        return today
    return None


def _market_open_now() -> bool:
    n = _now()
    return n.weekday() < 5 and "09:15" <= n.strftime("%H:%M")


def _live(symbols: list[str]) -> dict[str, dict]:
    if not symbols:
        return {}
    if time.time() - _live_cache["at"] < LIVE_TTL and all(s in _live_cache["quotes"] for s in symbols):
        return _live_cache["quotes"]
    try:
        q = fetch.live_quotes_groww(symbols)
        quotes = {r.symbol: {"ltp": r.ltp, "open": r.open, "prev_close": r.prev_close,
                             "volume": pd.to_numeric(r.volume, errors="coerce")}
                  for r in q.itertuples() if pd.notna(r.ltp)}
    except Exception as exc:
        print(f"results_reactions live: {exc}")
        quotes = {}
    _live_cache.update(at=time.time(), quotes={**_live_cache["quotes"], **quotes})
    return _live_cache["quotes"]


def _pct(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return None
    return round((float(a) / float(b) - 1) * 100, 2)


def reactions(raw: pd.DataFrame, events: list[dict]) -> list[dict]:
    eq = raw[raw["series"].astype(str).str.upper() == "EQ"]
    eq = eq[eq["close"] > 0]
    sessions = sorted(pd.to_datetime(eq["date"].unique()))
    last_session = sessions[-1]
    mkt = (eq.assign(r=eq["close"] / eq["prev_close"] - 1)
             .groupby("date")["r"].median())
    want = {e["symbol"] for e in events}
    mine = eq[eq["symbol"].isin(want)].sort_values(["symbol", "date"])
    by_sym = {s: g.reset_index(drop=True) for s, g in mine.groupby("symbol", sort=False)}
    today_open = _market_open_now()

    rows, need_live = [], []
    for e in events:
        g = by_sym.get(e["symbol"])
        if g is None:
            continue
        rd = _reaction_date(e["when"], sessions, today_open)
        out = {
            "symbol": e["symbol"],
            "earnings_date": e["when"].strftime("%Y-%m-%d"),
            "declared_at": e["when"].strftime("%Y-%m-%d %H:%M"),
            "timing": _timing(e["when"]),
            "filing_title": e["title"],
            "filing_url": e["url"],
            "declared": True,
            "reaction_date": None if rd is None else rd.strftime("%Y-%m-%d"),
            "status": "pending" if rd is None else "done",
            "name": e["name"],
        }
        # The opening gap only reflects the results if they were out before that open.
        gap_ok = rd is not None and (rd.date() != e["when"].date() or out["timing"] == "before_open")
        if rd is not None and rd > last_session:
            out["status"] = "live"
            need_live.append(e["symbol"])
            out["_prev_close"] = float(g["close"].iloc[-1])
            out["_gap_ok"] = gap_ok
        elif rd is not None:
            hit = g.index[pd.to_datetime(g["date"]) == rd]
            if len(hit) == 0:
                continue
            i = int(hit[0])
            r = g.iloc[i]
            vol_med = g["volume"].iloc[max(0, i - 20):i].median()
            day = _pct(r.close, r.prev_close)
            m = mkt.get(r.date)
            out.update({
                "prev_close": round(float(r.prev_close), 2),
                "react_open": round(float(r.open), 2),
                "react_close": round(float(r.close), 2),
                "gap_pct": _pct(r.open, r.prev_close) if gap_ok else None,
                "day_pct": day,
                "vs_market_pct": None if day is None or m is None or pd.isna(m) else round(day - m * 100, 2),
                "vol_x": None if not vol_med or pd.isna(vol_med) else round(float(r.volume) / float(vol_med), 1),
                "next_pct": _pct(g["close"].iloc[i + 1], r.close) if i + 1 < len(g) else None,
                "after_pct": _pct(g["close"].iloc[-1], r.close) if i + 1 < len(g) else None,
                "after_sessions": int(len(g) - 1 - i),
                "total_pct": _pct(g["close"].iloc[-1], r.prev_close),
                "last_close": round(float(g["close"].iloc[-1]), 2),
                "last_date": pd.Timestamp(g["date"].iloc[-1]).strftime("%Y-%m-%d"),
            })
        rows.append(out)

    if need_live:
        quotes = _live(need_live)
        for out in rows:
            if out["status"] != "live":
                continue
            q = quotes.get(out["symbol"])
            prev = out.pop("_prev_close", None)
            gap_ok = out.pop("_gap_ok", False)
            if not q:
                out["status"] = "pending"
                continue
            pc = q.get("prev_close") or prev
            out.update({
                "prev_close": round(float(pc), 2),
                "react_open": None if pd.isna(q.get("open")) else round(float(q["open"]), 2),
                "react_close": round(float(q["ltp"]), 2),
                "gap_pct": _pct(q.get("open"), pc) if gap_ok else None,
                "day_pct": _pct(q["ltp"], pc),
                "after_sessions": 0,
            })
    for out in rows:
        out.pop("_prev_close", None)
        out.pop("_gap_ok", None)
    return rows


_rows_cache: dict = {"key": None, "at": 0.0, "rows": None}


def declared_rows(raw: pd.DataFrame | None, days_back: int) -> list[dict]:
    """Declared results with reactions; cached (shorter while the market is open)."""
    if raw is None or raw.empty:
        return []
    key = (days_back, len(raw), str(raw["date"].max()))
    ttl = LIVE_TTL if _market_open_now() else CACHE_TTL
    if _rows_cache["key"] == key and time.time() - _rows_cache["at"] < ttl:
        return [dict(r) for r in _rows_cache["rows"]]
    try:
        rows = reactions(raw, declarations(days_back))
    except Exception as exc:
        print(f"results_reactions: {exc}")
        return [dict(r) for r in (_rows_cache["rows"] or [])]
    _rows_cache.update(key=key, at=time.time(), rows=rows)
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Revenue / profit numbers (background)
# --------------------------------------------------------------------------

def _load_numbers() -> dict:
    global _numbers
    with _numbers_lock:
        if _numbers is None:
            try:
                _numbers = json.loads(NUMBERS_FILE.read_text())
            except Exception:
                _numbers = {}
        return _numbers


def _save_numbers() -> None:
    with _numbers_lock:
        tmp = NUMBERS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_numbers))
        tmp.replace(NUMBERS_FILE)


RETRY_AFTER = 6 * 3600


def _expected_quarter_end(day: str) -> pd.Timestamp:
    """Results declared on `day` are for the last quarter that ended before it."""
    return pd.Timestamp(day).to_period("Q").start_time - pd.Timedelta(days=1)


def _due(nums: dict, key: str) -> bool:
    """Never looked up, or looked up too early and the retry time has come."""
    if key not in nums:
        return True
    e = nums[key]
    return isinstance(e, dict) and e.get("retry_at", float("inf")) <= time.time()


def _numbers_worker() -> None:
    nums = _load_numbers()
    done = 0
    while _worker["queue"]:
        sym, day = _worker["queue"].pop(0)
        key = f"{sym}|{day}"
        if not _due(nums, key):
            continue
        try:
            r = filing_facts.results_numbers(sym, around=day)
        except Exception as exc:
            print(f"results numbers {sym}: {exc}")
            r = None
        # NSE's XBRL copy can lag the announcement; an older quarter is not this result.
        if r and abs((pd.Timestamp(r["quarter_end"]) - _expected_quarter_end(day)).days) > 5:
            r = None
        if r is None:
            r = {"retry_at": time.time() + RETRY_AFTER}
        with _numbers_lock:
            nums[key] = r
        done += 1
        if done % 10 == 0:
            _save_numbers()
    _save_numbers()
    _worker["thread"] = None


def _queue_numbers(rows: list[dict]) -> None:
    nums = _load_numbers()
    todo = [(r["symbol"], r["earnings_date"]) for r in
            sorted(rows, key=lambda r: -(r.get("market_cap_cr") or 0))
            if _due(nums, f"{r['symbol']}|{r['earnings_date']}")]
    if not todo:
        return
    queued = set(_worker["queue"])
    _worker["queue"].extend(t for t in todo if t not in queued)
    if _worker["thread"] is None:
        t = threading.Thread(target=_numbers_worker, daemon=True)
        _worker["thread"] = t
        t.start()


def attach_numbers(rows: list[dict]) -> None:
    nums = _load_numbers()
    for r in rows:
        n = nums.get(f"{r['symbol']}|{r['earnings_date']}")
        if n and "retry_at" not in n:
            r["results"] = {k: n.get(k) for k in ("quarter", "basis", "revenue_cr", "profit_cr",
                                                   "compare", "revenue_chg", "profit_chg", "profit_swing")}
    _queue_numbers(rows)
