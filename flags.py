"""
Evidence-backed risk flags for the stock page, the watchlist and the Results
page. Both describe how similar stocks behaved afterwards; neither is a call.

weak_results  The results-day move was 4%+ worse than the market's median
              stock. Non-F&O only, price >= 50, 20-day median turnover >= 5 cr.
              Active for the 10 sessions after the reaction session.
              Evidence: attic/falling_patterns.py 1. Non-F&O stocks lagged the
              average stock by 1.6% over 5 sessions and 2.1% over 10, about 65%
              of the time (Apr 2024 - Oct 2026); F&O stocks showed no drift.

weak_bounce   Downtrend (close and EMA50 below EMA200), RSI(14) turns back
              below 60 on 1.5x+ the usual volume, after a resistance rejection
              or a touch of EMA50 within the last 3 sessions. Price >= 50,
              20-day median turnover >= 5 cr. Active for 10 sessions.
              Evidence: attic/rsi_ema_sr.py. Lagged the average stock by
              0.6-1.4% over 5-20 sessions, right 58-61% of the time, on F&O
              stocks (2009-26) and cash stocks (2024-26).
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

ACTIVE_SESSIONS = 10
MIN_PRICE = 50
MIN_TURNOVER_CR = 5

WEAK_RESULTS_PCT = -4.0
RESULTS_LOOKBACK_DAYS = 25

K = 5                 # swing-high half-width, confirmed K sessions later
MAX_AGE = 250
TOUCH = 0.005
BREAK = 0.01
RSI_LEVEL = 60
VOL_X = 1.5
CORP_ACTION = 0.35    # a daily move beyond this is a split/bonus, not trading

NOTES = {
    "weak_results": (
        "Historically, non-F&O stocks with a results-day reaction like this kept lagging the market "
        "for about 2 weeks (about 2 in 3 cases, Apr 2024 - Oct 2026). Describes past behaviour; "
        "not a recommendation."
    ),
    "weak_bounce": (
        "Historically, stocks with this pattern lagged the market by about 1% over the next 2 weeks "
        "(about 6 in 10 cases, F&O stocks 2009-26 and cash stocks 2024-26). It points to relative "
        "weakness, not necessarily a fall. Describes past behaviour; not a recommendation."
    ),
}

_lock = threading.Lock()
_cache: dict = {"key": None, "bounce": {}}


def _day(iso) -> str:
    return pd.Timestamp(iso).strftime("%-d %b")


def _key(raw: pd.DataFrame) -> tuple:
    return (len(raw), str(raw["date"].max()))


def _eq(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[raw["series"].astype(str).str.upper() == "EQ"]
    d = d[(d["close"] > 0) & (d["prev_close"] > 0)]
    d = d.assign(date=pd.to_datetime(d["date"]))
    return d.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])


def _liquid(g: pd.DataFrame) -> bool:
    if len(g) < 20:
        return False
    to = pd.to_numeric(g["turnover"], errors="coerce").iloc[-20:].median() / 100   # lakhs -> cr
    return bool(g["close"].iloc[-1] >= MIN_PRICE and to >= MIN_TURNOVER_CR)


# --------------------------------------------------------------------------
# weak_bounce
# --------------------------------------------------------------------------

def _rsi(c: pd.Series) -> np.ndarray:
    diff = c.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-diff).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()


def _rejections(h: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Days that tested an unbroken prior swing high and closed back below it."""
    n = len(c)
    piv = np.zeros(n, bool)
    for i in range(K, n - K):
        if h[i] == h[i - K:i + K + 1].max():
            piv[i] = True
    out = np.zeros(n, bool)
    levels: list[list] = []
    for t in range(n):
        i = t - K
        if i >= K and piv[i]:
            levels.append([h[i], i])
        keep = []
        for lvl, formed in levels:
            if t - formed > MAX_AGE:
                continue
            if c[t] > lvl * (1 + BREAK):
                continue
            if (h[t] >= lvl * (1 - TOUCH) and c[t] < lvl and t - formed > K + 5
                    and t > 0 and c[t - 1] < lvl * (1 - BREAK)):
                out[t] = True
            keep.append([lvl, formed])
        levels = keep
    return out


def _bounce_one(g: pd.DataFrame) -> dict | None:
    if len(g) < 220 or not _liquid(g):
        return None
    # Split-safe close index. The bhavcopy prev_close is not always adjusted on
    # the ex-date, so a move beyond any circuit band uses that day's open instead.
    ratio = g["close"] / g["prev_close"]
    bad = (ratio < 1 - CORP_ACTION) | (ratio > 1 + CORP_ACTION)
    a = ratio.where(~bad, g["close"] / g["open"].where(g["open"] > 0, g["close"])).cumprod()
    k = a / g["close"]
    c, h = a, g["high"] * k
    e20, e50, e200 = (c.ewm(span=n, adjust=False, min_periods=n).mean() for n in (20, 50, 200))
    rsi = _rsi(c)
    vol = pd.to_numeric(g["volume"], errors="coerce")
    vol_x = (vol / vol.shift(1).rolling(20, min_periods=10).median()).to_numpy()
    c_, h_, e50_, e200_ = c.to_numpy(), h.to_numpy(), e50.to_numpy(), e200.to_numpy()
    n = len(c_)
    recent = range(max(1, n - ACTIVE_SESSIONS), n)
    cheap = [t for t in recent
             if c_[t] < e200_[t] and e50_[t] < e200_[t] and rsi[t] < RSI_LEVEL <= rsi[t - 1]
             and vol_x[t] >= VOL_X]
    if not cheap:
        return None
    rej = _rejections(h_, c_)
    touch = (h_ >= e50_ * 0.99) & (c_ < e50_)
    hit = [t for t in cheap if (rej[max(0, t - 2):t + 1] | touch[max(0, t - 2):t + 1]).any()]
    if not hit:
        return None
    t = hit[-1]
    since = n - 1 - t
    day = pd.Timestamp(g["date"].iloc[t]).strftime("%Y-%m-%d")
    return {
        "kind": "weak_bounce",
        "label": "Weak bounce, heavy selling",
        "since": day,
        "sessions_left": ACTIVE_SESSIONS - since,
        "vol_x": round(float(vol_x[t]), 1),
        "detail": (f"In a downtrend, RSI turned back below {RSI_LEVEL} on {vol_x[t]:.1f}x its usual volume "
                   f"near resistance on {_day(day)}."),
        "note": NOTES["weak_bounce"],
    }


def _bounce_all(raw: pd.DataFrame) -> dict[str, dict]:
    key = _key(raw)
    with _lock:
        if _cache["key"] == key:
            return _cache["bounce"]
    out = {}
    for sym, g in _eq(raw).groupby("symbol", sort=False):
        try:
            f = _bounce_one(g)
        except Exception as exc:
            print(f"flags weak_bounce {sym}: {exc}")
            f = None
        if f:
            out[sym] = f
    with _lock:
        _cache.update(key=key, bounce=out)
    return out


# --------------------------------------------------------------------------
# weak_results
# --------------------------------------------------------------------------

def _fno() -> set[str]:
    import gapfade
    return gapfade._fno()


def weak_results_flag(row: dict, fno: set[str], liquid: set[str], sessions: list[str]) -> dict | None:
    """The flag for one declared-results row from results_reactions, or None."""
    vs = row.get("vs_market_pct")
    rd = row.get("reaction_date")
    if (row.get("status") != "done" or vs is None or vs > WEAK_RESULTS_PCT or not rd
            or vs < -CORP_ACTION * 100
            or row["symbol"] in fno or row["symbol"] not in liquid):
        return None
    # Market sessions, not the stock's own rows: a halted stock must not look fresh.
    after = sum(1 for s in sessions if s > rd)
    if after >= ACTIVE_SESSIONS:
        return None
    return {
        "kind": "weak_results",
        "label": "Weak results reaction",
        "since": row.get("reaction_date"),
        "sessions_left": ACTIVE_SESSIONS - int(after),
        "vs_market_pct": vs,
        "detail": f"After results, it fell {abs(vs):.1f}% more than the market on {_day(rd)}.",
        "note": NOTES["weak_results"],
    }


def _liquid_set(d: pd.DataFrame, symbols: set[str]) -> set[str]:
    last = d["date"].max()
    d = d[d["symbol"].isin(symbols)]
    return {s for s, g in d.groupby("symbol", sort=False)
            if g["date"].iloc[-1] == last and _liquid(g)}


def mark_results_rows(rows: list[dict], raw: pd.DataFrame | None) -> None:
    """Attach `weak_reaction` to declared-results rows that carry the flag."""
    if raw is None or raw.empty or not rows:
        return
    cands = {r["symbol"] for r in rows if (r.get("vs_market_pct") or 0) <= WEAK_RESULTS_PCT}
    if not cands:
        return
    d = _eq(raw)
    sessions = sorted({t.strftime("%Y-%m-%d") for t in d["date"].unique().tolist()})
    fno, liquid = _fno(), _liquid_set(d, cands)
    for r in rows:
        f = weak_results_flag(r, fno, liquid, sessions)
        if f:
            r["weak_reaction"] = f


def _weak_results_all(raw: pd.DataFrame) -> dict[str, dict]:
    import results_reactions
    rows = results_reactions.declared_rows(raw, RESULTS_LOOKBACK_DAYS)
    mark_results_rows(rows, raw)
    out: dict[str, dict] = {}
    for r in rows:
        f = r.get("weak_reaction")
        if f and (r["symbol"] not in out or f["since"] > out[r["symbol"]]["since"]):
            out[r["symbol"]] = f
    return out


# --------------------------------------------------------------------------
# Public
# --------------------------------------------------------------------------

def active(raw: pd.DataFrame | None) -> dict[str, list[dict]]:
    """Every symbol with an active flag -> its flags."""
    if raw is None or raw.empty:
        return {}
    out: dict[str, list[dict]] = {}
    for src in (_weak_results_all, _bounce_all):
        try:
            for sym, f in src(raw).items():
                out.setdefault(sym, []).append(f)
        except Exception as exc:
            print(f"flags {src.__name__}: {exc}")
    return out


def for_symbol(symbol: str, raw: pd.DataFrame | None) -> list[dict]:
    return active(raw).get(symbol.upper(), [])
