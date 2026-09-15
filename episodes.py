#!/usr/bin/env python3
"""
Base episodes: what became of a coiled base after it showed up on the list.

The lists answer "what looks ready tonight". A daily snapshot has no memory,
so it cannot answer the question a reader actually asks a week later — the
name I saw is gone, did it break out or fall apart? This gives a base a
persistent identity from the session it first qualifies to the session it
resolves, and a plain reason for every transition.

Two measured facts shape the design (numbers from eval_lifecycle.py):

  - 49% of drop-offs are back on the list within 10 sessions, 28% within 3.
    Treating each drop-off as an ending would declare a base dead and then
    brand-new a few sessions later. Gaps up to BRIDGE sessions are held open
    instead, which folds roughly a quarter of apparent episodes back into
    their true parent.
  - Half of all episodes resolve within 20 sessions — median 6 to trigger, 9
    to break down — so the window worth retaining is short.

Two details are load-bearing and easy to get wrong:

  - The trigger is frozen at entry. A rolling 20-day high climbs with the
    stock, so no episode would ever register a breakout.
  - Price events are evaluated before list membership. A base that closes
    through its trigger on the same session it falls off the gates has broken
    out; calling it a drop-off would bury the outcome under the noise.

Membership comes from `stocks.gate_flags`, the same evaluation the published
list is drawn from, so this can never disagree with what the reader saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import stocks as stk

BRIDGE = 3                  # non-qualifying sessions tolerated inside a base
HORIZON = 20                # sessions from entry to wait for a resolution
BREAK_DOWN = -0.07          # close this far under entry, thesis is dead
HEAVY_VOL = 1.5             # vol_ratio at which a breakout counts as convicted
TRACK_AFTER_TRIGGER = 10    # keep watching a fired base this long for a failure
FAIL_CLOSES = 2             # closes back inside the base before calling it dead

# `dropped` is watched, not finished. A base that leaves the gates and then
# closes through its trigger a week later did break out, and that case is a
# real share of the 49% trigger rate -- resolving on drop-off would hide it.
LIVE = ("basing", "dropped", "triggered")
RESOLVED = ("triggered", "failed", "broke_down", "stale")

# Order matters: the first failing gate is the one reported, and these run
# from most structural to most incidental so the reason is the useful one.
# A base that lost its uptrend AND saw volume return is a trend failure.
GATE_ORDER = ["trend", "pos", "price", "range", "ext", "rsi", "vol"]

FIELDS = [
    "symbol", "sector", "started_on", "entry_price", "entry_trigger",
    "state", "state_since", "reason", "lost_gates", "last_seen_on",
    "last_close", "peak_close", "peak_high", "coil_sessions", "age", "gap", "below",
    "mom12_1", "triggered_on", "trigger_age", "trigger_vol", "resolved_on",
]


def _num(v):
    """Float or None. NaN is absence, and absence must not become 0.0."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


# Wording tracks MISSING_REASON in the UI glossary so a base that leaves the
# list explains itself the same way the near-miss list does.
GATE_TEXT = {
    "trend": "Lost its uptrend",
    "pos": "Drifted from its highs",
    "price": "Price fell under the floor",
    "range": "Range widened out",
    "ext": "Stretched from the 20-EMA",
    "rsi": "RSI left the 45-68 band",
    "vol": "Volume came back",
}


def _failing(row) -> str:
    """Which gates this row lost, most structural first."""
    lost = [c[2:] for c in stk.FLAG_COLS if not bool(row.get(c, True))]
    lost.sort(key=lambda g: GATE_ORDER.index(g) if g in GATE_ORDER else 99)
    return ",".join(lost)


def _gate_reason(lost: str) -> str:
    """The lost gates as one line, leading with the most structural."""
    if not lost:
        return "Stopped trading"
    parts = lost.split(",")
    text = GATE_TEXT.get(parts[0], parts[0])
    return text if len(parts) == 1 else f"{text} (+{len(parts) - 1} more)"


def _to(ep, state, as_of, reason, resolved=False):
    ep["state"] = state
    ep["state_since"] = as_of
    ep["reason"] = reason
    if resolved:
        ep["resolved_on"] = as_of
    return ep


def _open(symbol, row, as_of) -> dict:
    px = _num(row.get("adj"))
    return {
        "symbol": symbol,
        "sector": row.get("sector"),
        "started_on": as_of,
        "entry_price": px,
        "entry_trigger": _num(row.get("trigger")),
        "state": "basing",
        "state_since": as_of,
        "reason": "New base",
        "lost_gates": None,
        "last_seen_on": as_of,
        "last_close": px,
        "peak_close": px,
        # Seeded at the entry close, not the found day's own high: the list
        # publishes after the close, so that day's intraday high was already
        # gone before anyone could see the name.
        "peak_high": px,
        "coil_sessions": 1,
        "age": 0,
        "gap": 0,
        "below": 0,
        "mom12_1": _num(row.get("mom12_1")),
        "triggered_on": None,
        "trigger_age": None,
        "trigger_vol": None,
        "resolved_on": None,
    }


def _step(ep, row, as_of) -> dict:
    """
    Advance one live episode by a single session.

    `row` is that symbol's row from `gate_flags`, or None when the symbol did
    not trade / had no usable data. A missing row is treated as a gap rather
    than an ending, since a suspension is not a thesis failure.
    """
    ep["age"] += 1

    px = _num(row.get("adj")) if row is not None else None
    if px is not None:
        ep["last_close"] = px
        ep["peak_close"] = max(ep["peak_close"] or px, px)
        # `adj_high` is the intraday high on the same split-adjusted scale as
        # `adj`, so it is comparable to the entry close. Circuit-locked and
        # bad-print days can leave it missing, and the close is the floor.
        hi = _num(row.get("adj_high")) or px
        ep["peak_high"] = max(ep["peak_high"] or hi, hi, px)

    trig, entry = ep["entry_trigger"], ep["entry_price"]

    # Already fired. The only thing left to report is a failure back through
    # the level, then it stops being interesting.
    if ep["state"] == "triggered":
        if px is not None and trig is not None and px < trig:
            ep["below"] += 1
            # Two consecutive closes, not one. Breakouts retest the level
            # constantly, and calling the first dip a failure made it the
            # largest bucket in the replay -- an artefact, not a finding.
            # Persistence is used rather than a price buffer so there is no
            # invented threshold here.
            if ep["below"] >= FAIL_CLOSES:
                return _to(ep, "failed", as_of,
                           f"Back under {trig:,.0f} for "
                           f"{FAIL_CLOSES} straight closes", resolved=True)
        else:
            ep["below"] = 0
        if ep["age"] - (ep["trigger_age"] or 0) >= TRACK_AFTER_TRIGGER:
            ep["resolved_on"] = as_of
        return ep

    # Price events first — see the module docstring.
    if px is not None and trig is not None and px > trig:
        heavy = (_num(row.get("vol_ratio")) or 0.0) >= HEAVY_VOL
        ep["triggered_on"] = as_of
        ep["trigger_age"] = ep["age"]
        ep["trigger_vol"] = bool(heavy)
        return _to(ep, "triggered", as_of,
                   f"Closed through {trig:,.0f} on "
                   f"{'heavy' if heavy else 'light'} volume")

    if px is not None and entry and px / entry - 1 <= BREAK_DOWN:
        return _to(ep, "broke_down", as_of,
                   f"Closed {px / entry - 1:.0%} under where it was found",
                   resolved=True)

    if row is not None and int(row.get("n_fail", 1)) == 0:
        # Only a base that is still basing accrues sessions. A dropped episode
        # is being watched for a late move, and its successor owns the symbol
        # from here -- crediting it for those sessions would count them twice.
        #
        # There is also deliberately no route back from `dropped`. Past BRIDGE
        # the run is over and requalifying starts a fresh episode, or a
        # 3-session tolerance quietly becomes unlimited.
        if ep["state"] == "basing":
            ep["last_seen_on"] = as_of
            ep["coil_sessions"] += 1
            ep["gap"] = 0
            ep["reason"] = "Still basing"
    else:
        ep["gap"] += 1
        if ep["gap"] > BRIDGE and ep["state"] != "dropped":
            # Kept on its own field: the lost gate is the most useful thing
            # this module knows, and the horizon transition below would
            # otherwise overwrite it with a generic message.
            ep["lost_gates"] = _failing(row) if row is not None else ""
            _to(ep, "dropped", as_of, _gate_reason(ep["lost_gates"]))

    # Out of patience. Whatever it was doing, it never resolved either way.
    if ep["age"] >= HORIZON:
        return _to(ep, "stale", as_of, ep["reason"], resolved=True)
    return ep


def advance(eps: list[dict], day: pd.DataFrame, as_of,
            eligible: set[str] | None = None) -> list[dict]:
    """
    Carry every episode forward one session and open one for each new base.

    `eps` is the full set so far; live ones are stepped, resolved ones are
    left untouched. Returns the same list, mutated.

    `eligible` restricts which symbols may START an episode -- normally the
    symbols on the published list for `as_of`. Existing episodes are always
    advanced regardless, because the whole point is to follow a base after it
    leaves the list, and that needs prices for names no longer on it.
    """
    as_of = pd.Timestamp(as_of)
    rows = {} if day is None or day.empty else {
        r["symbol"]: r for r in day.to_dict("records")
    }

    # A `dropped` episode is still being watched for a late breakout, but it no
    # longer owns the symbol -- so it does not block a new base from opening.
    holding = set()
    for ep in eps:
        if ep["state"] in LIVE and ep["resolved_on"] is None:
            _step(ep, rows.get(ep["symbol"]), as_of)
            if ep["resolved_on"] is None and ep["state"] != "dropped":
                holding.add(ep["symbol"])

    if eligible is not None:
        # The published list is the authority on what was shown, so it is not
        # re-checked against the gates here. Re-deriving membership would
        # silently drop any name whose gate result no longer reproduces --
        # a slightly different panel depth is enough to do that -- and the
        # base would vanish from tracking having been on the list.
        for sym in eligible:
            row = rows.get(sym)
            if sym in holding or row is None:
                continue
            eps.append(_open(sym, row, as_of))
        return eps

    for sym, row in rows.items():
        if sym in holding or int(row.get("n_fail", 1)) != 0:
            continue
        eps.append(_open(sym, row, as_of))

    return eps


def build(stocks: pd.DataFrame, p: stk.CoilParams | None = None,
          published: dict | None = None,
          dates=None, warmup: int | None = None) -> pd.DataFrame:
    """
    Replay session by session and return every episode.

    Always a replay, never an incremental step, so this is the one code path
    and there is no second branch whose counters can drift out of agreement
    with a reconstruction. Re-running it is free and repairs itself.

    `published` maps each scan date to the symbols on the published list that
    day, and is how this should normally be called. It makes the tracked set
    exactly the names that were actually shown: an episode can only begin on
    a date the symbol appeared on the list, and the replay starts at the
    earliest published date rather than at the start of the price panel.

    Without it, the fallback replays the raw gates over the whole panel, which
    tracks every base the gates would have found whether or not anyone saw it
    -- useful for research, far too broad for the app. `warmup` then skips the
    leading sessions where indicators are still filling in, or every name
    looks like it starts a base on day one.
    """
    p = p or stk.CoilParams()
    pub = None

    if published:
        pub = {pd.Timestamp(k): set(v) for k, v in published.items() if v}

    if dates is None:
        all_dates = np.sort(stocks["date"].unique())
        if pub:
            # Start where publishing started. Sessions after the last
            # published date are still replayed so open episodes can resolve.
            start = min(pub)
            dates = [d for d in all_dates if pd.Timestamp(d) >= start]
        else:
            if warmup is None:
                warmup = p.min_sessions
            dates = all_dates[warmup:]

    eps: list[dict] = []
    for d in dates:
        # An empty set, not None: a session with no saved list published
        # nothing, so nothing may start. None means "unrestricted" and would
        # quietly reopen the raw-gate behaviour on every unpublished date,
        # including the sessions after the last save while episodes resolve.
        eligible = pub.get(pd.Timestamp(d), set()) if pub is not None else None
        advance(eps, stk.gate_flags(stocks, p, d), d, eligible)
    return frame(eps)


def tag(rows: pd.DataFrame, df: pd.DataFrame, as_of) -> pd.DataFrame:
    """
    Attach each name's live episode age to a displayed list.

    `coil_days` resets to 1 on any single missed session, so a base that has
    been building for six weeks and wobbled once reads as brand new -- that
    was true of 88% of the rows showing `coil_days == 1`. `episode_days` is
    the bridged streak and `episode_new` means the episode really did start
    today.

    Named `episode_*` and not `base_*` because `base_days` is already taken:
    it is the coil base-length indicator, it feeds the coil score, and it is
    stored on both tables. Writing this into that column would corrupt it.
    """
    if rows is None or rows.empty:
        return rows
    out = rows.copy()
    if df is None or df.empty:
        out["episode_days"] = out.get("coil_days")
        out["episode_new"] = False
        return out

    as_of = pd.Timestamp(as_of)
    live = df[(df["state"].isin(LIVE)) & (df["resolved_on"].isna())]
    live = live.sort_values("started_on").drop_duplicates("symbol", keep="last")
    by_sym = live.set_index("symbol")

    out["episode_days"] = out["symbol"].map(by_sym["coil_sessions"])
    out["episode_days"] = out["episode_days"].fillna(out.get("coil_days"))
    started = out["symbol"].map(by_sym["started_on"])
    out["episode_new"] = pd.to_datetime(started, errors="coerce") == as_of
    return out


def frame(eps: list[dict]) -> pd.DataFrame:
    """Episodes as a DataFrame, most recently active first."""
    if not eps:
        return pd.DataFrame(columns=FIELDS)
    df = pd.DataFrame(eps)
    for c in FIELDS:
        if c not in df.columns:
            df[c] = None
    return (df[FIELDS]
            .sort_values(["state_since", "started_on"], ascending=False)
            .reset_index(drop=True))


def digest(df: pd.DataFrame, as_of) -> dict:
    """
    What changed on `as_of` — the line worth reading before the table.

    Counts transitions that happened today, not standing totals, so a quiet
    session honestly reports nothing rather than restating the backlog.
    """
    as_of = pd.Timestamp(as_of)
    if df.empty:
        return {"date": str(as_of.date()), "triggered": 0, "broke_down": 0,
                "new": 0, "dropped": 0, "failed": 0, "live": 0}
    today = df[pd.to_datetime(df["state_since"]) == as_of]
    live = df[(df["state"].isin(LIVE)) & (df["resolved_on"].isna())]
    return {
        "date": str(as_of.date()),
        "triggered": int((today["state"] == "triggered").sum()),
        "broke_down": int((today["state"] == "broke_down").sum()),
        "failed": int((today["state"] == "failed").sum()),
        "dropped": int((today["state"] == "dropped").sum()),
        "new": int((pd.to_datetime(df["started_on"]) == as_of).sum()),
        "live": int(len(live)),
    }
