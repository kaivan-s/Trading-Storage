"""
System jobs: panel refresh + For Tom scan.

These are not per-user. One run updates the in-memory engine (and
Supabase picks) for everyone. Subscription only gates what the UI shows.
"""

from __future__ import annotations

import threading
import time
from datetime import date, time as dtime

import pandas as pd

import db
import position as posscan

# IST weekday slots. Latest due slot runs once if the process starts late.
SCAN_SLOTS = [(12, 0), (14, 30), (15, 35)]
REFRESH_SLOTS = [(16, 15), (18, 0), (20, 30)]
# Sector Lookouts: dedicated post-market sector scan at 19:30 (after bhavcopy ~18:30)
SECTOR_LOOKOUT_SLOT = (19, 30)
SATURDAY_REFRESH = (10, 0)

_started = False
_start_lock = threading.Lock()

state = {
    "refresh_at": None,
    "scan_at": None,
    "sector_lookout_at": None,
    "last_refresh_key": None,
    "last_scan_key": None,
    "last_sector_lookout_key": None,
}


def start(engine, default_days: int) -> None:
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
    threading.Thread(
        target=_loop, args=(engine, default_days), daemon=True, name="jobs"
    ).start()


def _slot_key(day: date, prefix: str, hh: int, mm: int) -> str:
    return f"{day.isoformat()}-{prefix}-{hh:02d}{mm:02d}"


def _latest_due(now, slots: list[tuple[int, int]], prefix: str) -> str | None:
    picked = None
    for hh, mm in slots:
        if now.time() >= dtime(hh, mm):
            picked = _slot_key(now.date(), prefix, hh, mm)
    return picked


def _have_session(engine, sess: date) -> bool:
    with engine._lock:
        raw = engine.raw
        as_of = engine.as_of
    if as_of and str(as_of)[:10] == sess.isoformat():
        return True
    if raw is None or getattr(raw, "empty", True) or "date" not in raw.columns:
        return False
    return bool((pd.to_datetime(raw["date"]).dt.date == sess).any())


def _try_refresh(engine, days: int) -> bool:
    if engine._busy.locked():
        print("[jobs] refresh skipped — busy")
        return False
    print("[jobs] refreshing panel")
    ok = engine.load(days, date.today())
    if ok:
        state["refresh_at"] = db.now_ist().isoformat(timespec="seconds")
    return bool(ok)


def _try_scan(engine) -> bool:
    if getattr(engine, "status", None) != "ready":
        print("[jobs] scan skipped — engine not ready")
        return False
    if engine._busy.locked():
        print("[jobs] scan skipped — busy")
        return False
    print("[jobs] scanning For Tom")
    ok = engine.scan_all()
    if ok:
        state["scan_at"] = db.now_ist().isoformat(timespec="seconds")
    return bool(ok)


def _try_sector_lookout(engine) -> bool:
    """
    Post-market sector lookouts: save sector scans with shape reports to DB.
    Runs at 19:30 IST after bhavcopy is available.
    """
    if getattr(engine, "status", None) != "ready":
        print("[jobs] sector lookout skipped — engine not ready")
        return False
    if engine._busy.locked():
        print("[jobs] sector lookout skipped — busy")
        return False
    
    with engine._lock:
        scan_rows = engine.scan_rows
        panel = engine.panel
        as_of = engine.as_of
    
    if scan_rows is None or scan_rows.empty:
        print("[jobs] sector lookout skipped — no scan data")
        return False
    
    print("[jobs] saving sector lookouts")
    ok = False
    try:
        n = db.save_sector_scans(as_of, scan_rows, panel)
        print(f"[jobs] saved {n} sector scans")
        state["sector_lookout_at"] = db.now_ist().isoformat(timespec="seconds")
        ok = n > 0
    except Exception as e:
        print(f"[jobs] sector lookout error: {e}")

    # Position Trades rides the same post-market slot: it is an EOD scan on a
    # 7-10 session horizon, so it wants closing prices, not the 15:35 quotes.
    try:
        with engine._lock:
            coil_stocks = engine.coil_stocks
        if coil_stocks is not None and not getattr(coil_stocks, "empty", True):
            rows = posscan.scan(
                posscan.add_position_features(coil_stocks), as_of=as_of
            )
            if not rows.empty:
                print(f"[jobs] saved {db.save_position_trades(as_of, rows)} "
                      "position trades")
            else:
                print("[jobs] no position trades — needs 270 sessions of history")
    except Exception as e:
        print(f"[jobs] position trades error: {e}")
    return ok


def _loop(engine, days: int) -> None:
    while getattr(engine, "status", "idle") in ("idle", "loading"):
        time.sleep(2)

    now = db.now_ist()
    if getattr(engine, "status", None) == "ready":
        if _try_scan(engine):
            state["last_scan_key"] = (
                _latest_due(now, SCAN_SLOTS, "S") or f"{now.date().isoformat()}-S-boot"
            )
        if _have_session(engine, db.session_date(now)):
            due_r = _latest_due(now, REFRESH_SLOTS, "R")
            if due_r:
                state["last_refresh_key"] = due_r

    while True:
        try:
            now = db.now_ist()
            wd = now.weekday()

            if wd < 5:
                due_s = _latest_due(now, SCAN_SLOTS, "S")
                if due_s and due_s != state["last_scan_key"]:
                    if _try_scan(engine):
                        state["last_scan_key"] = due_s

                due_r = _latest_due(now, REFRESH_SLOTS, "R")
                if due_r and due_r != state["last_refresh_key"]:
                    if _try_refresh(engine, days):
                        state["last_refresh_key"] = due_r
                        if _try_scan(engine):
                            state["last_scan_key"] = f"{now.date().isoformat()}-S-after-r"
                
                # Sector Lookouts at 19:30 (after bhavcopy is available)
                hh, mm = SECTOR_LOOKOUT_SLOT
                due_l = (
                    _slot_key(now.date(), "L", hh, mm)
                    if now.time() >= dtime(hh, mm)
                    else None
                )
                if due_l and due_l != state["last_sector_lookout_key"]:
                    if _try_sector_lookout(engine):
                        state["last_sector_lookout_key"] = due_l

            elif wd == 5:
                hh, mm = SATURDAY_REFRESH
                due_r = (
                    _slot_key(now.date(), "R", hh, mm)
                    if now.time() >= dtime(hh, mm)
                    else None
                )
                if due_r and due_r != state["last_refresh_key"]:
                    if _try_refresh(engine, days):
                        state["last_refresh_key"] = due_r
                        if _try_scan(engine):
                            state["last_scan_key"] = f"{now.date().isoformat()}-S-sat"
        except Exception as exc:
            print(f"[jobs] {exc}")
        time.sleep(30)
