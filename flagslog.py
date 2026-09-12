"""
Persistent flags log.

Every time the engine builds a dashboard it appends that session's buy setups
and the sector shape verdicts behind them. This is the point the README keeps
making: a hit rate is only meaningful once you have logged the flags you
didn't like too, over enough sessions.

Two files under data/cache/:

    flags_log.csv   one row per buy-setup name (coil ∩ clean pullback)
    shape_log.csv   one row per actionable / buy-ready sector that day

Re-running the same `as_of` replaces that day's rows rather than duplicating
them. Forward returns are deliberately NOT stored — they are unknown at flag
time. Join this log to a later bhavcopy to score it, exactly as `buytest` does
on history.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

import fetch

FLAGS_PATH = fetch.CACHE_DIR / "flags_log.csv"
SHAPE_PATH = fetch.CACHE_DIR / "shape_log.csv"

COLUMNS = [
    "as_of", "logged_at", "symbol", "sector", "sector_klass", "verdict",
    "why", "adj", "trigger", "coil", "to_trigger", "pos_hi", "rsi", "vol_ratio",
    "range20", "cmf", "deliv_quality_rel", "base_days",
]

_CAPTURE = [
    "symbol", "sector", "why", "adj", "trigger", "coil", "to_trigger", "pos_hi",
    "rsi", "vol_ratio", "range20", "cmf", "deliv_quality_rel", "base_days",
]

SHAPE_COLUMNS = [
    "as_of", "logged_at", "sector", "klass", "verdict", "buy_ready",
    "T_rel", "B", "n_setups",
]


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=columns)


def load_flags() -> pd.DataFrame:
    return _read(FLAGS_PATH, COLUMNS)


def load_shapes() -> pd.DataFrame:
    return _read(SHAPE_PATH, SHAPE_COLUMNS)


def _replace_day(path: Path, columns: list[str], as_of: str,
                 new: pd.DataFrame) -> int:
    fetch.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read(path, columns)
    if not existing.empty:
        existing = existing[existing["as_of"].astype(str) != str(as_of)]
    out = pd.concat([existing, new], ignore_index=True)
    out.to_csv(path, index=False)
    return int(len(new))


def append_flags(as_of: str, buys: pd.DataFrame,
                 verdict_by_sector: dict[str, str] | None = None) -> int:
    """Append today's buy setups. Same `as_of` replaces that day's rows."""
    verdict_by_sector = verdict_by_sector or {}
    if buys is None or buys.empty:
        new = pd.DataFrame(columns=COLUMNS)
    else:
        cols = [c for c in _CAPTURE if c in buys.columns]
        new = buys[cols].copy()
        new["as_of"] = as_of
        new["logged_at"] = datetime.now().isoformat(timespec="seconds")
        new["sector_klass"] = "PULLBACK"
        new["verdict"] = new["sector"].map(verdict_by_sector).fillna("")
        if "trigger" not in new.columns or new["trigger"].isna().all():
            adj = pd.to_numeric(new.get("adj"), errors="coerce")
            tt = pd.to_numeric(new.get("to_trigger"), errors="coerce").fillna(0)
            new["trigger"] = adj * (1.0 + tt)
        new = new.reindex(columns=COLUMNS)
    return _replace_day(FLAGS_PATH, COLUMNS, as_of, new)


def append_shapes(as_of: str, scan_rows: pd.DataFrame,
                  verdict_by_sector: dict[str, str] | None = None,
                  n_setups_by_sector: dict[str, int] | None = None) -> int:
    """Append today's sector verdicts for actionable / buy-ready names."""
    verdict_by_sector = verdict_by_sector or {}
    n_setups_by_sector = n_setups_by_sector or {}
    if scan_rows is None or scan_rows.empty:
        new = pd.DataFrame(columns=SHAPE_COLUMNS)
    else:
        mask = scan_rows["klass"].isin(
            ["CROSSING", "PULLBACK", "CROSSING_UNVERIFIED"]
        )
        if "buy_ready" in scan_rows.columns:
            mask = mask | scan_rows["buy_ready"].fillna(False).astype(bool)
        keep = scan_rows[mask].copy()
        if keep.empty:
            new = pd.DataFrame(columns=SHAPE_COLUMNS)
        else:
            new = pd.DataFrame({
                "as_of": as_of,
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                "sector": keep["sector"].values,
                "klass": keep["klass"].values,
                "verdict": keep["sector"].map(verdict_by_sector).fillna(""),
                "buy_ready": keep["buy_ready"].astype(bool).values
                if "buy_ready" in keep.columns else False,
                "T_rel": keep["T_rel"].values if "T_rel" in keep.columns else None,
                "B": keep["B"].values if "B" in keep.columns else None,
                "n_setups": keep["sector"].map(n_setups_by_sector).fillna(0).astype(int),
            }).reindex(columns=SHAPE_COLUMNS)
    return _replace_day(SHAPE_PATH, SHAPE_COLUMNS, as_of, new)


def append_day(as_of: str, buys: pd.DataFrame, scan_rows: pd.DataFrame,
               verdict_by_sector: dict[str, str] | None = None) -> tuple[int, int]:
    """Write both logs for one session. Returns (n_setups, n_sectors)."""
    n_setups = {}
    if buys is not None and not buys.empty:
        n_setups = buys.groupby("sector").size().to_dict()
    a = append_flags(as_of, buys, verdict_by_sector)
    b = append_shapes(as_of, scan_rows, verdict_by_sector, n_setups)
    return a, b
