"""
Does F&O positioning say anything about the next few weeks?

Two tests on the cached cash history, joined to NSE's F&O bhavcopy:

A. Wide: every F&O stock, every day. Bucket by the 5-day change in total
   futures OI (all expiries, de-meaned against the same day's cross-section so
   expiry roll-offs do not read as unwinding) crossed with the 5-day price
   change. Forward returns are measured against the average F&O stock on the
   same date.

B. Coils: the production coil scan, restricted to F&O names. Coils with OI
   building versus coils with OI falling.

Also reports put/call OI ratio (options, all strikes) as a contrarian check.

t-stats use non-overlapping dates (every h-th session) so a 20-day window is
not counted twenty times.

    python fotest.py fetch     # download/cache F&O files for cached sessions
    python fotest.py run       # run both tests
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from datetime import date

import numpy as np
import pandas as pd

import fetch
import panel as pnl
import stocks as stk

FO_URL = fetch.ARCHIVES + "/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
FO_DIR = fetch.RAW_DIR / "fo"
HORIZONS = (5, 10, 20)
LOOK = 5


def _cash_dates() -> list[date]:
    out = []
    for p in sorted(fetch.RAW_DIR.glob("bhav_*.*")):
        out.append(pd.Timestamp(p.name[5:13]).date())
    return out


def _fo_path(d: date):
    return FO_DIR / f"fo_{d:%Y%m%d}.csv.gz"


def _parse_fo(content: bytes, d: date) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(content))
    df = pd.read_csv(z.open(z.namelist()[0]))
    df = df[df["FinInstrmTp"].isin(["STF", "STO"])]
    fut = (df[df["FinInstrmTp"] == "STF"]
           .groupby("TckrSymb")
           .agg(fut_oi=("OpnIntrst", "sum"), fut_vol=("TtlTradgVol", "sum")))
    opt = df[df["FinInstrmTp"] == "STO"]
    call_oi = opt[opt["OptnTp"] == "CE"].groupby("TckrSymb")["OpnIntrst"].sum()
    put_oi = opt[opt["OptnTp"] == "PE"].groupby("TckrSymb")["OpnIntrst"].sum()
    out = fut.join(call_oi.rename("call_oi"), how="left").join(put_oi.rename("put_oi"), how="left")
    out = out.reset_index().rename(columns={"TckrSymb": "symbol"})
    out["date"] = pd.Timestamp(d)
    return out


def fetch_fo() -> None:
    FO_DIR.mkdir(parents=True, exist_ok=True)
    todo = [d for d in _cash_dates() if not _fo_path(d).exists()]
    print(f"{len(todo)} F&O files to fetch")
    sess = fetch.NSESession() if todo else None
    for i, d in enumerate(todo, 1):
        try:
            r = sess.get(FO_URL.format(yyyymmdd=f"{d:%Y%m%d}"))
            _parse_fo(r.content, d).to_csv(_fo_path(d), index=False, compression="gzip")
            print(f"  {d} ok ({i}/{len(todo)})")
        except Exception as exc:
            print(f"  {d} failed: {exc}")
        time.sleep(0.4)


def load_fo() -> pd.DataFrame:
    frames = [pd.read_csv(p, compression="gzip") for p in sorted(FO_DIR.glob("fo_*.csv.gz"))]
    fo = pd.concat(frames, ignore_index=True)
    fo["date"] = pd.to_datetime(fo["date"])
    fo = fo.sort_values(["symbol", "date"])
    g = fo.groupby("symbol", sort=False)
    fo["oi_chg"] = g["fut_oi"].transform(lambda s: s / s.shift(LOOK) - 1.0)
    fo["pcr"] = fo["put_oi"] / fo["call_oi"].replace(0, np.nan)
    fo["pcr_chg"] = g["pcr"].transform(lambda s: s - s.shift(LOOK))
    # Expiry days move every name's OI together; rank within the day instead.
    fo["oi_rel"] = fo["oi_chg"] - fo.groupby("date")["oi_chg"].transform("median")
    return fo


def load_stocks() -> pd.DataFrame:
    raw = pd.concat([fetch._read_cache(p) for p in sorted(fetch.RAW_DIR.glob("bhav_*.*"))],
                    ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = fetch.drop_repeat_sessions(raw, verbose=False)
    smap = pd.read_json(fetch.CACHE_DIR / "sector_map.json", orient="index") \
             .rename_axis("symbol").reset_index()
    stocks, _ = pnl.build(raw, smap)
    stocks = stk.add_indicators(stocks)
    stocks = stocks.sort_values(["symbol", "date"])
    g = stocks.groupby("symbol", sort=False)["tri"]
    stocks["px_chg"] = g.transform(lambda s: s / s.shift(LOOK) - 1.0)
    for h in HORIZONS:
        stocks[f"fwd_{h}"] = g.transform(lambda s, h=h: s.shift(-h) / s - 1.0)
    return stocks


def _tstat(per_date: pd.Series, h: int) -> tuple[float, float, int]:
    """Mean and t-stat on every h-th date so forward windows do not overlap."""
    s = per_date.dropna()
    s = s.iloc[::h]
    if len(s) < 3:
        return np.nan, np.nan, len(s)
    return s.mean(), s.mean() / (s.std(ddof=1) / np.sqrt(len(s))), len(s)


def _bucket_report(df: pd.DataFrame, label_col: str, title: str) -> None:
    print(f"\n=== {title} ===")
    rows = []
    for lab, grp in df.groupby(label_col):
        rec = {"bucket": lab, "name_days": len(grp)}
        for h in HORIZONS:
            per_date = grp.groupby("date")[f"ex_{h}"].mean()
            m, t, n = _tstat(per_date, h)
            rec[f"ex_{h}%"] = round(m * 100, 2) if pd.notna(m) else np.nan
            rec[f"t_{h}"] = round(t, 2) if pd.notna(t) else np.nan
            rec[f"hit_{h}%"] = round((grp[f"ex_{h}"] > 0).mean() * 100, 1)
        rows.append(rec)
    print(pd.DataFrame(rows).to_string(index=False))


def _spread_report(df: pd.DataFrame, col: str, title: str) -> None:
    """Top-quintile minus bottom-quintile excess return, per date."""
    print(f"\n=== {title}: top quintile minus bottom quintile of {col} ===")
    d = df.dropna(subset=[col]).copy()
    d["q"] = d.groupby("date")[col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if s.size >= 10 else np.nan)
    for h in HORIZONS:
        top = d[d["q"] == 4].groupby("date")[f"ex_{h}"].mean()
        bot = d[d["q"] == 0].groupby("date")[f"ex_{h}"].mean()
        m, t, n = _tstat((top - bot), h)
        print(f"  {h:>2}d: spread {m*100:+.2f}%  t={t:.2f}  (non-overlapping dates={n})")


def run() -> None:
    print("Loading cash panel + indicators…")
    stocks = load_stocks()
    fo = load_fo()
    fno = set(fo["symbol"])

    df = stocks[stocks["symbol"].isin(fno)].merge(
        fo[["symbol", "date", "oi_chg", "oi_rel", "pcr", "pcr_chg"]],
        on=["symbol", "date"], how="inner")
    for h in HORIZONS:
        df[f"ex_{h}"] = df[f"fwd_{h}"] - df.groupby("date")[f"fwd_{h}"].transform("mean")
    df = df.dropna(subset=["oi_rel", "px_chg"])
    print(f"F&O name-days: {len(df):,}  symbols: {df['symbol'].nunique()}  "
          f"dates: {df['date'].nunique()} ({df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d})")

    # A. Price × OI quadrants
    px_up = df["px_chg"] > 0
    oi_up = df["oi_rel"] > 0
    df["quadrant"] = np.select(
        [px_up & oi_up, px_up & ~oi_up, ~px_up & oi_up, ~px_up & ~oi_up],
        ["1 long build-up", "2 short covering", "3 short build-up", "4 long unwinding"],
        default="")
    _bucket_report(df, "quadrant", "A. All F&O stocks — 5d price × 5d futures OI (excess vs F&O average)")

    strong = df[df["oi_rel"].abs() > df.groupby("date")["oi_rel"].transform(lambda s: s.abs().quantile(0.7))]
    _bucket_report(strong, "quadrant", "A2. Same, only the 30% largest OI moves each day")

    _spread_report(df, "oi_rel", "A3. OI change alone")
    _spread_report(df, "pcr_chg", "A4. Put/call OI ratio change")

    # B. Coils in F&O
    print("\nRunning coil scan on every date (slow)…")
    dates = sorted(df["date"].unique())
    hits = []
    for i, as_of in enumerate(dates, 1):
        h = stk.scan(stocks[stocks["date"] <= as_of], as_of=as_of, top=10_000)
        if not h.empty:
            h = h[h["symbol"].isin(fno)][["symbol"]].assign(date=as_of)
            hits.append(h)
        if i % 20 == 0:
            print(f"  {i}/{len(dates)}")
    if not hits:
        print("No F&O coils found.")
        return
    coils = pd.concat(hits).merge(df, on=["symbol", "date"], how="inner")
    coils["oi_side"] = np.where(coils["oi_rel"] > 0, "OI building", "OI falling")
    _bucket_report(coils, "oi_side", "B. F&O coils — OI building vs falling (excess vs F&O average)")
    coils["all"] = "all F&O coils"
    _bucket_report(coils, "all", "B0. F&O coils, no OI filter (reference)")
    print(f"\nF&O coil name-days: {len(coils)}, distinct stocks: {coils['symbol'].nunique()}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"fetch": fetch_fo, "run": run}[cmd]()
