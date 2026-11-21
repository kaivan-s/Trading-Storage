"""
One audit of every live list, on the full cash bhavcopy and the fair ruler.

Ruler (same as rsi_ema_sr.py): entry at the next session's open, forward
returns at 5/10/20 sessions, benchmark = that day's MEAN return of the liquid
universe on the same measurement (clipped -0.9..3), one signal per symbol per
horizon (cooldown = horizon). PASS = beats the benchmark by more than 0.25%
cost in both halves of the window.

Sections
  1. Coils (all seven gates) and the coil score.
  2. Filter ablation: names failing exactly one gate vs full coils.
  3. Leaders at rest (12-1 momentum ordering of the coil pool) vs other coils.
  4. 12-1 momentum alone (the dormant Position scan), 20-session holds.
  5. Sector classifier: stocks and coils inside CROSSING / PULLBACK sectors.
  6. Confirmed breakouts: coils that later closed through the trigger on volume.
  7. Data check: daily moves of 35-60% (likely unadjusted splits).

    ~/anaconda3/bin/python attic/algo_audit.py build   # features -> /tmp parquet
    ~/anaconda3/bin/python attic/algo_audit.py run
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch  # noqa: E402
import panel as pnl  # noqa: E402
import position as pos  # noqa: E402
import scan as sc  # noqa: E402
import stocks as stk  # noqa: E402

H = (5, 10, 20)
COST = 0.25
FEAT = Path("/tmp/algo_audit_feat.parquet")
PANEL = Path("/tmp/algo_audit_panel.parquet")
KLASS = Path("/tmp/algo_audit_klass.parquet")
P = stk.CoilParams()


def build() -> None:
    files = sorted(fetch.RAW_DIR.glob("bhav_*"))
    last = pd.Timestamp(files[-1].stem.split("_")[1]).date()
    raw = fetch.load_history(last, len(files), verbose=False)
    smap = pd.read_json(fetch.CACHE_DIR / "sector_map.json", orient="index").rename_axis("symbol").reset_index()
    stocks, panel = pnl.build(raw, smap, sector_level="basic_industry")
    d = pos.add_position_features(stk.add_indicators(stocks))
    d.to_parquet(FEAT)
    panel.to_parquet(PANEL)
    print("features", d.shape, d["date"].min(), d["date"].max())

    # Sector class on every date. classify_one reads only the last 12 rows.
    th = sc.Thresholds()
    rows = []
    for sec, g in panel.sort_values(["sector", "date"]).groupby("sector", sort=False):
        g = g.reset_index(drop=True)
        for i in range(25, len(g)):
            r = sc.classify_one(g.iloc[max(0, i - 11):i + 1], th)
            rows.append((sec, g.at[i, "date"], r["klass"]))
    k = pd.DataFrame(rows, columns=["sector", "date", "klass"])
    k.to_parquet(KLASS)
    print("klass", k.shape, k["klass"].value_counts().to_dict())


def forward(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["symbol", "date"]).copy()
    g = d.groupby("symbol", sort=False)
    scale = d["adj"] / d["close"].replace(0, np.nan)
    d["adj_open"] = d["open"] * scale
    entry = g["adj_open"].shift(-1)
    for h in H:
        d[f"f{h}"] = (g["adj"].shift(-h) / entry - 1).clip(-0.9, 3)
    d["liquid"] = ((d["own_sessions"] >= P.min_sessions) & (d["med_turn60"] >= P.min_median_turnover_lacs)
                   & (d["adj"] >= P.min_price))
    for h in H:
        bench = d[d["liquid"]].groupby("date")[f"f{h}"].mean()
        d[f"x{h}"] = d[f"f{h}"] - d["date"].map(bench)
    return d


def gates(d: pd.DataFrame) -> pd.DataFrame:
    d["f_price"] = d["adj"] >= P.min_price
    d["f_trend"] = (d["adj"] > d["ema50"]) & (d["ema50"] > d["ema200"])
    d["f_pos"] = d["pos_hi"].between(P.near_high_min, P.near_high_max)
    d["f_rsi"] = d["rsi"].between(P.rsi_low, P.rsi_high)
    d["f_ext"] = d["ext_ema20"].abs() <= P.max_ext_ema20
    d["f_vol"] = d["vol_ratio"] <= P.vol_dryup_max
    d["f_range"] = d["range20"] <= P.range_max
    d["n_fail"] = (~d[stk.FLAG_COLS]).sum(axis=1)
    d["coil_hit"] = d["liquid"] & (d["n_fail"] == 0)
    return d


def cooldown(d: pd.DataFrame, mask: pd.Series, h: int) -> pd.Series:
    """Keep a signal only if the same symbol hasn't fired in the previous h sessions."""
    sess = d.groupby("symbol", sort=False).cumcount()
    out = pd.Series(False, index=d.index)
    last = {}
    for i, s, k in zip(d.index[mask], d.loc[mask, "symbol"], sess[mask]):
        if s not in last or k - last[s] >= h:
            out.at[i] = True
            last[s] = k
    return out


def report(name: str, d: pd.DataFrame, mask: pd.Series, sign: int = 1, hs=H) -> dict:
    mid = d["date"].sort_values().unique()
    mid = mid[len(mid) // 2]
    parts = []
    res = {"name": name}
    for h in hs:
        m = cooldown(d, mask & d[f"x{h}"].notna(), h)
        x = sign * d.loc[m, f"x{h}"] * 100
        a = x[d.loc[m, "date"] < mid]
        b = x[d.loc[m, "date"] >= mid]
        ok = a.mean() > COST and b.mean() > COST
        res[h] = (len(x), x.mean(), (x > 0).mean() * 100, a.mean(), b.mean(), ok)
        parts.append(f"{h:>2}d n={len(x):>6,} {x.mean():+6.2f}% beat {100 * (x > 0).mean():3.0f}% "
                     f"[{a.mean():+.2f} / {b.mean():+.2f}]{' PASS' if ok else ''}")
    print(f"{name:52s} " + " | ".join(parts))
    return res


def run() -> None:
    d = forward(pd.read_parquet(FEAT))
    d = gates(d)
    k = pd.read_parquet(KLASS)
    d = d.merge(k, on=["sector", "date"], how="left")
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"{d['date'].min():%Y-%m-%d} -> {d['date'].max():%Y-%m-%d}, mean benchmark, next-open entry, "
          "[1st half / 2nd half], PASS = both halves > 0.25%\n")

    liq = d["liquid"]
    coil = d["coil_hit"]
    print("== 1. Coils")
    report("All liquid stocks (sanity, ~0)", d, liq & (d.groupby("symbol").cumcount() % 20 == 0))
    report("Coils (all 7 gates)", d, coil)
    parts = ["vol_ratio", "range20", "contraction", "cmf", "deliv_quality_rel", "base_days"]
    sc_ = pd.DataFrame({
        "s_vol": stk._norm(d["vol_ratio"], 0.40, 1.00, invert=True),
        "s_range": stk._norm(d["range20"], 0.04, P.range_max, invert=True),
        "s_contract": stk._norm(d["contraction"].fillna(1.0), 0.4, 1.2, invert=True),
        "s_cmf": stk._norm(d["cmf"].fillna(0.0), -0.05, 0.20),
        "s_deliv": stk._norm(d["deliv_quality_rel"].fillna(1.0), 0.85, 1.40),
        "s_base": stk._norm(d["base_days"].fillna(0), 25, 55),
    })
    d["coil_score"] = sc_.mean(axis=1) * 100
    med = d.loc[coil].groupby("date")["coil_score"].transform("median")
    hi = pd.Series(False, index=d.index)
    hi.loc[med.index] = d.loc[med.index, "coil_score"] >= med
    report("  coil score top half", d, coil & hi)
    report("  coil score bottom half", d, coil & ~hi)
    for c, lab in [("deliv_quality_rel", "delivery quality > 1.1"), ("cmf", "CMF > 0.1")]:
        report(f"  coils with {lab}", d, coil & (d[c] > (1.1 if c == "deliv_quality_rel" else 0.1)))

    print("\n== 2. Filter ablation (fails exactly this gate, passes the other six)")
    for f in stk.FLAG_COLS:
        if f == "f_price":
            continue
        report(f"  fails only {f[2:]}", d, liq & (d["n_fail"] == 1) & ~d[f])

    print("\n== 3. Leaders at rest")
    elig = liq & (d["med_turn60"] >= 100) & (d["own_sessions"] >= pos.FORMATION + pos.SKIP) & d["mom12_1"].notna()
    d["mom_pct"] = np.nan
    d.loc[elig, "mom_pct"] = d[elig].groupby("date")["mom12_1"].rank(pct=True)
    rest_ok = coil & (d["mom_pct"] >= pos.REST_MIN_PCTILE) & (d["klass"] != "DISQUALIFIED")
    rank = d[rest_ok].groupby("date")["mom_vadj"].rank(ascending=False)
    top20 = pd.Series(False, index=d.index)
    top20.loc[rank.index] = rank <= pos.REST_TOP_N
    report("Leaders at rest (top 20)", d, top20)
    report("Coils not in Leaders at rest", d, coil & ~top20 & d["mom12_1"].notna())
    report("Coils with 12-1 in bottom half", d, coil & (d["mom_pct"] < 0.5))

    print("\n== 4. 12-1 momentum alone (Position scan, dormant), every 20th session")
    every = d["date"].isin(d["date"].drop_duplicates().sort_values().iloc[::20])
    top10 = elig & (d["mom_pct"] >= 0.9) & (d["adj"] > d["ema50"])
    report("Top decile 12-1, above EMA50", d, top10 & every, hs=(20,))
    report("Top decile 12-1, any", d, elig & (d["mom_pct"] >= 0.9) & every, hs=(20,))
    report("Bottom decile 12-1", d, elig & (d["mom_pct"] <= 0.1) & every, hs=(20,))

    print("\n== 5. Sector classifier")
    for kl in ["CROSSING", "PULLBACK", "CROSSING_UNVERIFIED", "BASE", "NEGLECT", "DISQUALIFIED"]:
        report(f"  liquid stocks in {kl} sectors", d, liq & (d["klass"] == kl))
    setup = d["klass"].isin(["CROSSING", "PULLBACK", "CROSSING_UNVERIFIED"])
    report("Coils in setup sectors ('sector agrees')", d, coil & setup)
    report("Coils elsewhere", d, coil & ~setup)

    print("\n== 6. Confirmed breakouts (coil in last 20 sessions, close > prior 20d high, vol > 1.5x)")
    g = d.groupby("symbol", sort=False)
    recent_coil = g["coil_hit"].transform(lambda s: s.astype(int).rolling(20, min_periods=1).max().shift(1)).fillna(0) > 0
    v20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    brk = liq & recent_coil & (d["adj"] > d["prior_trigger"]) & (d["volume"] > 1.5 * v20)
    report("Breakout after a coil", d, brk)
    report("Breakout, no coil before", d, liq & ~recent_coil & (d["adj"] > d["prior_trigger"]) & (d["volume"] > 1.5 * v20) & d["f_trend"])

    print("\n== 7. Data check")
    r = d["close"] / d["prev_close"] - 1
    big = d[(r.abs() >= 0.35) & (r.abs() < 0.6)]
    print(f"daily moves of 35-60%: {len(big)} rows, {big['symbol'].nunique()} symbols; "
          f"falls {(big['close'] < big['prev_close']).sum()}")
    print(big.assign(r=r[big.index].round(3))[["symbol", "date", "prev_close", "open", "close", "r"]]
          .sort_values("date").tail(15).to_string(index=False))


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1]]()
