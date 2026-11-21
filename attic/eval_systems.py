#!/usr/bin/env python3
"""
Trading systems on one ruler, over the whole cache (Oct 2023 onward).

Two families, each judged the way it would actually be traded.

INTRADAY  decided at the 09:15 open from yesterday's bar plus today's open;
          entered at the open, exited at the close. Shorts are allowed in any
          cash stock because they are squared off the same day. A day's
          result is the equal-weight mean of its top MAX_PER_DAY signals, net
          of INTRA_COST. Reported raw (what the account earns) and as excess
          over the universe's own open->close median that day, so a short
          that only works on red days shows up as market beta, not edge.

SWING     decided at the close; entered at the next open (adjusted, as in
          baserates.outcomes); held a fixed k sessions. Excess is against the
          liquid universe's MEDIAN k-session return measured on the same
          entry date -- the eval_strategies ruler -- so a no-selection
          control reads 0.00. Statistic: mean over entry dates of the day's
          median excess, Newey-West at lag k because daily k-session windows
          overlap. SWING_COST is a delivery round trip. Short-side swing
          signals are only tradable in F&O names (futures), so those are
          reported for the F&O subset as well.

STABILITY every system is split by calendar year (2024 / 2025 / 2026). Many
          cells are tested here, so a single t-stat is not the bar: a
          system has to clear costs, carry |t| >= 3, and keep its sign in
          every year.

VERDICT (Oct 2023 -> Nov 2026)
  Survives: fading opening gaps in liquid non-F&O names. Short gap-ups 2-9%
  (trend does not matter) and buy gap-downs 2-9% in downtrends. Each leg
  alone is mostly market direction (shorts earn on red days, buys on green
  days); together they are +0.62%/day net, t 8.4, positive every year but
  shrinking (0.91 / 0.52 / 0.36 for 2024-26), max drawdown -24%. Outcomes
  are skewed: dropping the best 5% of trades leaves ~+0.2%. On 5-min bars
  (Jul-Oct 2026) the short keeps ~+0.3% when entered at 09:25-09:30; the
  gap-down buy's median turns negative after the opening print. In F&O
  names the gap fades do not work.
  Dead: every "go with the gap" version, every day-after-a-big-move system.
  Avoid-only: a -5% day through the 200-DMA and a low-delivery volume spike
  lag the universe by ~1% over 10-20 sessions, but not in F&O names, so
  they are exit signals, not shorts.
  No swing long clears the bar. The worst-2% reversal and 12-1 momentum
  verdicts from the 15-month data did not hold over three years.

    cd backend && .venv/bin/python attic/eval_systems.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import baserates as br  # noqa: E402
import filing_facts  # noqa: E402

MIN_TURNOVER_CR = 10
MIN_PRICE = 50
MAX_PER_DAY = 5
INTRA_COST = 0.0015    # brokerage + STT + charges + slippage, intraday round trip
SWING_COST = 0.0030    # delivery round trip
TOP_SWING = 20


def nw_t(x: pd.Series, lag: int) -> float:
    x = x.dropna().to_numpy(float)
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    s = (e @ e) / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def features(f: pd.DataFrame) -> pd.DataFrame:
    f = f.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = f.groupby("symbol", sort=False)
    a = f["adj"]
    f["gap"] = f["open"] / f["prev_close"] - 1
    f["oc"] = f["close"] / f["open"] - 1
    f["close_loc"] = ((f["close"] - f["low"]) / (f["high"] - f["low"])).where(f["high"] > f["low"], 1.0)
    f["r1"] = a / g["adj"].shift(1) - 1
    f["r3"] = a / g["adj"].shift(3) - 1
    f["r20"] = a / g["adj"].shift(20) - 1
    f["mom12_1"] = g["adj"].shift(20) / g["adj"].shift(250) - 1
    f["sma50"] = g["adj"].transform(lambda s: s.rolling(50, min_periods=40).mean())
    f["sma200"] = g["adj"].transform(lambda s: s.rolling(200, min_periods=150).mean())
    f["hi250"] = g["adj"].transform(lambda s: s.rolling(250, min_periods=200).max())
    f["vol_x"] = f["volume"] / g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    f["deliv_avg60"] = g["deliv_pct"].transform(lambda s: s.shift(1).rolling(60, min_periods=30).mean())
    f["med_to"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=15).median()) / 100
    dn = (f["r1"] < 0).astype(int)
    f["down3"] = dn.groupby(f["symbol"]).transform(lambda s: s.rolling(3).sum()) == 3
    f["frozen"] = f["high"] == f["low"]
    # Yesterday's facts, for the intraday family.
    for c in ("r1", "r20", "close_loc", "vol_x", "deliv_pct", "adj", "sma50", "sma200", "med_to"):
        f[f"p_{c}"] = g[c].shift(1)
    f["mkt_gap"] = f.groupby("date")["gap"].transform("median")
    return f


# --------------------------------------------------------------------------
# Intraday
# --------------------------------------------------------------------------

def intraday_systems():
    up = lambda d: d.p_adj > d.p_sma50  # noqa: E731
    gap_up = lambda d, lo=0.02: (d.gap >= lo) & (d.gap < 0.09)  # noqa: E731
    gap_dn = lambda d, lo=0.02: (d.gap <= -lo) & (d.gap > -0.09)  # noqa: E731
    return [
        ("Gap-up in downtrend -> short", "short", lambda d: gap_up(d) & ~up(d), lambda d: d.gap),
        ("Gap-up 4-9% in downtrend -> short", "short", lambda d: gap_up(d, 0.04) & ~up(d), lambda d: d.gap),
        ("Gap-up in uptrend -> short", "short", lambda d: gap_up(d) & up(d), lambda d: d.gap),
        ("Gap-up in uptrend -> buy (go)", "long", lambda d: gap_up(d) & up(d), lambda d: d.gap),
        ("Gap-down in uptrend -> buy", "long", lambda d: gap_dn(d) & up(d), lambda d: -d.gap),
        ("Gap-down in downtrend -> buy", "long", lambda d: gap_dn(d) & ~up(d), lambda d: -d.gap),
        ("Gap-down in downtrend -> short (go)", "short", lambda d: gap_dn(d) & ~up(d), lambda d: -d.gap),
        ("Lone gap-up >=3% (market flat/red) -> short", "short",
         lambda d: gap_up(d, 0.03) & (d.mkt_gap <= 0), lambda d: d.gap),
        ("Lone gap-down >=3% (market flat/green) -> buy", "long",
         lambda d: gap_dn(d, 0.03) & (d.mkt_gap >= 0), lambda d: -d.gap),
        ("Day after ~circuit close (+9.5%, at high) -> short", "short",
         lambda d: (d.p_r1 >= 0.095) & (d.p_close_loc >= 0.98), lambda d: d.p_r1),
        ("Day after low-delivery spike (+5%, 4x vol, <=20% deliv) -> short", "short",
         lambda d: (d.p_r1 >= 0.05) & (d.p_vol_x >= 4) & (d.p_deliv_pct <= 20), lambda d: d.p_vol_x),
        ("Low-delivery spike, then gaps up again -> short", "short",
         lambda d: (d.p_r1 >= 0.05) & (d.p_deliv_pct <= 25) & (d.gap >= 0.02), lambda d: d.gap),
        ("Day after -8% loser -> buy", "long", lambda d: d.p_r1 <= -0.08, lambda d: -d.p_r1),
        ("Day after +8% winner -> short", "short", lambda d: d.p_r1 >= 0.08, lambda d: d.p_r1),
    ]


def run_intraday(f: pd.DataFrame, universe: pd.Series) -> pd.DataFrame:
    base = f[universe & f["gap"].notna() & f["oc"].notna()]
    days = pd.Index(sorted(base["date"].unique()))
    mkt_oc = base.groupby("date")["oc"].median()
    out = []
    for name, side, mask, rank in intraday_systems():
        sel = base[mask(base).fillna(False)].copy()
        if len(sel) < 30:
            continue
        sign = 1 if side == "long" else -1
        sel["gross"] = sign * sel["oc"]
        sel["xs"] = sign * (sel["oc"] - sel["date"].map(mkt_oc))
        sel["rank"] = rank(sel)
        sel = sel.sort_values(["date", "rank"], ascending=[True, False]).groupby("date").head(MAX_PER_DAY)
        net = sel["gross"] - INTRA_COST
        daily = (sel.assign(net=net).groupby("date")["net"].mean())
        dxs = sel.groupby("date")["xs"].mean()
        yr = daily.groupby(daily.index.year).mean() * 100
        eq = (1 + daily.reindex(days, fill_value=0)).cumprod()
        out.append({
            "system": name, "trades": len(sel), "days": len(daily),
            "avg_net%": net.mean() * 100, "median%": net.median() * 100,
            "win%": (net > 0).mean() * 100,
            "day_t": daily.mean() / daily.std() * math.sqrt(len(daily)) if daily.std() > 0 else np.nan,
            "xs_vs_mkt%": dxs.mean() * 100,
            "y2024": yr.get(2024, np.nan), "y2025": yr.get(2025, np.nan), "y2026": yr.get(2026, np.nan),
            "maxDD%": (eq / eq.cummax() - 1).min() * 100,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Swing
# --------------------------------------------------------------------------

def swing_systems():
    return [
        ("Worst 2% of 20-day performers -> buy", +1, lambda d: d.r20 <= d.r20_q02, lambda d: -d.r20),
        ("Worst 2% 20-day, still above 200-DMA -> buy", +1,
         lambda d: (d.r20 <= d.r20_q02) & (d.adj > d.sma200), lambda d: -d.r20),
        ("12-1 momentum top 20 -> buy", +1, lambda d: d.mom12_1.notna(), lambda d: d.mom12_1),
        ("12-1 momentum top 20, within 10% of 52w high -> buy", +1,
         lambda d: d.adj >= 0.9 * d.hi250, lambda d: d.mom12_1),
        ("Power gap: gap>=4%, closes strong, 3x vol -> buy", +1,
         lambda d: (d.gap >= 0.04) & (d.close_loc >= 0.7) & (d.close >= d.open) & (d.vol_x >= 3), lambda d: d.gap),
        ("52-week-high close on 2x volume -> buy", +1,
         lambda d: (d.adj >= d.hi250) & (d.vol_x >= 2), lambda d: d.vol_x),
        ("3 down days in an uptrend (-6%+) -> buy", +1,
         lambda d: d.down3 & (d.r3 <= -0.06) & (d.adj > d.sma200), lambda d: -d.r3),
        ("Quiet accumulation near highs (deliv 1.5x, flat day) -> buy", +1,
         lambda d: (d.deliv_pct >= 1.5 * d.deliv_avg60) & (d.r1.abs() < 0.02) & (d.adj >= 0.9 * d.hi250),
         lambda d: d.deliv_pct / d.deliv_avg60),
        ("Gap-down in uptrend that closes green -> buy", +1,
         lambda d: (d.gap <= -0.02) & (d.close > d.prev_close) & (d.adj > d.sma50), lambda d: -d.gap),
        ("Low-delivery spike (+5%, 4x vol, <=20% deliv) -> short", -1,
         lambda d: (d.r1 >= 0.05) & (d.vol_x >= 4) & (d.deliv_pct <= 20), lambda d: d.vol_x),
        ("Bounce after steep fall (-20% over prior 20d, +5% today) -> short", -1,
         lambda d: (d.p_r20 <= -0.20) & (d.r1 >= 0.05), lambda d: d.r1),
        ("Circuit-type close (+9.5% at high) -> short", -1,
         lambda d: (d.r1 >= 0.095) & (d.close_loc >= 0.98), lambda d: d.r1),
        ("Breaks below 200-DMA on a -5% day -> short", -1,
         lambda d: (d.r1 <= -0.05) & (d.adj < d.sma200) & (d.p_adj >= d.p_sma200), lambda d: -d.r1),
    ]


def run_swing(f: pd.DataFrame, universe: pd.Series, out: pd.DataFrame, horizons=(5, 10, 20),
              fno: set | None = None) -> pd.DataFrame:
    base = f[universe].merge(out, on=["date", "symbol"], how="left")
    base["r20_q02"] = base.groupby("date")["r20"].transform(lambda s: s.quantile(0.02))
    bench = {k: base.groupby("date")[f"f{k}"].median() for k in horizons}
    rows = []
    for name, sign, mask, rank in swing_systems():
        sel = base[mask(base).fillna(False)].copy()
        if fno is not None:
            sel = sel[sel["symbol"].isin(fno)]
        if len(sel) < 30:
            continue
        sel["rank"] = rank(sel)
        sel = sel.sort_values(["date", "rank"], ascending=[True, False]).groupby("date").head(TOP_SWING)
        row = {"system": name, "side": "long" if sign > 0 else "short", "signals": len(sel),
               "days": sel["date"].nunique()}
        for k in horizons:
            x = sel.dropna(subset=[f"f{k}"])
            xs = sign * (x[f"f{k}"] - x["date"].map(bench[k]))
            day = xs.groupby(x["date"]).median()
            row[f"xs{k}%"] = day.mean() * 100
            row[f"t{k}"] = nw_t(day, k)
            row[f"net{k}%"] = (day.mean() - SWING_COST) * 100
            if k == 10:
                yr = day.groupby(day.index.year).mean() * 100
                row.update({"y2024": yr.get(2024, np.nan), "y2025": yr.get(2025, np.nan),
                            "y2026": yr.get(2026, np.nan)})
        rows.append(row)
    return pd.DataFrame(rows)


def outcomes(frame: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    g = frame.groupby("symbol", sort=False)
    entry = g["open"].shift(-1) * (g["adj"].shift(-1) / g["close"].shift(-1))
    out = frame[["date", "symbol"]].copy()
    for k in horizons:
        out[f"f{k}"] = g["adj"].shift(-k) / entry - 1
    return out


def main():
    frame = br.load_frame()
    f = features(frame)
    fno = set(filing_facts.fno_symbols() or [])
    liquid = (f["med_to"] >= MIN_TURNOVER_CR) & (f["close"] >= MIN_PRICE) & ~f["frozen"]
    p_liquid = (f["p_med_to"] >= MIN_TURNOVER_CR) & (f["prev_close"] >= MIN_PRICE) & ~f["frozen"]
    pd.set_option("display.width", 260, "display.max_colwidth", 64, "display.max_rows", 200)
    print(f"Data {f['date'].min().date()} -> {f['date'].max().date()}, "
          f"{f['symbol'].nunique()} symbols, F&O names {len(fno)}")

    for label, uni in (("ALL LIQUID", p_liquid), ("NON-F&O LIQUID", p_liquid & ~f["symbol"].isin(fno)),
                       ("F&O ONLY", p_liquid & f["symbol"].isin(fno))):
        r = run_intraday(f, uni).sort_values("day_t", ascending=False)
        print(f"\n=== INTRADAY (open->close, top {MAX_PER_DAY}/day, {INTRA_COST*1e4:.0f} bps cost) -- {label}")
        print(r.round(2).to_string(index=False))

    out = outcomes(f)
    s = run_swing(f, liquid, out).sort_values("t10", ascending=False)
    print(f"\n=== SWING (next open, fixed hold, top {TOP_SWING}/day, excess vs universe median; "
          f"net = minus {SWING_COST*100:.1f}% round trip) -- ALL LIQUID")
    print(s.round(2).to_string(index=False))
    s2 = run_swing(f, liquid, out, fno=fno)
    s2 = s2[s2["side"] == "short"].sort_values("t10", ascending=False)
    print("\n=== SWING shorts, F&O names only (tradable via futures)")
    print(s2.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
