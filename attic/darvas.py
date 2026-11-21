"""
Darvas box: backtest and current scan.

Rules (Nicolas Darvas, "How I Made $2,000,000 in the Stock Market"):
  1. Only stocks making a new 52-week high are considered.
  2. Box top: a high that is not exceeded on the next 3 days.
  3. Box bottom: after the top is set, the lowest low that is not broken on
     the next 3 days. If the top is exceeded first, the top resets.
  4. A box is void if price trades below its bottom before breaking out.
  5. Entry: buy-stop just above the top (filled at max(open, top)).
     Variant "vol": close above the top on >= 1.5x 50-day volume, buy next open.
  6. Stop: just below the entry box's bottom. Each new box that forms higher
     up raises the stop to its bottom. Exit at min(open, stop) when hit.
     If the entry day's low also breaks the stop, the trade is stopped that day.

Comparison: each trade's return vs the equal-weight average of the whole
universe over the same dates (excess), and a 10-slot portfolio vs holding
the universe equal-weight.

Data: Yahoo daily, split/dividend adjusted, current F&O stocks since 2010.
Today's F&O list is survivorship-biased (stocks that did well enough to be in
it); that flatters any long-only breakout system. The scan uses NSE bhavcopies.

    Run:   ~/anaconda3/bin/python attic/darvas.py 2>/dev/null
    Scan:  ~/anaconda3/bin/python attic/darvas.py scan 2>/dev/null
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

CACHE = os.path.join(HERE, "..", "data", "cache", "yahoo_daily_fo.parquet")
START = "2009-01-01"
COST = 0.0025            # delivery round trip: STT both sides + charges + slippage
SLOTS = 10
CONFIRM = 3


def universe() -> list[str]:
    from intraday_backtest import fno_symbols
    return sorted(fno_symbols())


def download(symbols: list[str]) -> pd.DataFrame:
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    import yfinance as yf
    frames = []
    for i in range(0, len(symbols), 40):
        batch = [f"{s}.NS" for s in symbols[i:i + 40]]
        h = yf.download(batch, start=START, auto_adjust=True, group_by="ticker",
                        progress=False, threads=True)
        for t in batch:
            if t not in h.columns.get_level_values(0):
                continue
            d = h[t].dropna(subset=["Close"]).reset_index()
            d.columns = [str(c).lower() for c in d.columns]
            d["symbol"] = t[:-3]
            frames.append(d[["symbol", "date", "open", "high", "low", "close", "volume"]])
        print(f"  downloaded {min(i + 40, len(symbols))}/{len(symbols)}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[(df.volume > 0) & (df.low > 0)]
    df.to_parquet(CACHE)
    return df


# --------------------------------------------------------------------------
# Box machine
# --------------------------------------------------------------------------

def boxes(d: pd.DataFrame, vol_confirm: bool = False, collect_boxes: bool = False):
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    dates = d["date"].to_numpy()
    n = len(c)
    hi250 = pd.Series(h).shift(1).rolling(250, min_periods=200).max().to_numpy()
    v50 = pd.Series(v).shift(1).rolling(50, min_periods=30).mean().to_numpy()

    state, top, top_day, bot, bot_day = "idle", np.nan, -1, np.nan, -1
    pos = None                    # dict(entry, entry_i, stop, box_top, box_bot)
    pending = None                # volume variant: enter next open
    trades, box_log = [], []

    for i in range(n):
        # --- position management first (stop checked against today's range)
        if pending is not None:
            e = o[i]
            if e > pending["stop"]:
                pos = dict(entry=e, entry_i=i, stop=pending["stop"],
                           box_top=pending["top"], box_bot=pending["stop"])
            pending = None
        if pos is not None and l[i] <= pos["stop"]:
            px = min(o[i], pos["stop"]) if i > pos["entry_i"] else pos["stop"]
            trades.append(dict(entry_date=dates[pos["entry_i"]], exit_date=dates[i], entry=pos["entry"],
                               exit=px, days=i - pos["entry_i"], box_h=pos["box_top"] / pos["box_bot"] - 1))
            pos = None

        # --- box state machine
        if state == "idle":
            if np.isfinite(hi250[i]) and h[i] >= hi250[i]:
                state, top, top_day = "top", h[i], i
        elif state == "top":
            if h[i] > top:
                top, top_day = h[i], i
            elif i - top_day >= CONFIRM:
                seg = l[top_day + 1:i + 1]
                j = int(np.argmin(seg))
                state, bot, bot_day = "bottom", seg[j], top_day + 1 + j
        elif state == "bottom":
            if h[i] > top:
                state, top, top_day = "top", h[i], i
            elif l[i] < bot:
                bot, bot_day = l[i], i
            elif i - bot_day >= CONFIRM:
                state = "box"
                if collect_boxes:
                    box_log.append(dict(top=top, bot=bot, start=dates[top_day], confirmed=dates[i]))
                if pos is not None and bot > pos["stop"]:
                    pos["stop"] = bot            # trail
        elif state == "box":
            if l[i] < bot:
                state = "idle"
            elif h[i] > top:
                if pos is None and pending is None:
                    if not vol_confirm:
                        e = max(o[i], top)
                        pos = dict(entry=e, entry_i=i, stop=bot, box_top=top, box_bot=bot)
                        if l[i] <= bot:
                            trades.append(dict(entry_date=dates[i], exit_date=dates[i], entry=e, exit=bot,
                                               days=0, box_h=top / bot - 1))
                            pos = None
                    elif c[i] > top and np.isfinite(v50[i]) and v[i] >= 1.5 * v50[i] and i + 1 < n:
                        pending = dict(stop=bot, top=top)
                state, top, top_day = "top", h[i], i

    open_pos = None
    if pos is not None:
        open_pos = dict(entry_date=dates[pos["entry_i"]], entry=pos["entry"], stop=pos["stop"], last=c[-1])
        trades.append(dict(entry_date=dates[pos["entry_i"]], exit_date=dates[-1], entry=pos["entry"],
                           exit=c[-1], days=n - 1 - pos["entry_i"], box_h=pos["box_top"] / pos["box_bot"] - 1,
                           open=True))
    state_out = dict(state=state, top=top, bot=bot, top_day=dates[top_day] if top_day >= 0 else None)
    if collect_boxes:
        return trades, box_log, state_out, open_pos
    return trades


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

def market_index(df: pd.DataFrame) -> pd.Series:
    """Equal-weight universe: average daily close-to-close return, compounded."""
    w = df.pivot(index="date", columns="symbol", values="close").sort_index()
    r = w.pct_change(fill_method=None).clip(-0.5, 1.0)
    return (1 + r.mean(axis=1).fillna(0)).cumprod()


def portfolio(trades: pd.DataFrame, idx: pd.Series, closes: pd.DataFrame) -> tuple[pd.Series, float]:
    """10 equal slots, first come first served, idle cash earns 0, marked to the
    daily close. Returns the equity curve and the average share invested."""
    t = trades.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    dates = idx.index
    cash, held, curve, invested = 1.0, [], [], []
    by_entry = {d: g for d, g in t.groupby("entry_date")}

    def value(p, d):
        px = closes.at[d, p["symbol"]] if d in closes.index else np.nan
        return p["alloc"] * (px / p["entry"] if np.isfinite(px) else 1.0)

    for d in dates:
        still = []
        for p in held:
            if p["exit_date"] <= d:
                cash += p["alloc"] * (p["exit"] / p["entry"]) * (1 - COST)
            else:
                still.append(p)
        held = still
        for _, r in by_entry.get(d, pd.DataFrame()).iterrows():
            if len(held) >= SLOTS:
                break
            equity = cash + sum(value(p, d) for p in held)
            alloc = min(cash, equity / SLOTS)
            if alloc <= 0:
                break
            cash -= alloc
            held.append(dict(symbol=r.symbol, alloc=alloc, entry=r.entry, exit=r.exit, exit_date=r.exit_date))
        pos_val = sum(value(p, d) for p in held)
        curve.append(cash + pos_val)
        invested.append(pos_val / (cash + pos_val))
    return pd.Series(curve, index=dates), float(np.mean(invested))


def cagr(s: pd.Series) -> float:
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1


def max_dd(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min())


def report(name: str, tr: pd.DataFrame, idx: pd.Series, closes: pd.DataFrame) -> None:
    tr = tr.copy()
    tr["ret"] = tr.exit / tr.entry - 1 - COST
    mkt = idx.reindex(pd.to_datetime(tr.exit_date)).to_numpy() / idx.reindex(pd.to_datetime(tr.entry_date)).to_numpy() - 1
    tr["excess"] = tr.ret - mkt
    wins = tr[tr.ret > 0]
    losses = tr[tr.ret <= 0]
    print(f"\n=== {name} ===")
    print(f"trades {len(tr)}  win rate {100 * len(wins) / len(tr):.0f}%  avg {100 * tr.ret.mean():+.2f}%  "
          f"median {100 * tr.ret.median():+.2f}%  avg win {100 * wins.ret.mean():+.1f}%  "
          f"avg loss {100 * losses.ret.mean():+.1f}%  avg hold {tr.days.mean():.0f}d (median {tr.days.median():.0f}d)")
    print(f"vs equal-weight universe over the same dates: avg excess {100 * tr.excess.mean():+.2f}%  "
          f"median {100 * tr.excess.median():+.2f}%  beat the market {100 * (tr.excess > 0).mean():.0f}%")
    print(f"biggest 5% of trades = {100 * tr.ret.nlargest(max(1, len(tr) // 20)).sum() / tr.ret.sum():.0f}% of total return points; "
          f"stopped within 5 days: {100 * (tr.days <= 5).mean():.0f}%")
    tr["year"] = pd.to_datetime(tr.entry_date).dt.year
    y = tr.groupby("year").agg(n=("ret", "size"), win=("ret", lambda x: 100 * (x > 0).mean()),
                               avg=("ret", lambda x: 100 * x.mean()), excess=("excess", lambda x: 100 * x.mean()))
    print("by entry year:  " + "  ".join(f"{yr}: {r.avg:+.1f}% ({r.excess:+.1f} vs mkt, n {r.n:.0f})"
                                         for yr, r in y.iterrows()))
    m = idx[idx.index >= pd.Timestamp("2010-01-01")]
    eq, inv = portfolio(tr, m, closes)
    print(f"10-slot portfolio 2010->: CAGR {100 * cagr(eq):.1f}%  max drawdown {100 * max_dd(eq):.0f}%  "
          f"avg invested {100 * inv:.0f}%   |   hold the universe equal-weight: CAGR {100 * cagr(m):.1f}%  "
          f"max drawdown {100 * max_dd(m):.0f}%")


def backtest() -> None:
    df = download(universe())
    print(f"{df.symbol.nunique()} stocks, {df.date.min().date()} -> {df.date.max().date()}")
    idx = market_index(df)
    closes = df.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    for name, vc in (("Darvas, buy-stop above the box", False), ("Darvas + close above box on 1.5x volume", True)):
        rows = []
        for sym, d in df.groupby("symbol", sort=False):
            for t in boxes(d.sort_values("date").reset_index(drop=True), vol_confirm=vc):
                t["symbol"] = sym
                rows.append(t)
        tr = pd.DataFrame(rows)
        tr = tr[pd.to_datetime(tr.entry_date) >= pd.Timestamp("2010-01-01")]
        report(name, tr, idx, closes)


# --------------------------------------------------------------------------
# Current scan on NSE bhavcopies
# --------------------------------------------------------------------------

def scan() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(HERE, "..", "data", "raw", "bhav_*.parquet")))[-300:]
    cols = ["symbol", "series", "date", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    df = df[df.series == "EQ"].drop(columns="series")
    df["date"] = pd.to_datetime(df["date"])
    etf = df.symbol.str.contains(r"ETF|BEES|^SETF|GOLD|SILVER|LIQUID|CASE$|NIFTY|SENSEX|IETF|MODEFENCE")
    df = df[~etf & (df.close > 0)].sort_values(["symbol", "date"])
    last_day = df.date.max()
    out = []
    for sym, d in df.groupby("symbol", sort=False):
        d = d.reset_index(drop=True)
        if len(d) < 220 or d.date.iloc[-1] != last_day:
            continue
        med_to = d.turnover.tail(20).median() / 100
        if med_to < 10 or d.close.iloc[-1] < 50:
            continue
        # bhavcopies are not split-adjusted: skip names with an impossible jump in the window
        jump = (d.open / d.close.shift(1)).iloc[-260:]
        if ((jump < 0.6) | (jump > 1.6)).any():
            continue
        trades, box_log, st, open_pos = boxes(d, collect_boxes=True)
        last = d.iloc[-1]
        v50 = d.volume.iloc[-51:-1].mean()
        broke_today = bool(open_pos is not None and pd.Timestamp(open_pos["entry_date"]) == last_day)
        if broke_today:
            kind = "BROKE OUT TODAY"
            top = box_log[-1]["top"] if box_log else open_pos["entry"]
            bot = open_pos["stop"]
        elif st["state"] == "box":
            kind = "IN A BOX"
            top, bot = st["top"], st["bot"]
        else:
            continue
        out.append(dict(symbol=sym, status=kind, close=round(last.close, 2), box_top=round(top, 2),
                        box_bottom=round(bot, 2), box_height_pct=round(100 * (top / bot - 1), 1),
                        to_top_pct=round(100 * (top / last.close - 1), 1),
                        vol_x=round(last.volume / v50, 1) if v50 else None, turnover_cr=round(med_to, 0)))
    res = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(f"as of {last_day.date()}")
    if res.empty:
        print("no boxes")
        return res
    br = res[res.status == "BROKE OUT TODAY"].sort_values("turnover_cr", ascending=False)
    print(f"\nBroke out of a Darvas box on {last_day.date()} ({len(br)}):")
    print(br.to_string(index=False))
    bx = res[res.status == "IN A BOX"].sort_values("to_top_pct").head(25)
    print(f"\nIn a confirmed box, closest to the top ({(res.status == 'IN A BOX').sum()} total, nearest 25):")
    print(bx.to_string(index=False))
    return res


def chart(sym: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    files = sorted(glob.glob(os.path.join(HERE, "..", "data", "raw", "bhav_*.parquet")))[-300:]
    cols = ["symbol", "series", "date", "open", "high", "low", "close", "volume"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    d = d[(d.symbol == sym) & (d.series == "EQ")].drop(columns="series")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    trades, box_log, _, _ = boxes(d, collect_boxes=True)
    show = d[d.date >= d.date.iloc[-1] - pd.Timedelta(days=200)]
    x0 = show.date.iloc[0]

    fig, (ax, axv) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw=dict(height_ratios=[4, 1]))
    for _, r in show.iterrows():
        col = "#16a34a" if r.close >= r.open else "#dc2626"
        ax.vlines(r.date, r.low, r.high, color=col, lw=0.8)
        ax.vlines(r.date, min(r.open, r.close), max(r.open, r.close), color=col, lw=3)
    for b in box_log:
        after = d[d.date > pd.Timestamp(b["confirmed"])]
        out = after[(after.high > b["top"]) | (after.low < b["bot"])]
        end = out.date.iloc[0] if len(out) else d.date.iloc[-1]
        if end < x0:
            continue
        start = max(pd.Timestamp(b["start"]), x0)
        broke_up = len(out) and out.high.iloc[0] > b["top"]
        ax.add_patch(Rectangle((start, b["bot"]), end - start, b["top"] - b["bot"], fill=True, alpha=0.12,
                               color="#2563eb" if broke_up else "#9ca3af", ec="#1d4ed8", lw=1.2))
    for t in trades:
        if pd.Timestamp(t["entry_date"]) >= x0:
            ax.scatter(t["entry_date"], t["entry"], marker="^", s=90, color="#1d4ed8", zorder=5)
            if not t.get("open"):
                ax.scatter(t["exit_date"], t["exit"], marker="x", s=80, color="black", zorder=5)
    axv.bar(show.date, show.volume, color="#94a3b8", width=0.8)
    ax.set_title(f"{sym}: Darvas boxes (blue = broke upward, grey = broke down / void), "
                 f"^ entry above box, x stop-out")
    ax.grid(alpha=0.2)
    path = os.path.join(HERE, "..", "data", "cache", f"darvas_{sym}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(os.path.abspath(path))
    return path


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "chart":
        chart(sys.argv[2].upper())
    elif len(sys.argv) > 1 and sys.argv[1] == "scan":
        scan()
    else:
        backtest()
