"""
Stress tests for the gap-fade strategies found by intraday_backtest.py.

Run:  ~/anaconda3/bin/python attic/intraday_robustness.py 2>/dev/null
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import intraday_backtest as bt


def gap_down_fade(d, g=0.02):
    m = (d.gap <= -g) & (d.gap > -0.09)
    return m, bt.long_oc(d), -d.gap


def gap_up_fade(d, g=0.02):
    m = (d.gap >= g) & (d.gap < 0.09)
    return m, bt.short_oc(d), d.gap


def downtrend_pop(d, stop=None):
    m = (d.gap >= 0.02) & (d.gap < 0.09) & (d.prev_close < d.sma50)
    return m, bt.short_oc(d, stop=stop), d.gap


def uptrend_dip(d, stop=None):
    m = (d.gap <= -0.02) & (d.gap > -0.09) & (d.prev_close > d.sma50)
    return m, bt.long_oc(d, stop=stop), -d.gap


def trades(period, fn, cost, per_day):
    m, r, rank = fn(period)
    m = m.fillna(False)
    sel = period[m].assign(ret=r[m] - cost, rank=rank[m])
    sel = sel[sel.ret.notna()]
    return (sel.sort_values(["date", "rank"], ascending=[True, False])
               .groupby("date").head(per_day))


def stats(sel, days, trim_top=0.0):
    if trim_top:
        sel = sel[sel.ret < sel.ret.quantile(1 - trim_top)]
    daily = sel.groupby("date")["ret"].mean().reindex(days, fill_value=0.0)
    eq = (1 + daily).cumprod()
    return {
        "trades": len(sel),
        "win%": round((sel.ret > 0).mean() * 100, 1),
        "avg_trade%": round(sel.ret.mean() * 100, 2),
        "total%": round((eq.iloc[-1] - 1) * 100, 1),
        "sharpe": round(daily.mean() / daily.std() * np.sqrt(252), 2),
        "maxDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
    }


def main():
    df = bt.features(bt.load())
    fno = bt.fno_symbols()
    end = df.date.max() + pd.Timedelta(days=1)
    base = (df.date >= bt.TEST_START) & (df.date < end) & (df.prev_close >= bt.MIN_PRICE) & ~df.frozen
    days = df.loc[base, "date"].drop_duplicates().sort_values()

    strats = {
        "Gap-down fade (buy gap <= -2%)": gap_down_fade,
        "Gap-up fade (short gap >= +2%)": gap_up_fade,
        "Downtrend pop (short gap >= +2%, below SMA50)": downtrend_pop,
        "Uptrend dip (buy gap <= -2%, above SMA50)": uptrend_dip,
    }

    rows = []
    for name, fn in strats.items():
        for to_floor in (10, 50, 200):
            for universe in ("all", "non-F&O"):
                u = base & (df.med_to >= to_floor)
                if universe == "non-F&O":
                    u &= ~df.symbol.isin(fno)
                period = df[u]
                for cost in (0.0010, 0.0025, 0.0040):
                    sel = trades(period, fn, cost, bt.MAX_PER_DAY)
                    rows.append({"strategy": name, "turnover>=cr": to_floor,
                                 "universe": universe, "cost_bps": int(cost * 1e4),
                                 **stats(sel, days)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_rows", 500)
    print("=== Sensitivity: liquidity floor x universe x round-trip cost (last 6 months) ===")
    print(res.to_string(index=False))

    # Remove the best 5% of trades: does the edge survive without the outliers?
    print("\n=== Without the best 5% of trades (Rs 10 cr floor, all stocks, 10 bps) ===")
    period = df[base & (df.med_to >= 10)]
    for name, fn in strats.items():
        sel = trades(period, fn, 0.0010, bt.MAX_PER_DAY)
        print(f"{name:50s}", stats(sel, days, trim_top=0.05))

    # Market-neutral pair: long gap-downs + short gap-ups on the same days.
    print("\n=== Combined book: 3 longs (gap-down fade) + 3 shorts (gap-up fade) per day ===")
    for to_floor in (10, 50):
        period = df[base & (df.med_to >= to_floor)]
        for cost in (0.0010, 0.0025):
            l = trades(period, gap_down_fade, cost, 3)
            s = trades(period, gap_up_fade, cost, 3)
            both = pd.concat([l, s])
            print(f"floor Rs {to_floor} cr, cost {int(cost*1e4)} bps:", stats(both, days))

    # Stops on the two trend-filtered versions.
    print("\n=== Stops on trend-filtered fades (Rs 10 cr floor, 10 bps) ===")
    for name, fn in (("Downtrend pop", downtrend_pop), ("Uptrend dip", uptrend_dip)):
        for stop in (None, 0.015, 0.02, 0.03):
            sel = trades(period, lambda d: fn(d, stop=stop), 0.0010, bt.MAX_PER_DAY)
            print(f"{name:15s} stop={stop}:", stats(sel, days))

    # Month-by-month for the top strategy, to see it is not one lucky month.
    print("\n=== Month by month (Rs 10 cr floor, 10 bps, sum of daily returns %) ===")
    period = df[base & (df.med_to >= 10)]
    tab = {}
    for name, fn in strats.items():
        sel = trades(period, fn, 0.0010, bt.MAX_PER_DAY)
        daily = sel.groupby("date")["ret"].mean().reindex(days, fill_value=0.0)
        tab[name[:28]] = (daily.groupby(daily.index.to_period("M")).sum() * 100).round(1)
    print(pd.DataFrame(tab).to_string())


if __name__ == "__main__":
    main()
