"""
Which stocks move with which commodity — measured, not assumed.

Weekly returns (Friday to Friday) so NSE's 15:30 close and the later US
futures settle land in the same bucket. Each stock's weekly return is
regressed on the Nifty and one driver at a time:

    r_stock = a + b_mkt * r_nifty + b_drv * r_driver

b_drv is the sensitivity after the market's own move is taken out. Without
the market term crude, copper and the Nifty all rise and fall with risk
appetite and every stock looks "commodity-linked".

Commodity prices are converted to rupees (USD price × USD/INR) because that
is the price an Indian company pays or receives. FX and yields are used as is.

With ~1,500 stocks per driver, strong-looking links appear by chance alone.
`reliable` keeps only links that hold in both halves of the window, and
`placebo` measures how many still pass when the driver is shifted in time,
which is the false-positive rate to compare against.

    python commodities.py           # research report (with placebo), cache/commodity_links.csv
    python commodities.py export    # writes data/commodity_links.json for the product

The export needs the local multi-year bhavcopy cache, so it is run by hand
(links change slowly) and the JSON ships with the backend.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import fetch

warnings.filterwarnings("ignore")

# (key, label, yfinance ticker, convert to INR)
DRIVERS = [
    ("BRENT",   "Brent crude",        "BZ=F",     True),
    ("WTI",     "WTI crude",          "CL=F",     True),
    ("NATGAS",  "Natural gas",        "NG=F",     True),
    ("GASOLINE","Gasoline",           "RB=F",     True),
    ("GOLD",    "Gold",               "GC=F",     True),
    ("SILVER",  "Silver",             "SI=F",     True),
    ("PLATINUM","Platinum",           "PL=F",     True),
    ("COPPER",  "Copper",             "HG=F",     True),
    ("ALUM",    "Aluminium",          "ALI=F",    True),
    ("STEEL",   "Steel HRC (US)",     "HRC=F",    True),
    ("SUGAR",   "Raw sugar",          "SB=F",     True),
    ("COTTON",  "Cotton",             "CT=F",     True),
    ("COFFEE",  "Coffee",             "KC=F",     True),
    ("COCOA",   "Cocoa",              "CC=F",     True),
    ("WHEAT",   "Wheat",              "ZW=F",     True),
    ("CORN",    "Corn",               "ZC=F",     True),
    ("SOYBEAN", "Soybeans",           "ZS=F",     True),
    ("SOYOIL",  "Soybean oil",        "ZL=F",     True),
    ("RICE",    "Rough rice",         "ZR=F",     True),
    ("USDINR",  "USD/INR",            "USDINR=X", False),
    ("EURINR",  "EUR/INR",            "EURINR=X", False),
    ("DXY",     "US dollar index",    "DX-Y.NYB", False),
    ("US10Y",   "US 10Y yield",       "^TNX",     False),
]
LABELS = {k: l for k, l, _, _ in DRIVERS}
TICKERS = {k: t for k, _, t, _ in DRIVERS}

# Drivers shown in the product: (key, group, unit, news query). The rest of
# DRIVERS stay in the research run but showed nothing stable, or matter little
# to Indian listed companies.
DISPLAY = [
    ("BRENT",  "Energy",          "USD/bbl", '("crude oil" OR "oil prices" OR OPEC OR Brent)'),
    ("NATGAS", "Energy",          "USD/MMBtu", '("natural gas prices" OR "LNG prices" OR "gas supply")'),
    ("GOLD",   "Precious metals", "USD/oz",  '("gold price" OR "gold prices")'),
    ("SILVER", "Precious metals", "USD/oz",  '("silver price" OR "silver prices")'),
    ("COPPER", "Base metals",     "USD/lb",  '("copper price" OR "copper prices" OR "copper supply")'),
    ("ALUM",   "Base metals",     "USD/t",   '("aluminium prices" OR "aluminum prices" OR "aluminium price" OR "aluminum price")'),
    ("SUGAR",  "Agri",            "USc/lb",  '("sugar prices" OR "sugar exports" OR "ethanol blending" OR "sugar production")'),
    ("COTTON", "Agri",            "USc/lb",  '("cotton prices" OR "cotton crop" OR "cotton imports")'),
    ("USDINR", "Currency & rates", "INR",    '(rupee (dollar OR forex OR RBI)) sourcecountry:IN'),
    ("US10Y",  "Currency & rates", "%",      '("Treasury yields" OR "Federal Reserve" OR "Fed rate")'),
]

# Industries with a known, argued-for exposure to a driver. A stable link from
# a stock in one of these is shown even without an industry peer confirming
# it (IndiGo is the only large listed airline). Anything outside this list
# needs a peer. Keys are NSE basic_industry names.
PRIORS = {
    "BRENT":  {"Refineries & Marketing", "Oil Exploration & Production", "Airline", "Paints",
               "Tyres & Rubber Products", "Specialty Chemicals", "Lubricants",
               "LPG/CNG/PNG/LNG Supplier", "Petrochemicals"},
    "NATGAS": {"LPG/CNG/PNG/LNG Supplier", "Fertilizers"},
    "GOLD":   {"Gems Jewellery And Watches", "Non Banking Financial Company (NBFC)"},
    "SILVER": {"Zinc", "Diversified Metals", "Gems Jewellery And Watches"},
    "COPPER": {"Copper", "Diversified Metals", "Aluminium", "Cables - Electricals"},
    "ALUM":   {"Aluminium", "Diversified Metals"},
    "SUGAR":  {"Sugar"},
    "COTTON": {"Garments & Apparels", "Other Textile Products"},
    "USDINR": {"Computers - Software & Consulting", "IT Enabled Services", "Pharmaceuticals"},
    "US10Y":  set(),
}
LINKS_FILE = fetch.DATA_DIR / "commodity_links.json"
MARKET = "^NSEI"
MIN_WEEKS = 50
MIN_TURN_LACS = 100.0
MIN_PRICE = 10.0


def load_drivers(start, end) -> pd.DataFrame:
    """Weekly driver returns, commodities in INR."""
    import yfinance as yf
    tickers = [t for _, _, t, _ in DRIVERS] + [MARKET]
    px = yf.download(tickers, start=start, end=end, progress=False,
                     auto_adjust=True)["Close"]
    wk = px.resample("W-FRI").last()
    out = pd.DataFrame(index=wk.index)
    inr = wk["USDINR=X"]
    for key, _, t, to_inr in DRIVERS:
        if t not in wk or wk[t].notna().sum() < MIN_WEEKS:
            continue
        s = wk[t] * inr if to_inr else wk[t]
        # Yields move in levels; a percent change of a yield is not meaningful.
        out[key] = s.diff() if key == "US10Y" else s.pct_change()
    out["_MKT"] = wk[MARKET].pct_change()
    return out


def load_stock_weekly() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly stock returns from the cached bhavcopies (corporate-action safe)."""
    raw = pd.concat([fetch._read_cache(p) for p in sorted(fetch.RAW_DIR.glob("bhav_*.*"))],
                    ignore_index=True)
    raw = raw[raw["series"].astype(str).str.strip().str.upper() == "EQ"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = fetch.drop_repeat_sessions(raw, verbose=False)
    # NSE's prev_close is adjusted for splits and bonuses, so this ratio is too.
    raw["ret"] = raw["close"] / raw["prev_close"] - 1.0
    raw = raw[raw["ret"].abs() < 0.5]

    # ETFs track the commodity by construction and would crowd every list.
    eq_list = fetch.CACHE_DIR / "equity_list.csv"
    if eq_list.exists():
        companies = set(pd.read_csv(eq_list)["symbol"].astype(str).str.strip())
        raw = raw[raw["symbol"].isin(companies)]

    recent = raw[raw["date"] >= raw["date"].max() - pd.Timedelta(days=90)]
    liq = recent.groupby("symbol").agg(turn=("turnover", "median"), px=("close", "last"))
    keep = liq.index[(liq["turn"] >= MIN_TURN_LACS) & (liq["px"] >= MIN_PRICE)]
    raw = raw[raw["symbol"].isin(keep)]

    daily = raw.pivot_table(index="date", columns="symbol", values="ret")
    weekly = (1 + daily).resample("W-FRI").prod(min_count=1) - 1
    smap = pd.read_json(fetch.CACHE_DIR / "sector_map.json", orient="index")
    info = pd.DataFrame({"industry": smap["basic_industry"]}).reindex(weekly.columns)
    return weekly, info


def sensitivities(weekly: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, driver): beta, t-stat, weeks used."""
    idx = weekly.index.intersection(drivers.index)
    W, D = weekly.loc[idx], drivers.loc[idx]
    mkt = D["_MKT"]
    rows = []
    for key in [c for c in D.columns if c != "_MKT"]:
        drv = D[key]
        for sym in W.columns:
            y = W[sym]
            m = y.notna() & mkt.notna() & drv.notna()
            n = int(m.sum())
            if n < MIN_WEEKS:
                continue
            X = np.column_stack([np.ones(n), mkt[m].values, drv[m].values])
            yy = y[m].values
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            resid = yy - X @ beta
            s2 = resid @ resid / (n - 3)
            cov = s2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(cov[2, 2])
            rows.append((sym, key, beta[2], beta[2] / se if se > 0 else np.nan, n))
    return pd.DataFrame(rows, columns=["symbol", "driver", "beta", "t", "weeks"])


T_FULL = 3.0
T_HALF = 1.0


def reliable(weekly: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """
    Links that hold over the whole window and in each half separately.

    Full-period |t| >= T_FULL, and the same sign with |t| >= T_HALF in both
    halves. A coincidence rarely survives being split in two.
    """
    idx = weekly.index.intersection(drivers.index)
    mid = idx[len(idx) // 2]
    full = sensitivities(weekly.loc[idx], drivers.loc[idx])
    h1 = sensitivities(weekly.loc[idx[idx < mid]], drivers.loc[idx[idx < mid]])
    h2 = sensitivities(weekly.loc[idx[idx >= mid]], drivers.loc[idx[idx >= mid]])
    key = ["symbol", "driver"]
    m = (full.merge(h1[key + ["t"]].rename(columns={"t": "t1"}), on=key)
             .merge(h2[key + ["t"]].rename(columns={"t": "t2"}), on=key))
    sign = np.sign(m["t"])
    m["reliable"] = ((m["t"].abs() >= T_FULL)
                     & (m["t1"] * sign >= T_HALF) & (m["t2"] * sign >= T_HALF))
    return m


def confirm(rel: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """
    A reliable link is confirmed when another stock in the same industry has a
    reliable link to the same driver in the same direction. Coincidences are
    scattered across industries; real exposures cluster.
    """
    r = rel.join(info, on="symbol")
    r["side"] = np.sign(r["t"])
    peers = (r[r["reliable"] & r["industry"].notna()]
             .groupby(["driver", "industry", "side"])["symbol"].transform("size"))
    r["confirmed"] = False
    r.loc[peers.index, "confirmed"] = peers >= 2
    return r


def placebo(weekly: pd.DataFrame, drivers: pd.DataFrame, info: pd.DataFrame,
            shifts=(17, 31, 47)) -> pd.DataFrame:
    """
    Average reliable and confirmed counts per driver when the driver is
    rotated in time.

    A circular shift keeps each series' own behaviour but breaks its alignment
    with stock returns, so anything that still passes is a false positive.
    """
    counts = []
    for k in shifts:
        d = drivers.copy()
        cols = [c for c in d.columns if c != "_MKT"]
        d[cols] = np.roll(d[cols].values, k, axis=0)
        r = confirm(reliable(weekly, d), info)
        counts.append(r.groupby("driver")[["reliable", "confirmed"]].sum())
    return sum(counts) / len(counts)


def export(conf: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    """
    Write the links the product shows: stable, and either confirmed by an
    industry peer or in an industry with a known exposure (PRIORS).
    """
    import json
    known = set(pd.read_json(fetch.CACHE_DIR / "sector_map.json", orient="index")["basic_industry"].dropna())
    for key, inds in PRIORS.items():
        missing = inds - known
        assert not missing, f"PRIORS[{key}] has unknown industries: {missing}"

    links = {}
    for key, *_ in DISPLAY:
        g = conf[(conf["driver"] == key) & conf["reliable"]].copy()
        g["prior"] = g["industry"].isin(PRIORS.get(key, set()))
        g = g[g["confirmed"] | g["prior"]].sort_values("t", key=lambda s: -s.abs())
        links[key] = [{
            "symbol": r.symbol,
            "industry": r.industry if isinstance(r.industry, str) else None,
            "beta": round(float(r.beta), 3),
            "t": round(float(r.t), 1),
            "basis": "peers" if r.confirmed else "industry",
        } for r in g.itertuples()]
    out = {
        "computed_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "window": {"start": f"{weekly.index.min():%Y-%m-%d}", "end": f"{weekly.index.max():%Y-%m-%d}",
                   "weeks": int(len(weekly))},
        "method": "Weekly returns, Nifty move removed; |t| >= 3 over the window and the same "
                  "direction in both halves; confirmed by an industry peer or an expected industry exposure.",
        "links": links,
    }
    LINKS_FILE.write_text(json.dumps(out, indent=1))
    return out


def report(conf: pd.DataFrame, fake: pd.DataFrame, top: int = 8) -> None:
    labels = LABELS
    print(f"\nPlacebo totals across all drivers: reliable ≈{fake['reliable'].sum():.1f}, "
          f"confirmed ≈{fake['confirmed'].sum():.1f}")
    for key in [k for k, *_ in DRIVERS if k in set(conf["driver"])]:
        g = conf[(conf["driver"] == key) & conf["confirmed"]]
        n_rel = int(conf[(conf["driver"] == key) & conf["reliable"]].shape[0])
        fk = fake.loc[key] if key in fake.index else pd.Series({"reliable": 0, "confirmed": 0})
        print(f"\n=== {labels[key]}: {len(g)} confirmed (placebo ≈{fk['confirmed']:.1f}), "
              f"{n_rel} reliable (placebo ≈{fk['reliable']:.1f})")
        for side, part in (("with", g[g["t"] > 0].nlargest(top, "t")),
                           ("against", g[g["t"] < 0].nsmallest(top, "t"))):
            if len(part):
                print(f"  moves {side:7}: " + ", ".join(
                    f"{r.symbol} [{r.industry}] ({r.beta:+.2f})" for r in part.itertuples()))


if __name__ == "__main__":
    weekly, info = load_stock_weekly()
    drivers = load_drivers(weekly.index.min() - pd.Timedelta(days=10),
                           weekly.index.max() + pd.Timedelta(days=3))
    print(f"{weekly.shape[1]} liquid stocks, {len(weekly)} weeks "
          f"({weekly.index.min():%Y-%m-%d} → {weekly.index.max():%Y-%m-%d}), "
          f"{drivers.shape[1] - 1} drivers")
    import sys
    conf = confirm(reliable(weekly, drivers), info)
    if "export" in sys.argv[1:]:
        out = export(conf, weekly)
        print({k: len(v) for k, v in out["links"].items()})
    else:
        fake = placebo(weekly, drivers, info)
        report(conf, fake)
        conf.to_csv(fetch.CACHE_DIR / "commodity_links.csv", index=False)
