#!/usr/bin/env python3
"""
What actually happens to a name after it appears in the coil list?

The lists are stateless: a stock is on today's screen or it is not. So when a
name vanishes, the reader cannot tell whether it broke out (the outcome they
were waiting for), broke down (thesis dead), or merely nudged a threshold and
will be back tomorrow. This measures which of those dominates, because the
answer decides whether a lifecycle view is rich or mostly says "still
waiting".

Four questions:

  1. FLICKER. Of names that drop off the list, how many are back within a few
     sessions? This is the size of the "now what" problem, and it sets how
     much hysteresis an episode needs before a gap counts as an ending.
  2. EPISODES. How many distinct basing episodes are there really, once
     flicker gaps are bridged? `coil_days` resets on every miss, so the raw
     count overstates it.
  3. RESOLUTION. Of episodes, how many trigger up, break down, or just go
     stale without doing either?
  4. DURATION. How long does an episode run, which sets the retention window
     for a tracking view.

Read-only. This measured the problem; episodes.py is what shipped from it.

The two do not report identical numbers and should not be expected to. This
script classifies bridged runs after the fact, while episodes.py maintains
state forward one session at a time, which means a base that drops off and
requalifies later starts a fresh episode here but not there. It also reads
membership from `_base_rates.coil_mask` (rolling liquidity) rather than
`stocks.gate_flags` (the windowed liquidity the live list actually uses). The
shape of the answer is the same -- roughly half of drop-offs are flicker, a
three-session bridge is the right tolerance, half of episodes resolve inside
20 sessions -- but for anything user-facing, quote episodes.py, because that
is the code that runs. Its rates are in ui/src/evidence.js as EPISODE_RATES.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _base_rates as br
import fetch
import panel as pnl
import stocks as st

MAX_HOLD = 20     # sessions to wait for an episode to resolve
GAPS = (0, 1, 2, 3, 5)   # flicker tolerances to sweep
BREAK_DOWN = -0.07       # close this far under the entry price = broken


def load():
    paths = sorted(fetch.RAW_DIR.glob("bhav_*"))
    print(f"Loading {len(paths)} sessions "
          f"({paths[0].stem[5:]} -> {paths[-1].stem[5:]})...")
    raw = pd.concat([fetch._read_cache(p) for p in paths], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    smap = (pd.read_json(fetch.CACHE_DIR / "sector_map.json", orient="index")
              .rename_axis("symbol").reset_index())
    s, _ = pnl.build(raw.sort_values(["symbol", "date"]), smap)
    s = st.add_indicators(s)
    s["is_coil"] = br.coil_mask(s, st.CoilParams())

    def piv(col, like=None):
        w = s.pivot_table(index="date", columns="symbol", values=col).sort_index()
        return w if like is None else w.reindex(index=like.index, columns=like.columns)

    P = piv("adj")
    return {
        "P": P.values,
        "C": piv("is_coil", P).fillna(0).values > 0,
        "TRIG": piv("trigger", P).values,
        "HI": piv("adj_high", P).values,
        "VR": piv("vol_ratio", P).values,
        "dates": P.index.to_numpy(),
        "symbols": P.columns.to_numpy(),
    }


def flicker(C):
    """Of every drop-off, how often is the name back within k sessions?"""
    T = C.shape[0]
    print("=" * 78)
    print("1. FLICKER — is a drop-off a real ending or threshold noise?")
    print("=" * 78)
    drops = C[:-1] & ~C[1:]
    n_drops = int(drops.sum())
    print(f"  {n_drops:,} drop-offs across the window.\n")
    print(f"  {'back within':<16}{'count':>10}{'share of drops':>17}")
    for k in (1, 2, 3, 5, 10):
        back = np.zeros_like(drops)
        for j in range(2, k + 2):
            if j >= T:
                break
            back[: T - j] |= C[j:][: T - j]
        hit = int((drops & back).sum())
        print(f"  {str(k) + ' sessions':<16}{hit:>10,}{100 * hit / max(n_drops, 1):>16.1f}%")


def bridge(C, gap):
    """Close gaps of up to `gap` non-qualifying sessions inside a run."""
    if gap <= 0:
        return C.copy()
    out = C.copy()
    T = C.shape[0]
    for g in range(1, gap + 1):
        # A hole of length g is bridged when the run resumes g+1 rows later.
        for start in range(T - g - 1):
            hole = ~C[start + 1: start + 1 + g]
            out[start + 1: start + 1 + g] |= (
                C[start] & C[start + g + 1] & hole
            )
    return out


def episodes(C, gap):
    """(symbol_idx, start_row, end_row) for each bridged run."""
    B = bridge(C, gap)
    eps = []
    for j in range(B.shape[1]):
        col = B[:, j]
        if not col.any():
            continue
        d = np.diff(col.astype(np.int8), prepend=0, append=0)
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
            eps.append((j, a, b))
    return eps


def resolve(eps, G):
    """
    Classify each episode by what happened first, from its entry row.

    The trigger is the 20-day high ON THE ENTRY DAY, frozen there -- letting
    it drift with the rolling high would move the target every session and
    no episode would ever count as triggered.
    """
    P, TRIG, HI, VR = G["P"], G["TRIG"], G["HI"], G["VR"]
    T = P.shape[0]
    out = []
    for j, a, b in eps:
        entry, trig = P[a, j], TRIG[a, j]
        if not np.isfinite(entry) or not np.isfinite(trig):
            continue
        end = min(a + MAX_HOLD, T - 1)
        kind, when = "stale", None
        for t in range(a + 1, end + 1):
            if np.isfinite(HI[t, j]) and P[t, j] > trig:
                heavy = np.isfinite(VR[t, j]) and VR[t, j] >= 1.5
                kind, when = ("triggered_vol" if heavy else "triggered"), t - a
                break
            if np.isfinite(P[t, j]) and P[t, j] / entry - 1 <= BREAK_DOWN:
                kind, when = "broke_down", t - a
                break
        out.append({"len": b - a + 1, "kind": kind, "when": when})
    return pd.DataFrame(out)


def main() -> None:
    G = load()
    C = G["C"]
    print(f"\n  grid: {C.shape[0]} sessions x {C.shape[1]:,} symbols, "
          f"{C.sum():,} qualifying (symbol, day) pairs\n")

    flicker(C)

    print()
    print("=" * 78)
    print("2. EPISODES — how many real bases, once flicker is bridged?")
    print("=" * 78)
    print(f"  {'gap bridged':<16}{'episodes':>10}{'median len':>13}{'mean len':>11}")
    per_gap = {}
    for gap in GAPS:
        eps = episodes(C, gap)
        lens = np.array([b - a + 1 for _, a, b in eps])
        per_gap[gap] = eps
        print(f"  {str(gap) + ' sessions':<16}{len(eps):>10,}"
              f"{np.median(lens):>13.0f}{lens.mean():>11.1f}")

    print()
    print("=" * 78)
    print(f"3. RESOLUTION — what happens within {MAX_HOLD} sessions of entry")
    print("=" * 78)
    print("  Using a 3-session bridge, so a one-day wobble does not end a base.")
    r = resolve(per_gap[3], G)
    n = len(r)
    print(f"  {n:,} episodes with a usable entry.\n")
    print(f"  {'outcome':<26}{'count':>9}{'share':>9}{'med sessions':>14}")
    labels = {
        "triggered_vol": "closed through, heavy vol",
        "triggered": "closed through, light vol",
        "broke_down": f"broke down ({BREAK_DOWN:.0%})",
        "stale": "neither — still basing",
    }
    for k, lab in labels.items():
        sub = r[r["kind"] == k]
        med = sub["when"].median()
        med_s = f"{med:.0f}" if np.isfinite(med) else "—"
        print(f"  {lab:<26}{len(sub):>9,}{100 * len(sub) / max(n, 1):>8.1f}%{med_s:>14}")

    trig = r[r["kind"].str.startswith("triggered")]
    print(f"\n  Any trigger: {100 * len(trig) / max(n, 1):.1f}% of episodes, "
          f"median {trig['when'].median():.0f} sessions to fire.")


if __name__ == "__main__":
    main()
