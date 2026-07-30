#!/usr/bin/env python3
"""
Experiment 2 — Sample size is not the explanation  (Section 6.3, Table 11).

Chrome contributes 3,311 benign flows and Firefox 14,090. We cap benign
training at the smaller value in BOTH directions and show the gap persists.

Reproduces:  Firefox -> Chrome  0.996
             Chrome  -> Firefox 0.661
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
_p = Path(__file__).resolve()
for _c in (_p.parent, _p.parents[1], _p.parents[1] / "src"):
    if (_c / "common.py").exists():
        sys.path.insert(0, str(_c)); break
from common import BROWSERS, RESULTS, TOOLS, fit_score, load_doh, subsample  # noqa: E402

SEEDS = (0, 1, 2)


def main() -> None:
    df, F = load_doh()
    cap = min((df.grp == b).sum() for b in BROWSERS)
    print(f"benign pool sizes: "
          f"{ {b: int((df.grp == b).sum()) for b in BROWSERS} }")
    print(f"capping benign training at n = {cap:,} in both directions\n")

    rows = []
    for br in BROWSERS:                       # br = held-out (test) client
        other = [b for b in BROWSERS if b != br][0]
        pool = np.flatnonzero(df.grp == other)
        for tool in TOOLS:
            mtr = np.flatnonzero(df.grp.isin([t for t in TOOLS if t != tool]))
            te = np.flatnonzero((df.grp == tool) | (df.grp == br))
            for s in SEEDS:
                tr = np.concatenate([subsample(mtr, 25_000, s),
                                     subsample(pool, cap, s)])
                r = fit_score(df, F, tr, te, seed=s)
                rows.append(dict(direction=f"{other} -> {br}", tool=tool,
                                 seed=s, **r))

    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "exp2_matched_size.csv", index=False)
    print("=== Table 11: matched-sample comparison ===")
    print(R.groupby("direction")[["roc", "pr"]]
           .agg(["mean", "std"]).round(4).to_string())
    print(f"\nwritten: {RESULTS / 'exp2_matched_size.csv'}")


if __name__ == "__main__":
    main()
