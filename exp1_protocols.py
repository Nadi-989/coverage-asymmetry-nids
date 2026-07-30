#!/usr/bin/env python3
"""
Experiment 1 — Evaluation protocols P1, P2, P3  (Section 6.2, Tables 9-10).

Reproduces:
  P1 random               ROC 1.000
  P2 leave-tool-out       ROC 0.999
  P3 leave-tool-+-client  ROC 0.800 +- 0.203   (0.998 Chrome / 0.601 Firefox)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
_p = Path(__file__).resolve()
for _c in (_p.parent, _p.parents[1], _p.parents[1] / "src"):
    if (_c / "common.py").exists():
        sys.path.insert(0, str(_c)); break
from common import (BROWSERS, RESULTS, TOOLS, fit_score, load_doh,   # noqa: E402
                    p3_splits, subsample)
from sklearn.model_selection import train_test_split

N_TRAIN, SEEDS = 25_000, (0, 1)


def main() -> None:
    df, F = load_doh()
    y = df.y.to_numpy()
    rows = []

    # ---- P1: stratified random -------------------------------------------
    tr, te = train_test_split(np.arange(len(df)), test_size=.3,
                              random_state=42, stratify=y)
    for s in SEEDS:
        r = fit_score(df, F, subsample(tr, N_TRAIN, s), te, seed=s)
        rows.append(dict(protocol="P1_random", tool="-", browser="-", seed=s, **r))

    # ---- P2: leave-one-tool-out ------------------------------------------
    ben = np.flatnonzero(y == 0)
    btr, bte = train_test_split(ben, test_size=.3, random_state=42,
                                stratify=df.grp.values[ben])
    for tool in TOOLS:
        mtr = np.flatnonzero(df.grp.isin([t for t in TOOLS if t != tool]))
        mte = np.flatnonzero(df.grp == tool)
        for s in SEEDS:
            tr = np.concatenate([subsample(mtr, N_TRAIN, s), btr])
            r = fit_score(df, F, tr, np.concatenate([mte, bte]), seed=s)
            rows.append(dict(protocol="P2_leave_tool", tool=tool,
                             browser="-", seed=s, **r))

    # ---- P3: leave-one-tool-and-one-client-out ---------------------------
    for tool, br, tr_all, te in p3_splits(df):
        for s in SEEDS:
            r = fit_score(df, F, subsample(tr_all, N_TRAIN, s), te, seed=s)
            rows.append(dict(protocol="P3_leave_tool_client", tool=tool,
                             browser=br, seed=s, **r))

    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "exp1_protocols.csv", index=False)

    print("\n=== Table 9: performance by protocol ===")
    print(R.groupby("protocol")[["roc", "pr"]].agg(["mean", "std"]).round(4).to_string())
    print("\n=== Table 10: P3 decomposed ===")
    p3 = R[R.protocol == "P3_leave_tool_client"]
    print(p3.pivot_table(index="tool", columns="browser",
                         values="roc", aggfunc="mean").round(4).to_string())
    print("\n=== Precision / recall decomposition (Figure 6) ===")
    print(R.groupby("protocol")[["precision", "recall"]].mean().round(4).to_string())
    print(f"\nwritten: {RESULTS / 'exp1_protocols.csv'}")


if __name__ == "__main__":
    main()
