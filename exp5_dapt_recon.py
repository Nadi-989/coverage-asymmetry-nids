#!/usr/bin/env python3
"""
Experiment 5 — Reconnaissance detection on DAPT 2020  (Section 8.2, Table 15).

Reproduces:  random          ROC 0.993  F1 0.920
             Mon+Tue -> Wed  ROC 0.992  F1 0.927
             Mon+Wed -> Tue  ROC 0.871  F1 0.369
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
_p = Path(__file__).resolve()
for _c in (_p.parent, _p.parents[1], _p.parents[1] / "src"):
    if (_c / "common.py").exists():
        sys.path.insert(0, str(_c)); break
from common import RESULTS, fit_score, load_dapt_recon, subsample   # noqa: E402
from sklearn.model_selection import train_test_split

SEEDS = (0, 1, 2)
SPLITS = {
    "random":        None,
    "Mon+Tue->Wed": (["monday", "monday-pvt", "tuesday"], ["wednesday"]),
    "Mon+Wed->Tue": (["monday", "monday-pvt", "wednesday"], ["tuesday"]),
}


def main() -> None:
    df, F = load_dapt_recon()
    y = df.y.to_numpy()
    print(f"{len(df):,} flows x {len(F)} features | positive {y.mean() * 100:.1f} %\n")

    rows = []
    for name, cfg in SPLITS.items():
        for s in SEEDS:
            if cfg is None:
                tr, te = train_test_split(np.arange(len(df)), test_size=.3,
                                          random_state=s, stratify=y)
            else:
                tr = np.flatnonzero(df.day.isin(cfg[0]))
                te = np.flatnonzero(df.day.isin(cfg[1]))
                if len(tr) == 0 or len(np.unique(y[te])) < 2:
                    raise SystemExit(f"split {name}: empty train set or single-class test set "
                                     f"- check day names {sorted(df.day.unique())}")
            r = fit_score(df, F, subsample(tr, 25_000, s), te, seed=s)
            rows.append(dict(split=name, seed=s, **r))

    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "exp5_dapt_recon.csv", index=False)
    print("=== Table 15 ===")
    print(R.groupby("split")[["roc", "pr", "f1", "precision", "recall"]]
           .agg(["mean", "std"]).round(4).to_string())

    print("\n=== Table 16: activity by capture day (the mechanism) ===")
    print("  see build_dapt_recon.py output; Wednesday contains 2 network-scan")
    print("  flows against 7,614 in Tuesday — the training day cannot cover it.")
    print(f"\nwritten: {RESULTS / 'exp5_dapt_recon.csv'}")


if __name__ == "__main__":
    main()
