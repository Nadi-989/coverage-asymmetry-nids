#!/usr/bin/env python3
"""
Experiment 6 — Coverage-invariant features  (Section 9, Tables 17-19).

Defines eight dimensionless ratios (Equation 11), validates that they are
near-identically dispersed across clients, and tests significance with a
paired Wilcoxon signed-rank test.

Reproduces:  spread ratios 1.06-2.20x (raw: 6.33-7.11x)
             PR-AUC +0.054, Wilcoxon p = 0.012, d_z = 0.71, n = 18
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
_p = Path(__file__).resolve()
for _c in (_p.parent, _p.parents[1], _p.parents[1] / "src"):
    if (_c / "common.py").exists():
        sys.path.insert(0, str(_c)); break
from common import (EPS, RESULTS, TOOLS, fit_score, load_doh,       # noqa: E402
                    subsample)

SEEDS = range(6)


def add_invariant(d: pd.DataFrame) -> list[str]:
    """Equation 11 — eight dimensionless ratios."""
    d["RATE_ASYM"]  = np.log1p(d.FlowSentRate) - np.log1p(d.FlowReceivedRate)
    d["BYTE_ASYM"]  = np.log1p(d.FlowBytesSent) - np.log1p(d.FlowBytesReceived)
    d["RT_STRUCT"]  = d.ResponseTimeTimeMedian / (d.PacketTimeMedian + EPS)
    d["RT_DISP"]    = (d.ResponseTimeTimeStandardDeviation
                       / (d.PacketTimeStandardDeviation + EPS))
    d["LEN_MODAL"]  = d.PacketLengthMode / (d.PacketLengthMean + EPS)
    d["LEN_CENTR"]  = d.PacketLengthMedian / (d.PacketLengthMean + EPS)
    d["TIME_CENTR"] = d.PacketTimeMedian / (d.PacketTimeMean + EPS)
    d["LEN_SHAPE"]  = (d.PacketLengthSkewFromMode
                       / (d.PacketLengthCoefficientofVariation.abs() + EPS))
    cols = ["RATE_ASYM", "BYTE_ASYM", "RT_STRUCT", "RT_DISP",
            "LEN_MODAL", "LEN_CENTR", "TIME_CENTR", "LEN_SHAPE"]
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return cols


def main() -> None:
    df, BASE = load_doh()
    CI = add_invariant(df)

    # ---- Table 17: design validation -------------------------------------
    ch, ff = df[df.grp == "chrome"], df[df.grp == "firefox"]
    print("=== Table 17: cross-client spread ratio (near 1 = invariant) ===")
    for c in ["FlowSentRate", "FlowReceivedRate"] + CI:
        a = ch[c].quantile(.95) - ch[c].quantile(.05)
        b = ff[c].quantile(.95) - ff[c].quantile(.05)
        tag = " (raw)" if c.startswith("Flow") else ""
        print(f"  {c + tag:26} {b / (a + EPS):>6.2f}x")

    # ---- Table 18: paired significance test ------------------------------
    rows = []
    for tool in TOOLS:
        tr_all = np.flatnonzero(
            df.grp.isin([t for t in TOOLS if t != tool]) | (df.grp == "chrome"))
        te = np.flatnonzero((df.grp == tool) | (df.grp == "firefox"))
        for s in SEEDS:
            tr = subsample(tr_all, 25_000, s)
            rows.append(dict(tool=tool, seed=s, feats="baseline",
                             **fit_score(df, BASE, tr, te, seed=s)))
            rows.append(dict(tool=tool, seed=s, feats="invariant",
                             **fit_score(df, CI, tr, te, seed=s)))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "exp6_invariant_features.csv", index=False)

    print(f"\n=== Table 18: paired comparison, hard direction, n = {len(SEEDS) * 3} ===")
    for met in ("roc", "pr"):
        b = R[R.feats == "baseline"].sort_values(["tool", "seed"])[met].to_numpy()
        c = R[R.feats == "invariant"].sort_values(["tool", "seed"])[met].to_numpy()
        d = c - b
        stat, p = wilcoxon(c, b)
        dz = d.mean() / (d.std(ddof=1) + EPS)
        print(f"  {met.upper():7} baseline {b.mean():.4f} +- {b.std(ddof=1):.4f} | "
              f"invariant {c.mean():.4f} +- {c.std(ddof=1):.4f} | "
              f"delta {d.mean():+.4f}  p={p:.4f}  dz={dz:+.2f}  "
              f"wins {int((d > 0).sum())}/{len(d)}")
    print(f"\nwritten: {RESULTS / 'exp6_invariant_features.csv'}")


if __name__ == "__main__":
    main()
