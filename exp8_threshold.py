#!/usr/bin/env python3
"""
Experiment 8 — Coverage is a threshold phenomenon  (Section 11, Table 23).

A controlled experiment in which coverage is the only manipulated variable.
Firefox benign flows are split into two disjoint halves; the test set is fixed
as half B plus the held-out Iodine tunnel; n flows are drawn from half A and
added to an otherwise Chrome-only benign training set.

Reproduces:  n=0   ROC 0.609 +- 0.038
             n=1   ROC 0.849 +- 0.182
             n=5   ROC 0.991 +- 0.005   <- variance collapses here
             n=100 ROC 0.996 +- 0.003
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (RESULTS, containment, fit_score, load_doh,      # noqa: E402
                    subsample, top_dispersed)

N_VALUES = [0, 1, 2, 3, 5, 10, 25, 50, 100]
SEEDS = range(8)


def main() -> None:
    df, F = load_doh()
    df[F] = df[F].fillna(df[F].median())

    rs = np.random.RandomState(0)
    ff_idx = np.array(df.index[df.grp == "firefox"]); rs.shuffle(ff_idx)
    ffA, ffB = ff_idx[:len(ff_idx) // 2], ff_idx[len(ff_idx) // 2:]
    chrome = np.array(df.index[df.grp == "chrome"])
    mal_tr = np.array(df.index[df.grp.isin(["dns2tcp", "dnscat2"])])
    mal_te = np.array(df.index[df.grp == "iodin"])
    S = top_dispersed(df[df.grp == "chrome"], df[df.grp == "firefox"], F)

    print("Firefox is split into two disjoint halves; test is always half B.")
    print(f"  half A (training pool): {len(ffA):,}   half B (test): {len(ffB):,}\n")
    print(f"{'n':>5} {'coverage':>10} {'ROC-AUC':>20} {'PR-AUC':>20}")

    rows = []
    for n in N_VALUES:
        acc = []
        for s in SEEDS:
            r = np.random.RandomState(100 + s)
            extra = r.choice(ffA, n, replace=False) if n else np.array([], int)
            ben = np.concatenate([chrome, extra]).astype(int)
            tr = np.concatenate([subsample(mal_tr, 20_000, s), ben])
            te = np.concatenate([mal_te, ffB])
            acc.append(fit_score(df, F, tr, te, seed=s))
            if s == 0:
                cov = containment(df.loc[ffB], df.loc[ben], S)
        roc = np.array([a["roc"] for a in acc])
        pr = np.array([a["pr"] for a in acc])
        print(f"{n:>5} {cov:>10.3f} {roc.mean():>12.4f} +- {roc.std():.4f}"
              f" {pr.mean():>12.4f} +- {pr.std():.4f}")
        rows.append(dict(n=n, coverage=cov, roc=roc.mean(), roc_sd=roc.std(),
                         pr=pr.mean(), pr_sd=pr.std()))

    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "exp8_threshold.csv", index=False)
    rho, p_s = spearmanr(T.coverage, T.roc)
    r, p_p = pearsonr(T.coverage, T.roc)
    print(f"\nSpearman(coverage, ROC) = {rho:.3f}  p = {p_s:.4f}")
    print(f"Pearson (coverage, ROC) = {r:.3f}  p = {p_p:.4f}")
    print("The dissociation is the finding: the relationship is ordinal and")
    print("saturating, not proportional.")
    print(f"\nwritten: {RESULTS / 'exp8_threshold.csv'}")


if __name__ == "__main__":
    main()
