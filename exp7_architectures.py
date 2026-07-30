#!/usr/bin/env python3
"""
Experiment 7 — Architecture independence  (Section 10, Tables 20-22).

Repeats P3 (DoHBrw) and the day-splits (DAPT) across six classifier families.

Reproduces:  DoHBrw — 5 of 6 families show the asymmetry; the gap scales with
                      model capacity (XGBoost 0.405, HistGB 0.352, ...)
             DAPT   — 6 of 6 families, unanimous, sigma <= 0.05
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (RESULTS, TOOLS, fit_score, load_dapt_recon,     # noqa: E402
                    load_doh, make_models, p3_splits, subsample)
from sklearn.model_selection import train_test_split

SEEDS_DOH, SEEDS_DAPT = (0, 1), (0, 1, 2)


def run_doh() -> pd.DataFrame:
    df, F = load_doh()
    rows, t0 = [], time.time()
    for tool, br, tr_all, te in p3_splits(df):
        for s in SEEDS_DOH:
            tr = subsample(tr_all, 22_000, s)
            for name, m in make_models(s).items():
                r = fit_score(df, F, tr, te, model=m, seed=s)
                rows.append(dict(model=name, tool=tool, browser=br, seed=s, **r))
        print(f"  [{time.time() - t0:5.0f}s] {br}/{tool}", flush=True)
    return pd.DataFrame(rows)


def run_dapt() -> pd.DataFrame:
    df, F = load_dapt_recon()
    y = df.y.to_numpy()
    splits = {"random": None,
              "Mon+Tue->Wed": (["monday", "monday-pvt", "tuesday"], ["wednesday"]),
              "Mon+Wed->Tue": (["monday", "monday-pvt", "wednesday"], ["tuesday"])}
    rows, t0 = [], time.time()
    for name, cfg in splits.items():
        for s in SEEDS_DAPT:
            if cfg is None:
                tr, te = train_test_split(np.arange(len(df)), test_size=.3,
                                          random_state=s, stratify=y)
            else:
                tr = np.flatnonzero(df.day.isin(cfg[0]))
                te = np.flatnonzero(df.day.isin(cfg[1]))
            tr = subsample(tr, 25_000, s)
            for mn, m in make_models(s).items():
                r = fit_score(df, F, tr, te, model=m, seed=s)
                rows.append(dict(model=mn, split=name, seed=s, **r))
        print(f"  [{time.time() - t0:5.0f}s] {name}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    print("--- DoHBrw, protocol P3 ---")
    D = run_doh(); D.to_csv(RESULTS / "exp7_arch_doh.csv", index=False)
    print("\n=== Table 20: ROC-AUC by family and held-out client ===")
    piv = D.pivot_table(index="model", columns="browser", values="roc",
                        aggfunc=["mean", "std"]).round(4)
    print(piv.to_string())
    gap = (D[D.browser == "chrome"].groupby("model").roc.mean()
           - D[D.browser == "firefox"].groupby("model").roc.mean())
    print("\n  gap (chrome - firefox), sorted:")
    print(gap.sort_values(ascending=False).round(3).to_string())

    print("\n--- DAPT 2020, day splits ---")
    P = run_dapt(); P.to_csv(RESULTS / "exp7_arch_dapt.csv", index=False)
    for met in ("roc", "f1"):
        print(f"\n=== Table {21 if met == 'roc' else 22}: {met.upper()} ===")
        mu = P.pivot_table(index="model", columns="split", values=met, aggfunc="mean")
        sd = P.pivot_table(index="model", columns="split", values=met, aggfunc="std")
        out = mu.round(4).astype(str) + " +- " + sd.round(4).astype(str)
        cols = [c for c in ["random", "Mon+Tue->Wed", "Mon+Wed->Tue"] if c in out]
        print(out[cols].to_string())
    print(f"\nwritten: {RESULTS}/exp7_arch_doh.csv, exp7_arch_dapt.csv")


if __name__ == "__main__":
    main()
