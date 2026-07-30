#!/usr/bin/env python3
"""
Experiment 4 — Near-duplicate analysis  (Section 6.4, Table 12).

Clusters flows by rounding features to N significant digits and checks:
  (a) how prevalent near-duplicates are,
  (b) whether any cluster mixes the two classes (label leakage),
  (c) whether any cluster spans a tool and a browser.

Reproduces:  2.0 % near-duplicates at 3 significant digits
             0 clusters mixing classes
             273 clusters crossing groups, all tool-to-tool
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
_p = Path(__file__).resolve()
for _c in (_p.parent, _p.parents[1], _p.parents[1] / "src"):
    if (_c / "common.py").exists():
        sys.path.insert(0, str(_c)); break
from common import DATA, RESULTS, load_doh                          # noqa: E402


def sig_round(a, sig):
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.floor(np.log10(np.abs(a)))
        e = np.where(np.isfinite(e), e, 0)
    f = 10.0 ** (sig - 1 - e)
    return np.nan_to_num(np.round(a * f) / f)


def main() -> None:
    # Table 12 characterises the dataset BEFORE near-duplicate removal.
    pre = DATA / "doh_exfil_predeup.parquet"
    if pre.exists():
        df, F = load_doh(pre)
        print(f"using pre-deduplication dataset: {pre.name}")
    else:
        df, F = load_doh()
        print("WARNING: pre-deduplication file not found; running on the "
              "de-duplicated set, which will report ~0 near-duplicates.\n"
              "         Re-run src/data/build_doh.py to generate it.")
    X = np.nan_to_num(df[F].to_numpy(np.float64))
    print(f"rows: {len(df):,}")
    print(f"exact duplicates: {int(df.duplicated(subset=F + ['y']).sum())}\n")

    rows = []
    for sig in (5, 4, 3, 2):
        _, inv = np.unique(sig_round(X, sig), axis=0, return_inverse=True)
        n_red = len(df) - len(np.unique(inv))
        rows.append(dict(sig_digits=sig, redundant=n_red,
                         share=round(n_red / len(df) * 100, 2)))
        print(f"  {sig} significant digits : {n_red:>7,}  "
              f"({n_red / len(df) * 100:5.2f} %)")
        if sig == 3:
            g = pd.DataFrame({"c": inv, "y": df.y.values, "grp": df.grp.values})
            mixed = int((g.groupby("c").y.nunique() > 1).sum())
            cross = g.groupby("c").grp.nunique()
            shared = cross[cross > 1].index
            print(f"      clusters mixing both classes : {mixed}")
            print(f"      clusters crossing groups     : {len(shared)}")
            spans = (g[g.c.isin(shared)].groupby("c").grp
                     .apply(lambda s: tuple(sorted(s.unique()))).value_counts())
            for k, v in spans.items():
                print(f"        {' + '.join(k):40} {v}")

    pd.DataFrame(rows).to_csv(RESULTS / "exp4_nearduplicate.csv", index=False)
    print(f"\nwritten: {RESULTS / 'exp4_nearduplicate.csv'}")


if __name__ == "__main__":
    main()
