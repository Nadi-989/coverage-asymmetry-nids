#!/usr/bin/env python3
"""
Experiment 3 — Asymmetric containment  (Section 7, Table 13; Section 8.3).

Computes C(X -> Y) of Equation 9 on both datasets and shows the same
quantity governs both, with the roles of benign and attack exchanged.

Reproduces:  C(Chrome -> Firefox) = 0.888   C(Firefox -> Chrome) = 0.325
             C(Wed -> Tue)        = 0.962   C(Tue -> Wed)        = 0.099
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (RESULTS, containment, load_dapt_recon,          # noqa: E402
                    load_doh, top_dispersed)


def main() -> None:
    rows = []

    # ---- DoHBrw: benign-client axis --------------------------------------
    df, F = load_doh()
    df[F] = df[F].fillna(df[F].median())
    ch, ff = df[df.grp == "chrome"], df[df.grp == "firefox"]
    S = top_dispersed(ch, ff, F)
    c_ch_in_ff = containment(ch, ff, S)
    c_ff_in_ch = containment(ff, ch, S)
    print("=== DoHBrw — benign client ===")
    print(f"  measurement features : {S}")
    print(f"  C(Chrome  -> Firefox) = {c_ch_in_ff:.3f}")
    print(f"  C(Firefox -> Chrome ) = {c_ff_in_ch:.3f}")
    rows += [dict(dataset="DoHBrw", axis="benign client",
                  train="Firefox", test="Chrome", coverage=c_ch_in_ff),
             dict(dataset="DoHBrw", axis="benign client",
                  train="Chrome", test="Firefox", coverage=c_ff_in_ch)]

    print("\n  dispersion ratios (Firefox / Chrome):")
    for c in S:
        a = ch[c].quantile(.95) - ch[c].quantile(.05)
        b = ff[c].quantile(.95) - ff[c].quantile(.05)
        print(f"    {c:38} {b / (a + 1e-12):>6.2f}x")

    # ---- DAPT: attack-mix axis -------------------------------------------
    dp, FD = load_dapt_recon()
    dp[FD] = dp[FD].fillna(dp[FD].median())
    tue = dp[(dp.day == "tuesday") & (dp.y == 1)]
    wed = dp[(dp.day == "wednesday") & (dp.y == 1)]
    S2 = top_dispersed(tue, wed, FD)
    c_wed_in_tue = containment(wed, tue, S2)
    c_tue_in_wed = containment(tue, wed, S2)
    print("\n=== DAPT — attack mix ===")
    print(f"  measurement features : {S2}")
    print(f"  C(Wed -> Tue) = {c_wed_in_tue:.3f}   [train Tue covers Wed]")
    print(f"  C(Tue -> Wed) = {c_tue_in_wed:.3f}   [train Wed covers Tue]")
    rows += [dict(dataset="DAPT", axis="attack mix",
                  train="Tuesday", test="Wednesday", coverage=c_wed_in_tue),
             dict(dataset="DAPT", axis="attack mix",
                  train="Wednesday", test="Tuesday", coverage=c_tue_in_wed)]

    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "exp3_containment.csv", index=False)
    print("\n=== Unified view ===")
    print(R.to_string(index=False))
    print(f"\nwritten: {RESULTS / 'exp3_containment.csv'}")


if __name__ == "__main__":
    main()
