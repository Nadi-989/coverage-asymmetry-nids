#!/usr/bin/env python3
"""
Build the DAPT 2020 reconnaissance subset (Section 8.1).

Input : DAPT 2020 flow CSVs (enp0s3-*_pcap_Flow.csv)
Output: data/dapt_recon.parquet   (71,770 flows x 65 features)

Note: DAPT 2020 is REJECTED for exfiltration by the audit (Section 4.2) — its
Data Exfiltration activity contains six flows, five of them empty. Its
reconnaissance classes, however, are well populated and are what we use here.
"""
from __future__ import annotations
import argparse, glob, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

_p = Path(__file__).resolve()
for _c in (_p.parent, _p.parents[1], _p.parents[1] / "src"):
    if (_c / "common.py").exists():
        sys.path.insert(0, str(_c)); break
from common import DATA                                          # noqa: E402

RECON = {"Network Scan", "Directory Bruteforce", "Web Vulnerability Scan",
         "Account Discovery", "Account Bruteforce"}
BENIGN = {"normal", "benign"}          # matched case-insensitively: Benign / BENIGN / Normal
DROP = {"Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
        "Timestamp", "Activity", "Stage"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of DAPT 2020 CSVs")
    ap.add_argument("--out", default=str(DATA / "dapt_recon.parquet"))
    a = ap.parse_args()

    parts = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.csv"))):
        d = pd.read_csv(f, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        if "Activity" not in d.columns:
            print(f"  [skip] {os.path.basename(f)} — no Activity column "
                  "(this file has lost its header row)")
            continue
        act = d["Activity"].astype(str).str.strip()
        is_ben = act.str.lower().isin(BENIGN)
        d = d[act.isin(RECON) | is_ben].copy()
        d["y"] = act[d.index].isin(RECON).astype(int)
        # official names look like enp0s3-public-tuesday.pcap_Flow.csv; strip
        # everything from ".pcap"/"_pcap" on so the day is "tuesday", not "tuesday.pcap"
        stem = re.split(r"[._]pcap", os.path.basename(f))[0]
        d["day"] = (stem
                    .replace("enp0s3-", "").replace("public-", "")
                    .replace("pvt-", "").replace("tcpdump-", ""))
        parts.append(d)

    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in df.columns if c not in DROP | {"y", "day"}]
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)

    const = [c for c in feats if df[c].nunique(dropna=False) <= 1]
    feats = [c for c in feats if c not in const]
    n0 = len(df)
    df = df.drop_duplicates(subset=feats + ["y"]).reset_index(drop=True)

    print(f"exact duplicates removed : {n0:,} -> {len(df):,}")
    print(f"constant columns removed : {len(const)}")
    print(f"features                 : {len(feats)}")
    print(f"positive rate            : {df.y.mean():.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df[feats + ["y", "day"]].to_parquet(a.out, index=False, compression="zstd")
    print(f"\nwritten: {a.out}")
    print(pd.crosstab(df.day, df.y).to_string())


if __name__ == "__main__":
    main()
