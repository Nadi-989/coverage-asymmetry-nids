#!/usr/bin/env python3
"""
Build the DoH exfiltration dataset (Section 4.3).

Input : CIRA-CIC-DoHBrw-2020 "all.csv" files
          MaliciousDoH-CSVs/CSVs/{dns2tcp,dnscat2,iodine}/all.csv
          BenignDoH-NonDoH-CSVs/CSVs/{Chrome,Firefox}/all.csv
Output: data/doh_exfil_dedup.parquet   (263,403 flows x 29 features)

Two construction decisions, both stated in the paper:
  1. The benign side is restricted to Benign-DoH (DoH == True). Admitting
     non-DoH HTTPS turns the task into DoH-vs-non-DoH discrimination.
  2. Identifiers (IPs, ports, timestamps) are dropped. Tunnels originate from
     fourteen known addresses; retaining them lets a model memorise a list.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA                                          # noqa: E402

DROP = ["SourceIP", "DestinationIP", "SourcePort", "DestinationPort",
        "TimeStamp", "DoH"]
TOOLS = {"dns2tcp": "dns2tcp", "dnscat2": "dnscat2", "iodine": "iodin"}
BROWSERS = ["chrome", "firefox"]


def sig_round(a: np.ndarray, sig: int) -> np.ndarray:
    """Round to `sig` significant digits (used for near-duplicate clustering)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.floor(np.log10(np.abs(a)))
        e = np.where(np.isfinite(e), e, 0)
    f = 10.0 ** (sig - 1 - e)
    return np.nan_to_num(np.round(a * f) / f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--malicious-dir", required=True,
                    help="directory containing dns2tcp/ dnscat2/ iodine/")
    ap.add_argument("--benign-dir", required=True,
                    help="directory containing Chrome/ Firefox/")
    ap.add_argument("--out", default=str(DATA / "doh_exfil_dedup.parquet"))
    ap.add_argument("--sig-digits", type=int, default=3,
                    help="significant digits for near-duplicate removal")
    a = ap.parse_args()

    parts = []
    for folder, tag in TOOLS.items():
        p = Path(a.malicious_dir) / folder / "all.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        d = pd.read_csv(p, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        d = d[d.DoH == True].copy()                              # noqa: E712
        d["y"], d["grp"] = 1, tag
        parts.append(d)
        print(f"  {tag:9} {len(d):>8,} malicious DoH flows")

    for br in BROWSERS:
        p = Path(a.benign_dir) / br.capitalize() / "all.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        d = pd.read_csv(p, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        d = d[d.DoH == True].copy()                              # benign DoH only
        d["y"], d["grp"] = 0, br
        parts.append(d)
        print(f"  {br:9} {len(d):>8,} benign DoH flows")

    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in df.columns if c not in DROP + ["y", "grp"]]
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)

    n0 = len(df)
    df = df.drop_duplicates(subset=feats + ["y"]).reset_index(drop=True)
    print(f"\nexact duplicates removed : {n0:,} -> {len(df):,}")

    pre = Path(a.out).with_name("doh_exfil_predeup.parquet")
    df[feats + ["y", "grp"]].to_parquet(pre, index=False, compression="zstd")
    print(f"pre-dedup copy written    : {pre}  (used by exp4)")

    X = np.nan_to_num(df[feats].to_numpy(np.float64))
    _, inv = np.unique(sig_round(X, a.sig_digits), axis=0, return_inverse=True)
    keep = ~pd.Series(inv).duplicated().to_numpy()
    df = df.loc[keep, feats + ["y", "grp"]].reset_index(drop=True)
    print(f"near-duplicates removed  : -> {len(df):,}  "
          f"({a.sig_digits} significant digits)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False, compression="zstd")
    print(f"\nwritten: {a.out}")
    print(pd.crosstab(df.grp, df.y).to_string())


if __name__ == "__main__":
    main()
