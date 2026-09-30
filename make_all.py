#!/usr/bin/env python3
"""
Regenerate every figure in the paper from the CSVs in results/.

Usage:  python make_all.py        (run from the repository root)
Writes: figures/fig0..fig9  (PDF for submission, PNG for preview)
"""
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.makedirs(HERE / "figures", exist_ok=True)
failed = []
SCRIPTS = ["fig_paper.py", "fig_desc.py"]
# inputs each script needs beyond the committed CSVs
NEEDS = {"fig_desc.py": ["data/ids2018_infiltration.parquet", "data/doh_exfil_dedup.parquet"],
         "fig_paper.py": ["data/doh_exfil_dedup.parquet", "results/extra_table19.csv", "results/exp7_arch_doh.csv"]}

for s in SCRIPTS:
    p = HERE / s
    if not p.exists():
        print(f"[skip] {s}")
        continue
    missing = [f for f in NEEDS.get(s, []) if not (HERE / f).exists()]
    if missing:
        print(f"[skip] {s} - missing input: {', '.join(missing)}")
        continue
    print(f"[run ] {s}")
    if subprocess.run([sys.executable, str(p)], cwd=HERE).returncode:
        failed.append(s)
if failed:
    sys.exit(f"\nFAILED: {failed}")
print("\nFigures written to figures/")
