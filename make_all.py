#!/usr/bin/env python3
"""
Regenerate every figure in the paper from the CSVs in results/.

Usage:  python src/figures/make_all.py
Writes: figures/fig0..fig9  (PDF for submission, PNG for preview)
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["fig0.py", "fig_desc.py", "fig_model.py", "fig6.py", "fig7.py", "fig9.py"]

for s in SCRIPTS:
    p = HERE / s
    if not p.exists():
        print(f"[skip] {s}")
        continue
    print(f"[run ] {s}")
    subprocess.run([sys.executable, str(p)], cwd=HERE.parents[1], check=False)
print("\nFigures written to figures/")
