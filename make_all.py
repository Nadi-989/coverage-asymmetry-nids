#!/usr/bin/env python3
"""
Regenerate every figure in the paper from the CSVs in results/.

Usage:  python src/figures/make_all.py
Writes: figures/fig0..fig9  (PDF for submission, PNG for preview)
"""
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.makedirs(HERE / "figures", exist_ok=True)
failed = []
SCRIPTS = ["fig0.py", "fig_desc.py", "fig_model.py", "fig6.py", "fig7.py", "fig9.py"]

for s in SCRIPTS:
    p = HERE / s
    if not p.exists():
        print(f"[skip] {s}")
        continue
    print(f"[run ] {s}")
    if subprocess.run([sys.executable, str(p)], cwd=HERE).returncode:
        failed.append(s)
if failed:
    sys.exit(f"\nFAILED: {failed}")
print("\nFigures written to figures/")
