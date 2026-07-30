#!/usr/bin/env bash
# Reproduce every result in the paper.
# Expects the two source datasets to have been downloaded and built first
# (see README, "Getting the data").
set -e

echo "=== Experiment 1 — evaluation protocols (Tables 9-10) ==="
python src/experiments/exp1_protocols.py

echo "=== Experiment 2 — matched sample size (Table 11) ==="
python src/experiments/exp2_matched_size.py

echo "=== Experiment 3 — asymmetric containment (Table 13) ==="
python src/experiments/exp3_containment.py

echo "=== Experiment 4 — near-duplicate analysis (Table 12) ==="
python src/experiments/exp4_nearduplicate.py

echo "=== Experiment 5 — DAPT reconnaissance (Table 15) ==="
python src/experiments/exp5_dapt_recon.py

echo "=== Experiment 6 — coverage-invariant features (Tables 17-18) ==="
python src/experiments/exp6_invariant_features.py

echo "=== Experiment 7 — architecture independence (Tables 20-22) ==="
python src/experiments/exp7_architectures.py

echo "=== Experiment 8 — threshold behaviour (Table 23) ==="
python src/experiments/exp8_threshold.py

echo "=== Figures ==="
python src/figures/make_all.py

echo
echo "Done. Results in results/, figures in figures/."
