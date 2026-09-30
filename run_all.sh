#!/usr/bin/env bash
# Reproduce every result in the paper.
# Build the datasets first (see README, "Getting the data").
set -e
for e in exp1_protocols exp2_matched_size exp3_containment exp4_nearduplicate \
         exp5_dapt_recon exp6_invariant_features exp7_architectures exp8_threshold \
         exp9_supplementary; do
  echo "=== $e ==="
  python "$e.py"
done
echo "=== figures ==="
python make_all.py
echo
echo "Done. Results in results/, figures in figures/."
