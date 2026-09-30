# Coverage Asymmetry, Not Sample Size

Reproducibility package for *Coverage Asymmetry, Not Sample Size: Auditing Benchmark Datasets and Generalisation Limits in Network Attack Detection*.

Every number, table and figure in the paper is produced by the scripts in this repository.

---

## What this repository shows

**1 — Two of three widely used benchmarks do not contain the behaviour they label.**

| Dataset | Labelled class | What the audit finds | Verdict |
|---|---|---|---|
| CSE-CIC-IDS2018 | `Infiltration` | median egress 40 B against 51 B benign; 50.6 % of flows ≤ 2 packets; 33.2 % empty | rejected |
| DAPT 2020 | `Data Exfiltration` | six flows, five empty; maximum egress 107 B to port 4444 | rejected |
| CIRA-CIC-DoHBrw-2020 | Malicious DoH | 249,969 flows, 0.00 % empty, egress to 8.0 MB | **accepted** |

**2 — On datasets that survive audit, the standard robustness protocol measures the wrong axis.**

| Protocol | ROC-AUC |
|---|---|
| Random 70/30 | 1.000 |
| Leave-one-tunnelling-tool-out | 1.000 |
| Leave-tool-**and-client**-out | 0.797 ± 0.212 |

The average conceals a split: 0.994 when Chrome is held out, 0.601 when Firefox is (20 seeds). Recall stays at 0.999; precision falls to 0.819, and at the default threshold 86 % of Firefox's benign flows are flagged as exfiltration.

**3 — The cause is coverage, not sample size, and five flows fix it.**

With benign training fixed at 3,311 flows in both directions the asymmetry persists (0.997 against 0.588). Adding just **five** benign flows from the held-out client raises ROC-AUC from 0.602 to 0.989.

---

## Installation

```bash
git clone https://github.com/Nadi-989/coverage-asymmetry-nids.git
cd coverage-asymmetry-nids
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later. `xgboost` is optional — without it, Tables 20–22 report five classifier families instead of six.

---

## Getting the data

The source datasets are **not redistributed** here. Download them from their providers, then build the derived datasets.

### CIRA-CIC-DoHBrw-2020 — required

Download `MaliciousDoH-CSVs.zip` and `BenignDoH-NonDoH-CSVs.zip` from
<https://www.unb.ca/cic/datasets/dohbrw-2020.html>, unzip both, then:

```bash
python `build_doh.py` \
    --malicious-dir path/to/MaliciousDoH-CSVs/CSVs \
    --benign-dir    path/to/BenignDoH-NonDoH-CSVs/CSVs
```

Produces `data/doh_exfil_dedup.parquet` — 263,403 flows × 29 features.

### DAPT 2020 — required for Sections 8 and 10

Download the flow CSVs from <https://gitlab.thothlab.org/> (`enp0s3-*_pcap_Flow.csv`), then:

```bash
python `build_dapt_recon.py` --dir path/to/dapt2020/csv
```

Produces `data/dapt_recon.parquet` — 71,770 flows × 65 features.

### CSE-CIC-IDS2018 — optional, for Section 4.1 only

Days `02-28-2018.csv` and `03-01-2018.csv` from
<https://www.unb.ca/cic/datasets/ids-2018.html>.

---

## Reproducing the paper

```bash
./run_all.sh
```

Roughly 40 minutes on one CPU core. Or run experiments individually:

| Script | Paper section | Reproduces |
|---|---|---|
| `harvest_signature.py` | §3 | the audit protocol, applicable to any dataset |
| `exp1_protocols.py` | §6.2 | Tables 9–10, Figures 4–6 |
| `exp2_matched_size.py` | §6.3 | Table 11 |
| `exp3_containment.py` | §7, §8.3 | Table 13, Equation 9 |
| `exp4_nearduplicate.py` | §6.4 | Table 12 |
| `exp5_dapt_recon.py` | §8.2 | Table 15 |
| `exp6_invariant_features.py` | §9 | Tables 17–18, Equation 11 |
| `exp7_architectures.py` | §10 | Tables 20–22, Figure 8 |
| `exp8_threshold.py` | §11 | Table 23, Figure 9 |
| `exp9_supplementary.py` | §6.2, §9.4, App. B | Table 19 (both directions), ablations, Table B2 |
| `fig_paper.py` | — | Figures 1, 3–9 |
| `make_all.py` | — | all figures |

---

## Auditing your own dataset

The audit protocol is the reusable part of this work. It answers one question before you train anything: *does the labelled class exhibit the behaviour its label claims?*

```bash
python `harvest_signature.py` \
    --data your_dataset.parquet \
    --label-col Label \
    --attack-values "Data Exfiltration"
```

It reports one of three verdicts:

| Verdict | Meaning |
|---|---|
| ⛔ no signature | the class is reconnaissance or noise — a relabelling will not help |
| ⚠ partial signature | low-volume or stealthy exfiltration — temporal aggregation is indicated |
| ✅ clear signature | flow-level detection is applicable |

The four measured properties are volume, directional asymmetry, persistence and destination concentration; the four rejection criteria are stated in Section 3.4 of the paper.

---

## Methodological controls

These are enforced in code rather than left to discipline, because each is a failure mode we encountered.

- **Imputation medians are estimated on training indices only.** `common.fit_score` takes `tr` and computes the median inside it; there is no path by which test statistics reach the imputer.
- **Identifiers are dropped at dataset construction.** In DoHBrw, tunnels originate from fourteen known addresses while browsing originates elsewhere; keeping `SourceIP` yields a meaningless 100 % accuracy. The columns are removed in `build_doh.py`, not filtered later.
- **Duplicates are removed before splitting.** Exact duplicates first, then near-duplicates at three significant digits. `exp4` verifies that no near-duplicate cluster spans the two classes, so residual redundancy cannot leak the label.
- **PR-AUC is the primary metric.** Under the class ratios here, ROC-AUC is optimistic; see Davis and Goadrich (2006) and Axelsson (2000).
- **Splits are along a dimension unseen in training** — a different day, a different tunnelling tool, a different client — never a random partition.

---

## Repository layout

```
common.py                  loading, models, metrics, coverage (Eq. 9)
harvest_signature.py       the audit protocol (§3)
build_doh.py               constructs the DoH dataset (§4.3)
build_dapt_recon.py        constructs the DAPT subset (§8.1)
exp1 … exp8_*.py           one script per result
fig*.py, make_all.py       figure generation
fig*.pdf                   figures as submitted
*.csv                      result tables
run_all.sh                 reproduces everything
requirements.txt
```

---

## Notes on runtime

Experiments subsample the training set (20–25 k flows) to keep single-core runtimes tractable. This does not affect the reported conclusions: Section 6.3 and Section 11 both establish that performance in these settings is insensitive to training volume above a few thousand flows, which is the point of the paper. To run at full scale, raise the `N_TRAIN` constant in each script.

Seed counts are 2–8 depending on experiment; the significance test in `exp6` uses 18 paired runs. Effect sizes throughout exceed seed variance by one to two orders of magnitude.

---

## Citation

```bibtex
@article{coverage_asymmetry_2026,
  title   = {Coverage Asymmetry, Not Sample Size: Auditing Benchmark Datasets
             and Generalisation Limits in Network Attack Detection},
  author  = {[Author names]},
  journal = {[Journal]},
  year    = {2026},
  doi     = {[DOI]}
}
```

Please also cite the source datasets:

- Sharafaldin, Habibi Lashkari, Ghorbani, *ICISSP* 2018 — CSE-CIC-IDS2018
- Myneni et al., *MLHat/KDD* 2020 — DAPT 2020
- MontazeriShatoori, Davidson, Kaur, Habibi Lashkari, *IEEE CyberSciTech* 2020 — CIRA-CIC-DoHBrw-2020

---

## Licence

Code: MIT (see `LICENSE`). The source datasets are distributed under their own licences by their respective providers and are not redistributed here.
