# qadv — Quantum Advantage Screening for Drug Discovery Kernels

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Qiskit 2.5](https://img.shields.io/badge/qiskit-2.5.2-purple)](https://qiskit.org/)

**qadv** is a research prototype for *label-independent screening and hardware validation of quantum kernels* in molecular property prediction and drug-discovery tasks. It provides:

- A **theoretically grounded screening pipeline** (geometric difference *g/√N*, concentration certificates, shot-cost estimates) backed by 11 numerically verified theorems
- **Exact numpy ZZ-feature-map kernels** that match Qiskit circuits to machine precision (error ≤ 1.8 × 10⁻¹⁵)
- **Engineered-advantage controls** (Huang-style, T4) and classical twin-kernel rivals for falsification
- A **single-PUB parametrised hardware estimator** that runs a full train-Gram experiment in seconds of QPU time on IBM Quantum
- **Analysis and reporting scripts** producing publication-ready tables and figures

> **Status (v0.1.1):** local simulation and Aer noisy-emulation paths fully tested; IBM hardware path demonstrated on `ibm_marrakesh` (156-qubit Heron r2) for N = 12 (actual QPU usage ≈ 11 s).

---

## Repository layout

```
qadv/
  kernels.py       exact numpy ZZ-type fidelity kernels; classical RBF/Laplacian/twin/Tanimoto; finite-shot emulation
  diagnostics.py   g/√N, eps_identity (gate G1b), off-diagonal stats, shots-to-resolve, CKA, depolarising fit
  models.py        train-only feature pipeline, kernel families, nested tuning with grid expansion, sklearn rivals
  data.py          dataset adapters (ESOL/Delaney, TDC, Garg & Garg, sklearn, synthetic & engineered controls)
  hardware.py      circuits; samplers (ideal | aer_noisy | ibm); single-PUB estimator; QPU usage estimate

scripts/
  validate_install.py        check environment + numpy-vs-Qiskit correctness
  verify_theorems.py         numerically verify T2–T6, T10 (11/11 checks)
  project_costs.py           kernel concentration and shot/QPU cost vs qubits
  run_grid.py                few-shot grid over all kernel families and datasets
  analyze.py                 tables, Holm tests, H1 predictor, publication figures
  calibrate_noise.py         fit survival factor f vs qubit count (hardware or Aer)
  run_hardware_minimal.py    train-Gram-only single-PUB hardware experiment
  run_hardware.py            train + test Gram for one pre-selected cell
  project_results.py         learning-curve and hardware-SNR projection

data/
  delaney-processed.csv      ESOL solubility dataset (1 128 molecules)

schema/
  certificate.schema.json    JSON schema for reproducibility certificates

tests/
  test_core.py               5 core unit tests
```

---

## Installation

```bash
git clone https://github.com/<YOUR-ORG>/qadv-quantum-kernel-screening.git
cd qadv-quantum-kernel-screening

python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verify the installation
python scripts/validate_install.py    # expect: ALL CORE CHECKS PASSED
python tests/test_core.py             # expect: 5 PASS lines
python scripts/verify_theorems.py --skip-noisy  # expect: 10/10
python scripts/verify_theorems.py               # expect: 11/11 (needs Qiskit Aer)
```

**Optional extras** (not required for the core pipeline):

```bash
pip install tabpfn PyTDC chemprop   # foundation-model baseline, TDC data, GNN rival
```

### Requirements

| Package | Version |
|---|---|
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| matplotlib | 3.10.8 |
| rdkit | 2026.3.6 |
| qiskit | 2.5.2 |
| qiskit-aer | 0.17.2 |
| qiskit-ibm-runtime | 0.50.0 |

Tested on Python 3.12 · 1 CPU · 3 GB RAM (macOS / Linux).

---

## Quick start

```bash
# 1. Run a tiny grid (sanity check, ~30 s)
python scripts/run_grid.py \
  --datasets syn:classical sk:wine \
  --n-qubits 8 --n-train 10 20 --draws 3 \
  --out results/quick.jsonl --no-extra

# 2. Analyse and generate figures/tables
python scripts/analyze.py results/quick.jsonl --out results/quick_report
```

Outputs in `results/quick_report/`: `table_gaps.csv`, `table_family_scores.csv`, `fig_gap_vs_N.png`, …

---

## Datasets

| Spec string | Description | Notes |
|---|---|---|
| `delaney` | ESOL solubility regression (1 128 molecules) | `data/delaney-processed.csv` required |
| `sk:wine` / `sk:breast_cancer` / `sk:diabetes` | scikit-learn tabular | sanity anchors |
| `syn:quantum` / `syn:classical` | Synthetic positive/negative controls | no external data |
| `eng:delaney@0.6@rbf` | Engineered-advantage control on ESOL features | rivals: `rbf`, `rbf+lap`, `all` |
| `tdc:<name>` | TDC ADMET benchmarks | needs `pip install PyTDC` + network |
| `garg:<csv>:<TARGET>:<threshold>` | Garg & Garg (2026) drug potency | needs `git lfs pull` (~199 MB) |

```bash
# Download ESOL data
mkdir -p data
curl -L -o data/delaney-processed.csv \
  https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv

# Download Garg & Garg potency data (Git LFS)
git clone https://github.com/shiv2158/potency_prediction_model_evaluation
cd potency_prediction_model_evaluation && git lfs pull
```

---

## Experiments

### E1 — Theorem verification

```bash
python scripts/verify_theorems.py
```

Numerically checks 11 theorems (T2–T6, T10). Writes `results/theorem_checks.json`.  
Expected result: `SUMMARY: 11 / 11 checks passed`.

### E2 — Concentration and cost curves

```bash
python scripts/project_costs.py \
  --qubits 4 6 8 10 12 \
  --bandwidth 0.1 0.3 0.6 1.0 \
  --n-train 16 --shots 1000
```

### E3 — Engineered-advantage controls

```bash
for riv in rbf rbf+lap all; do
  python scripts/run_grid.py \
    --datasets "eng:delaney@0.6@$riv" \
    --n-qubits 8 --n-train 10 20 30 50 75 \
    --draws 30 --n-test 150 \
    --out results/eng_${riv/+/_}.jsonl --no-extra
  python scripts/analyze.py results/eng_${riv/+/_}.jsonl \
    --out results/report_eng_${riv/+/_}
done
```

### E5 — Pharma few-shot grid

```bash
python scripts/run_grid.py \
  --datasets delaney tdc:Caco2_Wang tdc:HIA_Hou tdc:BBB_Martins tdc:hERG tdc:DILI \
  --n-qubits 8 --n-train 10 20 30 50 75 100 \
  --draws 30 --n-test 150 \
  --out results/pharma.jsonl
python scripts/analyze.py results/pharma.jsonl --out results/report_pharma
```

### E8 — Hardware experiment (train-Gram only)

**Local rehearsal (no IBM account needed):**

```bash
# Ideal statevector
python scripts/run_hardware_minimal.py \
  --dataset "eng:delaney@0.6@rbf" --n 16 --mode ideal --shots 1000

# Noisy emulation (FakeKingston, Heron r2 noise snapshot)
python scripts/run_hardware_minimal.py \
  --dataset "eng:delaney@0.6@rbf" --n 16 --mode aer_noisy --shots 1000
```

**IBM Quantum hardware** (requires an IBM account; see [IBM Quantum account setup](#ibm-quantum-account-setup)):

```bash
# Dry run — prints budget estimate, submits no job
python scripts/run_hardware_minimal.py \
  --dataset "eng:delaney@0.6@rbf" --n 12 \
  --mode ibm --backend ibm_kingston --shots 500 --dry-run

# Live job
python scripts/run_hardware_minimal.py \
  --dataset "eng:delaney@0.6@rbf" --n 12 \
  --mode ibm --backend ibm_kingston --shots 500
```

**Measured result (ibm_marrakesh, N = 12, 500 shots):**

| Quantity | Ideal | Noisy Aer | Real ibm_marrakesh |
|---|---|---|---|
| Off-diagonal rel. error | 0.075 | 0.299 | **0.502** |
| Survival f | 0.94 | 0.706 | **0.514** |
| CKA(hw, exact) | 1.000 | 0.995 | **0.995** |
| KTA: quantum exact / hardware | 0.242 / 0.244 | — | 0.307 / **0.301** |
| KTA: RBF / Laplacian / twin | 0.129 / 0.225 / 0.263 | — | 0.241 / 0.305 / **0.308** |

---

## IBM Quantum account setup

1. Create a free account at [quantum.ibm.com](https://quantum.ibm.com).
2. Copy your API key from the dashboard.
3. Save it **once** (stored in `~/.qiskit/qiskit-ibm.json`, never commit this file):

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="<YOUR_IBM_QUANTUM_API_KEY>",   # replace — do NOT commit
    overwrite=True
)
```

4. Confirm access:

```python
svc = QiskitRuntimeService()
print([(b.name, b.num_qubits)
       for b in svc.backends(simulator=False, operational=True)])
```

> **Never hard-code or commit your API key.** The scripts call `QiskitRuntimeService()` with no arguments and rely on the saved account file.

---

## QPU budget guide

Formula (IBM docs): `2 s × sub-jobs + (rep_delay + circuit_length) × rows × shots`  
Assumed: rep_delay = 250 µs, circuit ≈ 6 µs, 1 PUB. Verify with `job.usage_estimation`.

| N | Pairs (rows) | 200 shots | 500 shots | 1 000 shots |
|---|---|---|---|---|
| 8 | 28 | 3.4 s | 5.6 s | 9.2 s |
| 12 | 66 | 5.4 s | 10.4 s | 18.9 s |
| 16 | 120 | 8.1 s | 17.4 s | 32.7 s |
| 20 | 190 | 11.7 s | 26.3 s | 50.6 s |

**Recommendation:** start at N ≤ 12, `--shots 200` until you confirm the circuit runs correctly, then scale up.

---

## Reproducibility

```bash
# Archive exact environment
pip freeze > freeze.txt

# All theorem checks pass
python scripts/verify_theorems.py   # 11/11

# Per-experiment reproducibility
python scripts/run_grid.py ... --draws 30 ...   # fixed seeds inside the script
```

Checklist before submission:
- [ ] `pip freeze` archived alongside results
- [ ] Seeds and split indices recorded in JSONL output
- [ ] Raw counts, job IDs, backend calibration snapshots archived
- [ ] `verify_theorems.py` 11/11 in the release commit
- [ ] Dataset licences and data hashes recorded
- [ ] Every table traceable to a command + seed + output file

---

## Adding a new domain

```python
# qadv/data.py
def load_my_domain(path):
    import pandas as pd
    d = pd.read_csv(path)
    return Dataset("my_set", "my_domain", "regression",
                   d[feature_cols].values.astype(float), d["label"].values,
                   groups=d["cluster"].values)

# Add a branch in get_dataset:
# if kind == "my":
#     return load_my_domain(rest)
```

Features are standardised → PCA(n_qubits) → scaled **inside each training split** (no leakage).

---

## Citation

If you use this software, please cite it (update with Zenodo DOI after minting):

```bibtex
@software{qadv2025,
  title  = {qadv: label-independent screening and hardware validation of quantum kernels},
  version = {0.1.1},
  license = {Apache-2.0},
  url    = {https://github.com/<YOUR-ORG>/qadv-quantum-kernel-screening}
}
```

See also [`CITATION.cff`](CITATION.cff).

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: data/delaney-processed.csv` | Run from repo root; download CSV (see Datasets section) |
| `AccountNotFoundError` | Run `QiskitRuntimeService.save_account(...)` (see IBM Quantum account setup) |
| `Cannot uninstall PyJWT … debian` | Use `python3 -m venv venv --system-site-packages` |
| Transpile warnings / API errors | Pin versions from `requirements.txt`; all version-sensitive calls are isolated in `qadv/hardware.py` |
| Negative Gram eigenvalues | Use `D.nearest_psd(K)` before SVM fitting |
| `eng:` dataset slow | Reduce `n_points` in `engineered_advantage(...)` |
| Grid-boundary rate high | Kernel grids auto-expand; widen Tanimoto power grid |
| LOO-AUC below 0.5 at tiny N | Known artefact with balanced classes at very small N; use KTA (as in E8) instead |
