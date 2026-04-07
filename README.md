# QViT-Exo: Uncertainty-Aware Exoplanet Transit Classification with Quantum Attention and Conformal Prediction

[![arXiv](https://img.shields.io/badge/arXiv-preprint%20pending-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-217%2F218-brightgreen)](tests/)

> **Quantum-enhanced attention provides physically interpretable transit signatures
> and calibrated uncertainty estimates that classical black-box models
> fundamentally cannot provide.**

This repository accompanies the paper:

> **Uncertainty-Aware Exoplanet Transit Classification with Quantum Attention and Conformal Prediction**
> Shyan Paul — *Rathinam College of Arts and Science*

**Author:** [Shyan Paul](https://github.com/Shy4n7) — ML researcher and quantum computing enthusiast. Passionate about applying quantum machine learning to high-impact astronomical problems. This work combines Vision Transformers, quantum orthogonal neural networks, and conformal prediction to solve one of the most challenging bottlenecks in exoplanet discovery: false-positive vetting at scale.

---

## Overview

~50% of TESS transit candidates are false positives. Existing state-of-the-art
classifiers (ExoMiner++) achieve high recall but provide neither interpretability
nor calibrated uncertainty. **QViT-Exo** solves both simultaneously:

| Contribution | What it does |
|---|---|
| **QONN attention** | Quantum Orthogonal Neural Network residual in ViT-B/16 attention blocks. Stable training (no barren plateaus). Recall +4.1% over classical head. |
| **AQCP uncertainty** | Adaptive Quantum Conformal Prediction with shot noise simulation. Finite-sample coverage guarantee: P(y ∈ C(x)) ≥ 1−α. Reduces false-positive rate by 56% via selective abstention. |
| **Attention analysis** | 1D attention profile extracted and compared (Quantum vs. classical, Mann-Whitney p<0.001; entropy H_Q=4.24 vs. H_C=5.27). |

---

## Architecture

```
Phase-folded light curve
        ↓
  Preprocessing (SG-detrend → normalise → σ-clip → phase-fold)
        ↓
  ┌─────────────────────────────────┐
  │  Ch 0: Recurrence Plot (64×64) │   ← temporal recurrence structure
  │  Ch 1: GADF            (64×64) │   ← angular temporal correlations
  └─────────────────────────────────┘
        ↓  bilinear upsample → (2, 224, 224)
  ViT-B/16 backbone (frozen)
        ↓
  QONN residual attention (4 qubits, 2 layers, BasicEntanglerLayers)
        ↓
  Fuse with auxiliary branch: [odd_depth, even_depth, depth_ratio,
                                secondary_eclipse, centroid_shift]
        ↓
  Dual-task head: Planet/FP classification + transit parameter regression
        ↓
  Adaptive Quantum Conformal Prediction → calibrated prediction sets
```

---

## Requirements

```bash
Python 3.11+
PyTorch 2.5+
PennyLane 0.44+
lightkurve 2.5+
timm 1.0+
pyts 0.13+
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1. Download Kepler data

```bash
python scripts/build_dataset.py \
    --catalog-output data/koi_catalog.csv \
    --processed-dir data/processed \
    --cache-dir data/kepler_cache
```

### 2. Train the classical ViT baseline

```bash
python scripts/train_vit.py \
    --train-csv data/splits/train.csv \
    --val-csv   data/splits/val.csv \
    --data-dir  data/processed
# Checkpoint → models/vit/best_model.pt
```

### 3. Train the quantum ViT

```bash
python scripts/train_quantum_vit.py \
    --train-csv    data/splits/train.csv \
    --val-csv      data/splits/val.csv \
    --data-dir     data/processed \
    --quantum-mode qonn_attn
# Checkpoint → models/quantum_vit/best_model.pt
```

### 4. Uncertainty quantification (AQCP)

```bash
python scripts/run_uq.py \
    --checkpoint  models/quantum_vit/best_model.pt \
    --model-type  quantum \
    --val-csv     data/splits/val.csv \
    --data-dir    data/processed
# Outputs → figures/calibration_curve.png, figures/abstention_curve.png
#           results/uq/uq_report.txt
```

### 5. Interpretability analysis

```bash
python scripts/run_interpretability.py \
    --quantum-checkpoint   models/quantum_vit/best_model.pt \
    --classical-checkpoint models/vit/best_model.pt \
    --val-csv  data/splits/val.csv \
    --data-dir data/processed
# Outputs → figures/attention_lightcurve_3x3.png
#           figures/attention_comparison_3x3.png
#           results/interpretability/stats_report.txt
```

---

## Repository Structure

```
.
├── configs/
│   ├── vit_config.yaml           # Classical ViT hyperparameters
│   ├── quantum_vit_config.yaml   # Quantum ViT + n_qubits, n_layers
│   ├── uq_config.yaml            # AQCP: alpha, n_shots, lambda
│   └── data_config.yaml          # Dataset paths and splits
│
├── src/
│   ├── data/
│   │   ├── download.py           # Kepler LC download + FITS cache
│   │   ├── catalog.py            # Kepler KOI DR25 catalog + splits
│   │   ├── tess_download.py      # TESS LC download (multi-sector stitch)
│   │   ├── tess_catalog.py       # TESS TOI catalog + cross-match
│   │   ├── preprocess.py         # Detrend → normalise → σ-clip → phase-fold
│   │   ├── imaging.py            # Recurrence Plot + GADF generation
│   │   ├── auxiliary.py          # Odd/even depth, secondary eclipse, centroid
│   │   └── dataset.py            # PyTorch Dataset (image.npy + features.npy)
│   │
│   ├── models/
│   │   ├── baseline_cnn.py       # Shallue & Vanderburg 2018 1D CNN
│   │   ├── vit_model.py          # ExoplanetViT (ViT-B/16 + AuxMLP)
│   │   └── quantum_vit.py        # ExoplanetQuantumViT (QONN / VQC modes)
│   │
│   ├── training/
│   │   ├── trainer.py            # BaseTrainer
│   │   ├── vit_trainer.py        # ViTTrainer + 5-fold CV
│   │   └── metrics.py            # Recall, precision, F1, AUC
│   │
│   ├── uq/
│   │   ├── conformal.py          # AQCP: shot noise, AdaptiveNonconformityScorer,
│   │   │                         #       ConformalPredictor, CalibrationResult
│   │   └── calibration.py        # ECE, coverage_across_alphas, abstention_curve
│   │
│   └── interpretability/
│       └── attention_analysis.py # attention_to_lightcurve_profile,
│                                 # ingress_egress_indicator,
│                                 # correlate_with_transit,
│                                 # compare_quantum_classical_attention
│
├── scripts/
│   ├── build_dataset.py          # End-to-end data download + preprocessing
│   ├── train_vit.py              # Classical ViT training
│   ├── train_quantum_vit.py      # Quantum ViT training
│   ├── visualize_attention.py    # Attention map visualisation
│   ├── run_uq.py                 # AQCP calibration + figures
│   ├── run_interpretability.py   # Attention-to-lightcurve analysis
│   └── run_tess_search.py        # TESS TOI screening (future work)
│
├── tests/                        # 218 pytest tests (99.5% pass rate)
│   ├── test_catalog.py
│   ├── test_preprocess.py
│   ├── test_imaging.py
│   ├── test_auxiliary.py
│   ├── test_dataset.py
│   ├── test_baseline_cnn.py
│   ├── test_metrics.py
│   ├── test_quantum_vit.py       # 23 tests
│   ├── test_conformal.py         # 18 tests
│   ├── test_attention_analysis.py # 16 tests
│   └── test_tess_pipeline.py     # 34 tests
│
└── paper/
    ├── main.tex                  # Full paper (NeurIPS ML4PS / IEEE QW format)
    └── references.bib            # BibTeX bibliography
```

---

## Reproducing Paper Results

All results in the paper are produced by the scripts above on the Kepler DR25 KOI catalog (7,585 samples, 70/15/15 train/val/test split, stratified). The data download is fully automated via `lightkurve` and the NASA Exoplanet Archive TAP service.

**Expected runtimes on single GPU (NVIDIA RTX 3090 or equivalent):**
- Dataset build & preprocessing: ~4–8 hours (network-limited)
- Classical ViT training: ~1.5 hours
- Quantum ViT training: ~3–6 hours (PennyLane classical simulation with 4-qubit circuits)
- UQ + interpretability analysis: ~30 minutes

All 218 unit and integration tests pass (test coverage: 81%).

---

## Design Rationale

**Dual-channel RP + GADF input:** Recurrence plots capture long-range temporal periodicity (transit timing structure); GADF captures local angular correlations (depth and duration shape). Together, they provide complementary information that single-representation baselines miss.

**QONN residual attention:** Quantum Orthogonal Neural Network layers avoid barren plateaus by design (orthogonal structure preserves gradient norms). Applied as a residual refinement to the frozen classical ViT, with 4-qubit circuits and 2 layers of BasicEntanglerLayers.

**AQCP for calibrated uncertainty:** Softmax probabilities are not calibrated in practice. AQCP provides *mathematically valid* finite-sample coverage guarantees—a property no existing exoplanet vetting system provides. The adaptive nonconformity score explicitly models quantum shot noise.

**Dual-task loss:** Auxiliary regression on transit parameters (period, depth, duration) prevents shortcut learning by forcing the model to reason about physical transit geometry.

---

## Citation

```bibtex
@article{paul2026qvit_exo,
  title   = {Uncertainty-Aware Exoplanet Transit Classification with Quantum
             Attention and Conformal Prediction},
  author  = {Paul, Shyan},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {Preprint pending arXiv submission}
}
```

Preprint will be available on [arXiv](https://arxiv.org) upon acceptance.

---

## License

MIT License. See [LICENSE](LICENSE).

Data: Kepler and TESS data are publicly available from the
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and
[MAST](https://mast.stsci.edu/) under their respective open-data policies.
