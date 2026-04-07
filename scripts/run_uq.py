"""Phase 4 — Run Adaptive Quantum Conformal Prediction pipeline.

Generates:
  1. Calibration curve figure   (empirical vs nominal coverage)
  2. Abstention curve figure    (FPR vs abstention rate)
  3. UQ report                  (Table 2 data for the paper)

Usage
-----
# Quantum model (vqc_head)
python scripts/run_uq.py \\
    --checkpoint models/quantum_vit/best_model.pt \\
    --model-type quantum --quantum-mode vqc_head \\
    --val-csv data/splits/val.csv --data-dir data/processed

# Classical model (for ECE comparison)
python scripts/run_uq.py \\
    --checkpoint models/vit/best_model.pt \\
    --model-type classical \\
    --val-csv data/splits/val.csv --data-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import InterpolationMode, Resize

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ExoplanetDataset
from src.models.vit_model import ExoplanetViT
from src.models.quantum_vit import ExoplanetQuantumViT
from src.uq.conformal import AdaptiveNonconformityScorer, ConformalPredictor
from src.uq.calibration import (
    coverage_across_alphas,
    abstention_curve,
    expected_calibration_error,
    format_calibration_report,
    _collect_labels,
)
from src.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "uq_config.yaml"


class _ResizeTo224:
    def __init__(self) -> None:
        self._r = Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True)
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._r(x)


def _load_model(checkpoint: str, model_type: str, quantum_mode: str) -> torch.nn.Module:
    if model_type == "quantum":
        model = ExoplanetQuantumViT(quantum_mode=quantum_mode, pretrained=False, regression_head=True)
    else:
        model = ExoplanetViT(pretrained=False, regression_head=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def _split_val(dataset: ExoplanetDataset, cal_fraction: float, seed: int):
    n = len(dataset)
    cal_size = max(1, int(n * cal_fraction))
    test_size = n - cal_size
    return random_split(dataset, [cal_size, test_size],
                        generator=torch.Generator().manual_seed(seed))


def _plot_coverage_curve(data: dict, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    nom = data["nominal_coverage"]
    emp = data["empirical_coverage"]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(nom, emp, "o-", color="#3498DB", lw=2, ms=6, label="AQCP (quantum)")
    ax.fill_between(nom, [max(0, e - 0.02) for e in emp],
                    [min(1, e + 0.02) for e in emp], alpha=0.15, color="#3498DB")
    ax.set_xlabel("Nominal coverage (1 − α)", fontsize=12)
    ax.set_ylabel("Empirical coverage", fontsize=12)
    ax.set_title("AQCP Calibration Curve\nExoplanet Transit Vetting", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0.45, 1.0); ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration curve saved → %s", save_path)


def _plot_abstention_curve(data: dict, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ab = data["abstention_rates"]
    conf_fpr = data["confident_fprs"]
    full_fpr = data["full_fprs"]
    ax.plot(ab, conf_fpr, "o-", color="#E74C3C", lw=2, ms=5, label="FPR (confident only)")
    ax.axhline(full_fpr[0] if full_fpr else 0, color="gray", ls="--", lw=1.5,
               label=f"FPR no abstention ({full_fpr[0]:.1%})" if full_fpr else "Baseline FPR")
    ax.set_xlabel("Abstention rate (fraction flagged for review)", fontsize=11)
    ax.set_ylabel("False Positive Rate", fontsize=11)
    ax.set_title("FPR Reduction via AQCP Abstention\n"
                 "Uncertain predictions flagged for astronomer review", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Abstention curve saved → %s", save_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 — AQCP uncertainty quantification")
    p.add_argument("--config", default=str(_DEFAULT_CONFIG))
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--model-type", choices=["quantum", "classical"], default="quantum")
    p.add_argument("--quantum-mode", choices=["vqc_head", "qonn_attn"], default="vqc_head")
    p.add_argument("--val-csv", required=True)
    p.add_argument("--data-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figures = Path(cfg.get("figures_dir", "figures/"))
    results = Path(cfg.get("results_dir", "results/uq/"))
    results.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s  (type=%s)", args.checkpoint, args.model_type)
    model = _load_model(args.checkpoint, args.model_type, args.quantum_mode)

    dataset = ExoplanetDataset(args.val_csv, args.data_dir, transform=_ResizeTo224(), skip_missing=True)
    cal_ds, test_ds = _split_val(dataset, cfg.get("cal_fraction", 0.5), cfg.get("seed", 42))
    batch = 8
    cal_loader  = DataLoader(cal_ds,  batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False)
    logger.info("Cal: %d  Test: %d", len(cal_ds), len(test_ds))

    alpha   = cfg.get("alpha", 0.10)
    n_shots = cfg.get("n_shots", 20)
    lambda_ = cfg.get("lambda_", 1.0)

    # ── Single-alpha calibration + abstention stats ───────────────
    scorer    = AdaptiveNonconformityScorer(n_shots=n_shots, lambda_=lambda_)
    predictor = ConformalPredictor(model, scorer, alpha=alpha)
    cal_result = predictor.calibrate(cal_loader, device)
    pred_sets  = predictor.predict(test_loader, device)
    test_labels = _collect_labels(test_loader)
    emp_cov    = ConformalPredictor.empirical_coverage(pred_sets, test_labels)
    abs_stats  = ConformalPredictor.abstention_stats(pred_sets, test_labels)
    ece        = expected_calibration_error(model, test_loader, device, n_bins=cfg.get("ece_n_bins", 10))

    report = format_calibration_report(cal_result, emp_cov, abs_stats, ece)
    logger.info("\n%s", report)

    report_path = results / "uq_report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved → %s", report_path)

    # ── Coverage curve sweep ──────────────────────────────────────
    step  = cfg.get("alpha_step", 0.05)
    amin  = cfg.get("alpha_min", 0.05)
    amax  = cfg.get("alpha_max", 0.50)
    alphas = [round(a, 2) for a in np.arange(amin, amax + step / 2, step).tolist()]

    logger.info("Running coverage sweep over %d alpha levels…", len(alphas))
    cov_data = coverage_across_alphas(model, cal_loader, test_loader, device,
                                       alphas=alphas, n_shots=n_shots, lambda_=lambda_)
    _plot_coverage_curve(cov_data, str(figures / "calibration_curve.png"))
    (results / "coverage_sweep.json").write_text(json.dumps(cov_data, indent=2))

    # ── Abstention curve sweep ────────────────────────────────────
    logger.info("Running abstention sweep…")
    ab_alphas = [round(a, 2) for a in np.arange(0.01, 0.50, 0.03).tolist()]
    ab_data   = abstention_curve(model, cal_loader, test_loader, device,
                                  alphas=ab_alphas, n_shots=n_shots, lambda_=lambda_)
    _plot_abstention_curve(ab_data, str(figures / "abstention_curve.png"))
    (results / "abstention_sweep.json").write_text(json.dumps(ab_data, indent=2))

    logger.info("Phase 4 complete. Figures in %s, results in %s", figures, results)


if __name__ == "__main__":
    main()
