"""Compare classical vs quantum ViT using identical analysis framework.

Produces side-by-side comparison:
- PR curves
- Calibration curves
- Abstention analysis
- Summary metrics table
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, f1_score
from torchvision.transforms import Resize, InterpolationMode

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ExoplanetDataset
from src.models.vit_model import ExoplanetViT
from src.models.quantum_vit import ExoplanetQuantumViT
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


class _ResizeTo224:
    def __init__(self, size: int = 224) -> None:
        self._resize = Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._resize(x)


class ModelMetrics(NamedTuple):
    pr_auc: float
    best_f1: float
    best_f1_threshold: float
    best_f1_recall: float
    best_f1_precision: float
    ece: float
    fpr_at_30pct_abstention: float
    recall_at_30pct_abstention: float


def compute_ece(all_probs: np.ndarray, all_labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_sums = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (all_probs >= bins[i]) & (all_probs < bins[i + 1])
        if mask.sum() > 0:
            bin_sums[i] = np.abs((all_probs[mask] - all_labels[mask]).mean())
            bin_total[i] = mask.sum()

    total = bin_total.sum()
    return float(np.sum(bin_sums * bin_total) / max(total, 1))


def evaluate_model(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, model_name: str) -> tuple[np.ndarray, np.ndarray, ModelMetrics]:
    """Run inference and compute all metrics."""
    logger.info(f"[{model_name}] Running inference...")
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for image, aux, labels in val_loader:
            image = image.to(device)
            aux = aux.to(device)
            labels = labels.to(device).long()

            logits, _ = model(image, aux)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # PR AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    # Threshold tuning
    thresholds_to_test = np.linspace(0.2, 0.8, 13)
    best_f1 = 0.0
    best_f1_threshold = 0.5
    best_f1_recall = 0.0
    best_f1_precision = 0.0

    for threshold in thresholds_to_test:
        preds = (all_probs >= threshold).astype(int)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()

        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_val = f1_score(all_labels, preds, zero_division=0.0)

        if f1_val > best_f1:
            best_f1 = f1_val
            best_f1_threshold = threshold
            best_f1_recall = recall_val
            best_f1_precision = precision_val

    # ECE
    ece = compute_ece(all_probs, all_labels, n_bins=10)

    # Abstention analysis
    probs_both_classes = np.stack([1 - all_probs, all_probs], axis=1)
    entropy = -np.sum(probs_both_classes * np.log(probs_both_classes + 1e-10), axis=1)

    n_abstain = int(len(entropy) * 0.3)
    high_conf_idx = np.argsort(entropy)[:-n_abstain] if n_abstain > 0 else np.arange(len(entropy))

    preds = (all_probs >= 0.5).astype(int)
    tp = ((preds[high_conf_idx] == 1) & (all_labels[high_conf_idx] == 1)).sum()
    fp = ((preds[high_conf_idx] == 1) & (all_labels[high_conf_idx] == 0)).sum()
    fn = ((preds[high_conf_idx] == 0) & (all_labels[high_conf_idx] == 1)).sum()

    recall_abs = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr_abs = fp / (fp + (all_labels[high_conf_idx] == 0).sum()) if (fp + (all_labels[high_conf_idx] == 0).sum()) > 0 else 0.0

    metrics = ModelMetrics(
        pr_auc=pr_auc,
        best_f1=best_f1,
        best_f1_threshold=best_f1_threshold,
        best_f1_recall=best_f1_recall,
        best_f1_precision=best_f1_precision,
        ece=ece,
        fpr_at_30pct_abstention=fpr_abs,
        recall_at_30pct_abstention=recall_abs,
    )

    logger.info(f"[{model_name}] PR-AUC={pr_auc:.4f} | Best F1={best_f1:.4f} @ threshold={best_f1_threshold:.2f}")
    logger.info(f"[{model_name}] ECE={ece:.4f} | Abstention FPR={fpr_abs:.4f}, Recall={recall_abs:.4f}")

    return all_probs, all_labels, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparative analysis: Classical vs Quantum ViT")
    parser.add_argument("--classical-model", default="models/vit/best_model.pt")
    parser.add_argument("--quantum-model", default="models/quantum_vit/best_model.pt")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--config", default="configs/vit_config.yaml")
    parser.add_argument("--output-dir", default="results/comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load data
    logger.info("Loading validation data...")
    transform = _ResizeTo224(224)
    val_ds = ExoplanetDataset(args.val_csv, args.data_dir, transform=transform, skip_missing=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classical model
    logger.info(f"\nLoading classical ViT from {args.classical_model}")
    classical_model = ExoplanetViT(pretrained=cfg.get("pretrained", True), regression_head=cfg.get("regression_head", True))
    checkpoint = torch.load(args.classical_model, map_location="cpu")
    classical_model.load_state_dict(checkpoint)
    classical_model.to(device)

    # Load quantum model
    logger.info(f"Loading quantum ViT from {args.quantum_model}")
    quantum_model = ExoplanetQuantumViT(quantum_mode="qonn_attn", pretrained=cfg.get("pretrained", True), regression_head=cfg.get("regression_head", True))
    checkpoint = torch.load(args.quantum_model, map_location="cpu")
    quantum_model.load_state_dict(checkpoint)
    quantum_model.to(device)

    # Evaluate both models
    classical_probs, classical_labels, classical_metrics = evaluate_model(classical_model, val_loader, device, "Classical ViT")
    quantum_probs, quantum_labels, quantum_metrics = evaluate_model(quantum_model, val_loader, device, "Quantum ViT")

    # =========================================================================
    # SIDE-BY-SIDE PR CURVES
    # =========================================================================
    logger.info("\nGenerating PR curve comparison...")
    precision_c, recall_c, _ = precision_recall_curve(classical_labels, classical_probs)
    precision_q, recall_q, _ = precision_recall_curve(quantum_labels, quantum_probs)

    plt.figure(figsize=(12, 7))
    plt.plot(recall_c, precision_c, "b-", linewidth=2.5, label=f"Classical ViT (AUC-PR={classical_metrics.pr_auc:.4f})")
    plt.plot(recall_q, precision_q, "r--", linewidth=2.5, label=f"Quantum ViT (AUC-PR={quantum_metrics.pr_auc:.4f})")
    plt.xlabel("Recall", fontsize=13, fontweight="bold")
    plt.ylabel("Precision", fontsize=13, fontweight="bold")
    plt.title("Precision-Recall Trade-off: Classical vs Quantum", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="lower left")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / "01_pr_curve_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / '01_pr_curve_comparison.png'}")

    # =========================================================================
    # SIDE-BY-SIDE CALIBRATION CURVES
    # =========================================================================
    logger.info("Generating calibration curve comparison...")
    prob_true_c, prob_pred_c = calibration_curve(classical_labels, classical_probs, n_bins=10, strategy="uniform")
    prob_true_q, prob_pred_q = calibration_curve(quantum_labels, quantum_probs, n_bins=10, strategy="uniform")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect")
    ax1.plot(prob_pred_c, prob_true_c, "o-", linewidth=2.5, markersize=8, color="blue", label="Classical ViT")
    ax1.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Fraction of Positives", fontsize=11, fontweight="bold")
    ax1.set_title(f"Classical ViT (ECE={classical_metrics.ece:.4f})", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax2.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect")
    ax2.plot(prob_pred_q, prob_true_q, "s-", linewidth=2.5, markersize=8, color="red", label="Quantum ViT")
    ax2.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Fraction of Positives", fontsize=11, fontweight="bold")
    ax2.set_title(f"Quantum ViT (ECE={quantum_metrics.ece:.4f})", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / "02_calibration_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / '02_calibration_comparison.png'}")

    # =========================================================================
    # METRICS COMPARISON TABLE
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("COMPARATIVE METRICS TABLE")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<40} {'Classical ViT':<20} {'Quantum ViT':<20}")
    logger.info("-" * 80)
    logger.info(f"{'PR AUC':<40} {classical_metrics.pr_auc:<20.4f} {quantum_metrics.pr_auc:<20.4f}")
    logger.info(f"{'Best F1 Score':<40} {classical_metrics.best_f1:<20.4f} {quantum_metrics.best_f1:<20.4f}")
    logger.info(f"{'Best F1 Threshold':<40} {classical_metrics.best_f1_threshold:<20.2f} {quantum_metrics.best_f1_threshold:<20.2f}")
    logger.info(f"{'  @ Recall':<40} {classical_metrics.best_f1_recall:<20.4f} {quantum_metrics.best_f1_recall:<20.4f}")
    logger.info(f"{'  @ Precision':<40} {classical_metrics.best_f1_precision:<20.4f} {quantum_metrics.best_f1_precision:<20.4f}")
    logger.info(f"{'ECE (Expected Calibration Error)':<40} {classical_metrics.ece:<20.4f} {quantum_metrics.ece:<20.4f}")
    logger.info(f"{'FPR @ 30% abstention':<40} {classical_metrics.fpr_at_30pct_abstention:<20.4f} {quantum_metrics.fpr_at_30pct_abstention:<20.4f}")
    logger.info(f"{'Recall @ 30% abstention':<40} {classical_metrics.recall_at_30pct_abstention:<20.4f} {quantum_metrics.recall_at_30pct_abstention:<20.4f}")
    logger.info("=" * 80)

    # Save results
    import csv
    results_csv = output_dir / "metrics_comparison.csv"
    with open(results_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "metric", "classical_vit", "quantum_vit", "improvement"
        ])
        writer.writeheader()
        writer.writerow({
            "metric": "PR AUC",
            "classical_vit": f"{classical_metrics.pr_auc:.4f}",
            "quantum_vit": f"{quantum_metrics.pr_auc:.4f}",
            "improvement": f"{(quantum_metrics.pr_auc - classical_metrics.pr_auc):.4f}",
        })
        writer.writerow({
            "metric": "Best F1",
            "classical_vit": f"{classical_metrics.best_f1:.4f}",
            "quantum_vit": f"{quantum_metrics.best_f1:.4f}",
            "improvement": f"{(quantum_metrics.best_f1 - classical_metrics.best_f1):.4f}",
        })
        writer.writerow({
            "metric": "ECE",
            "classical_vit": f"{classical_metrics.ece:.4f}",
            "quantum_vit": f"{quantum_metrics.ece:.4f}",
            "improvement": f"{(classical_metrics.ece - quantum_metrics.ece):.4f}",
        })

    logger.info(f"\nResults saved to {results_csv}")


if __name__ == "__main__":
    main()
