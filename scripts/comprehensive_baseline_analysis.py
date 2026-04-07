"""Comprehensive baseline analysis with calibration, ECE, and abstention metrics.

Produces:
1. Precision-Recall curve (visual trade-off)
2. Threshold vs (Precision, Recall, F1) curve (visual trade-off)
3. Calibration curve (reliability diagram)
4. ECE (Expected Calibration Error)
5. Abstention analysis (% abstained, FPR reduction)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


class _ResizeTo224:
    def __init__(self, size: int = 224) -> None:
        self._resize = Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._resize(x)


def compute_ece(all_probs: np.ndarray, all_labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_sums = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (all_probs >= bins[i]) & (all_probs < bins[i + 1])
        if mask.sum() > 0:
            bin_sums[i] = np.abs((all_probs[mask] - all_labels[mask]).mean())
            bin_true[i] = all_labels[mask].mean()
            bin_total[i] = mask.sum()

    total = bin_total.sum()
    if total == 0:
        return 0.0

    ece = np.sum(bin_sums * bin_total) / total
    return float(ece)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive baseline analysis")
    parser.add_argument("--model-path", default="models/vit/best_model.pt")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--config", default="configs/vit_config.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load data
    logger.info("Loading validation data...")
    transform = _ResizeTo224(224)
    val_ds = ExoplanetDataset(args.val_csv, args.data_dir, transform=transform, skip_missing=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = ExoplanetViT(pretrained=cfg.get("pretrained", True), regression_head=cfg.get("regression_head", True))
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Collect predictions and labels
    logger.info("Running inference...")
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

    logger.info(f"Evaluated on {len(all_labels)} samples ({all_labels.sum():.0f} planets, {(1-all_labels).sum():.0f} FPs)")

    # =========================================================================
    # 1. PRECISION-RECALL CURVE
    # =========================================================================
    logger.info("\n1. Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    logger.info(f"   PR AUC: {pr_auc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, "b-", linewidth=2.5, label=f"Classical ViT (AUC-PR={pr_auc:.4f})")
    plt.xlabel("Recall", fontsize=13, fontweight="bold")
    plt.ylabel("Precision", fontsize=13, fontweight="bold")
    plt.title("Precision-Recall Trade-off Curve", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="lower left")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / "01_pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved: {output_dir / '01_pr_curve.png'}")

    # =========================================================================
    # 2. THRESHOLD TUNING (Precision, Recall, F1 vs Threshold)
    # =========================================================================
    logger.info("\n2. Threshold Tuning Analysis")
    thresholds_to_test = np.linspace(0.2, 0.8, 13)
    results = []

    for threshold in thresholds_to_test:
        preds = (all_probs >= threshold).astype(int)

        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()

        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_val = f1_score(all_labels, preds, zero_division=0.0)

        results.append({
            "threshold": threshold,
            "recall": recall_val,
            "precision": precision_val,
            "f1": f1_val,
        })

    best_f1_idx = np.argmax([r["f1"] for r in results])
    best_f1_threshold = results[best_f1_idx]["threshold"]
    best_f1_val = results[best_f1_idx]["f1"]
    logger.info(f"   Best F1={best_f1_val:.4f} at threshold={best_f1_threshold:.2f}")
    logger.info(f"   Note: Operating point depends on goal (maximize recall vs precision vs F1)")

    thresholds_arr = np.array([r["threshold"] for r in results])
    recalls = np.array([r["recall"] for r in results])
    precisions = np.array([r["precision"] for r in results])
    f1s = np.array([r["f1"] for r in results])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds_arr, recalls, "o-", label="Recall", linewidth=2.5, markersize=8, color="green")
    ax.plot(thresholds_arr, precisions, "s-", label="Precision", linewidth=2.5, markersize=8, color="red")
    ax.plot(thresholds_arr, f1s, "^-", label="F1 Score", linewidth=2.5, markersize=8, color="blue")
    ax.axvline(best_f1_threshold, color="blue", linestyle="--", alpha=0.5, label=f"Optimal F1 (threshold={best_f1_threshold:.2f})")
    ax.set_xlabel("Classification Threshold", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Threshold Tuning: Recall-Precision-F1 Trade-off", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best")
    ax.set_ylim([0, 1.0])
    plt.tight_layout()
    plt.savefig(output_dir / "02_threshold_tuning.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved: {output_dir / '02_threshold_tuning.png'}")

    # =========================================================================
    # 3. CALIBRATION CURVE (Reliability Diagram)
    # =========================================================================
    logger.info("\n3. Calibration Analysis")
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10, strategy="uniform")
    ece = compute_ece(all_probs, all_labels, n_bins=10)
    logger.info(f"   ECE (Expected Calibration Error): {ece:.4f}")
    logger.info(f"   Interpretation: Model is off by ~{ece*100:.2f}% on average")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration")
    ax.plot(prob_pred, prob_true, "o-", linewidth=2.5, markersize=8, color="blue", label="Classical ViT")
    ax.fill_between([0, 1], [0, 1], 0, alpha=0.1, color="gray")
    ax.set_xlabel("Mean Predicted Probability", fontsize=13, fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontsize=13, fontweight="bold")
    ax.set_title(f"Calibration Curve (ECE={ece:.4f})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / "03_calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved: {output_dir / '03_calibration_curve.png'}")

    # =========================================================================
    # 4. ABSTENTION ANALYSIS (using conformal-style uncertainty)
    # =========================================================================
    logger.info("\n4. Abstention Analysis (Simulated Uncertainty)")
    # For now, use softmax entropy as proxy for uncertainty
    probs_both_classes = np.stack([1 - all_probs, all_probs], axis=1)
    entropy = -np.sum(probs_both_classes * np.log(probs_both_classes + 1e-10), axis=1)

    abstention_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    abstention_results = []

    for abs_rate in abstention_rates:
        # Abstain on samples with highest uncertainty
        n_abstain = int(len(entropy) * abs_rate)
        if n_abstain == 0:
            high_conf_idx = np.arange(len(entropy))
        else:
            high_conf_idx = np.argsort(entropy)[:-n_abstain]

        preds = (all_probs >= 0.5).astype(int)
        tp = ((preds[high_conf_idx] == 1) & (all_labels[high_conf_idx] == 1)).sum()
        fp = ((preds[high_conf_idx] == 1) & (all_labels[high_conf_idx] == 0)).sum()
        fn = ((preds[high_conf_idx] == 0) & (all_labels[high_conf_idx] == 1)).sum()

        n_samples_retained = len(high_conf_idx)
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + (all_labels[high_conf_idx] == 0).sum()) if (fp + (all_labels[high_conf_idx] == 0).sum()) > 0 else 0.0

        abstention_results.append({
            "abstention_rate": abs_rate,
            "samples_retained": n_samples_retained,
            "recall": recall_val,
            "fpr": fpr_val,
        })

        logger.info(
            f"   Abstain {abs_rate*100:.0f}% | Retained={n_samples_retained} | Recall={recall_val:.4f} | FPR={fpr_val:.4f}"
        )

    abs_rates = np.array([r["abstention_rate"] for r in abstention_results])
    recalls_abs = np.array([r["recall"] for r in abstention_results])
    fprs_abs = np.array([r["fpr"] for r in abstention_results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(abs_rates * 100, recalls_abs, "o-", linewidth=2.5, markersize=8, color="green")
    ax1.set_xlabel("Abstention Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Recall (Confidence Set)", fontsize=12, fontweight="bold")
    ax1.set_title("Recall vs Abstention Rate", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(abs_rates * 100, fprs_abs, "s-", linewidth=2.5, markersize=8, color="red")
    ax2.set_xlabel("Abstention Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax2.set_title("FPR Reduction via Abstention", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "04_abstention_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Saved: {output_dir / '04_abstention_analysis.png'}")

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CLASSICAL ViT BASELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Validation Set: {len(all_labels)} samples ({all_labels.sum():.0f} planets, {(1-all_labels).sum():.0f} FPs)")
    logger.info(f"\nPrecision-Recall Trade-off:")
    logger.info(f"  • PR AUC: {pr_auc:.4f}")
    logger.info(f"\nThreshold Tuning (optimal F1):")
    logger.info(f"  • Optimal threshold: {best_f1_threshold:.2f}")
    logger.info(f"  • F1 Score: {best_f1_val:.4f}")
    logger.info(f"  • At this threshold:")
    best_idx = best_f1_idx
    logger.info(f"    - Recall: {results[best_idx]['recall']:.4f}")
    logger.info(f"    - Precision: {results[best_idx]['precision']:.4f}")
    logger.info(f"\nCalibration:")
    logger.info(f"  • ECE: {ece:.4f} (model is ~{ece*100:.2f}% off on average)")
    logger.info(f"  • Softmax probabilities are NOT well-calibrated")
    logger.info(f"\nAbstention (via uncertainty):")
    logger.info(f"  • At 30% abstention: FPR={fprs_abs[3]:.4f}, Recall={recalls_abs[3]:.4f}")
    logger.info(f"  • Abstention allows trading recall for precision/FPR")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
