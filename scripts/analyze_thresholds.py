"""Threshold tuning and PR curve analysis for classical ViT baseline.

Computes:
- Precision-Recall curve (AUC-PR)
- F1, Recall, Precision across thresholds [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
- Optimal operating points
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
    """Resize transform for ViT input."""
    def __init__(self, size: int = 224) -> None:
        self._resize = Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._resize(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold tuning and PR curve analysis")
    parser.add_argument("--model-path", default="models/vit/best_model.pt", help="Path to trained model")
    parser.add_argument("--val-csv", default="data/splits/val.csv", help="Validation CSV")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    parser.add_argument("--config", default="configs/vit_config.yaml", help="Config file")
    parser.add_argument("--output-dir", default="results", help="Output directory for plots")
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
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Probability of class 1 (planet)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    logger.info(f"Evaluated on {len(all_labels)} samples")

    # Compute PR curve
    logger.info("Computing Precision-Recall curve...")
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    logger.info(f"PR AUC: {pr_auc:.4f}")

    # Threshold tuning
    logger.info("\nThreshold analysis:")
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    for threshold in thresholds_to_test:
        preds = (all_probs >= threshold).astype(int)

        # Compute metrics
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        tn = ((preds == 0) & (all_labels == 0)).sum()

        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_val = f1_score(all_labels, preds, zero_division=0.0)

        results.append({
            "threshold": threshold,
            "recall": recall_val,
            "precision": precision_val,
            "f1": f1_val,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

        logger.info(
            f"Threshold {threshold:.1f} | Recall={recall_val:.4f} | Precision={precision_val:.4f} | F1={f1_val:.4f}"
        )

    # Find best thresholds
    best_f1 = max(results, key=lambda x: x["f1"])
    best_recall = max(results, key=lambda x: x["recall"])
    best_precision = max(results, key=lambda x: x["precision"])

    logger.info(f"\nBest F1 (threshold={best_f1['threshold']:.1f}): {best_f1['f1']:.4f}")
    logger.info(f"Best Recall (threshold={best_recall['threshold']:.1f}): {best_recall['recall']:.4f}")
    logger.info(f"Best Precision (threshold={best_precision['threshold']:.1f}): {best_precision['precision']:.4f}")

    # Plot PR curve
    logger.info(f"\nSaving PR curve to {output_dir / 'pr_curve.png'}")
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC={pr_auc:.4f})", linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve (Classical ViT)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close()

    # Plot threshold tuning
    logger.info(f"Saving threshold analysis to {output_dir / 'threshold_tuning.png'}")
    thresholds_arr = [r["threshold"] for r in results]
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]
    f1s = [r["f1"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds_arr, recalls, "o-", label="Recall", linewidth=2, markersize=8)
    ax.plot(thresholds_arr, precisions, "s-", label="Precision", linewidth=2, markersize=8)
    ax.plot(thresholds_arr, f1s, "^-", label="F1 Score", linewidth=2, markersize=8)
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Tuning: Recall vs Precision vs F1", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_tuning.png", dpi=150)
    plt.close()

    # Save results to CSV
    import csv
    csv_path = output_dir / "threshold_results.csv"
    logger.info(f"Saving results to {csv_path}")
    with open(csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "recall", "precision", "f1", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(results)

    logger.info("Done!")


if __name__ == "__main__":
    main()
