"""Phase 2 training script — fine-tune ExoplanetViT on Kepler DR25.

Runs the full Phase 2 pipeline:
    1. Load ExoplanetDataset (2-channel RP+GADF images, 64×64)
    2. Apply resize transform to 224×224 for ViT-B/16
    3. Fine-tune ExoplanetViT with dual-task loss (classification + regression)
    4. Optional 5-fold cross-validation
    5. Print results summary and save best checkpoint

Usage
-----
# Single train/val split
python scripts/train_vit.py \\
    --train-csv data/splits/train.csv \\
    --val-csv   data/splits/val.csv \\
    --data-dir  data/processed

# 5-fold cross-validation (uses full dataset, ignores --val-csv)
python scripts/train_vit.py \\
    --train-csv data/splits/train.csv \\
    --data-dir  data/processed \\
    --kfold

# Skip W&B (dry run)
python scripts/train_vit.py ... --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize, InterpolationMode

# Project root on sys.path so src.* imports work when running from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ExoplanetDataset
from src.models.vit_model import ExoplanetViT
from src.training.vit_trainer import ViTTrainer
from src.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default config location (relative to repo root)
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "vit_config.yaml"


# ---------------------------------------------------------------------------
# Resize transform wrapper — images in dataset are (2, 64, 64); ViT needs 224
# ---------------------------------------------------------------------------

class _ResizeTo224:
    """Callable transform: (2, H, W) float32 tensor → (2, 224, 224)."""

    def __init__(self, size: int = 224) -> None:
        self._resize = Resize(
            (size, size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._resize(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_dataset(csv_path: str, data_dir: str, image_size: int, skip_missing: bool = True) -> ExoplanetDataset:
    transform = _ResizeTo224(image_size) if image_size != 64 else None
    return ExoplanetDataset(
        split_csv_path=csv_path,
        processed_dir=data_dir,
        transform=transform,
        skip_missing=skip_missing,
    )


def _make_loader(dataset: ExoplanetDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _print_summary(label: str, metrics_list: list) -> None:
    """Print mean ± std for a list of EpochMetrics across folds."""
    import statistics

    recalls = [m.planet_recall for m in metrics_list]
    accs = [m.accuracy for m in metrics_list]
    losses = [m.loss for m in metrics_list]

    def _fmt(vals: list[float]) -> str:
        if len(vals) == 1:
            return f"{vals[0]:.4f}"
        return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"

    logger.info("=" * 52)
    logger.info("  %s RESULTS", label)
    logger.info("  Planet recall : %s", _fmt(recalls))
    logger.info("  Accuracy      : %s", _fmt(accs))
    logger.info("  Val loss      : %s", _fmt(losses))
    logger.info("=" * 52)


# ---------------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------------

def train_single_split(
    cfg: dict,
    train_csv: str,
    val_csv: str,
    data_dir: str,
    use_wandb: bool,
) -> None:
    """Fine-tune on a fixed train/val split and save best checkpoint."""
    image_size: int = cfg.get("image_size", 224)

    train_ds = _build_dataset(train_csv, data_dir, image_size)
    val_ds = _build_dataset(val_csv, data_dir, image_size)

    logger.info("Train samples: %d  |  Val samples: %d", len(train_ds), len(val_ds))

    train_loader = _make_loader(train_ds, cfg["batch_size"], shuffle=True)
    val_loader = _make_loader(val_ds, cfg["batch_size"], shuffle=False)

    model = ExoplanetViT(
        pretrained=cfg.get("pretrained", True),
        regression_head=cfg.get("regression_head", True),
        freeze_backbone=cfg.get("freeze_backbone", False),
        aux_dim=cfg.get("aux_dim", 5),
    )

    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        checkpoint_dir=cfg.get("checkpoint_dir", "models/vit/"),
        use_wandb=use_wandb,
        use_class_weights=cfg.get("use_class_weights", True),
        use_regression=cfg.get("regression_head", True),
    )

    result = trainer.train()
    final_metrics = trainer.evaluate(val_loader)

    _print_summary("SINGLE SPLIT", [final_metrics])
    logger.info(
        "Training complete — epochs trained: %d  best F1: %.4f  best recall: %.4f",
        result["epochs_trained"],
        result["best_f1"],
        result["best_planet_recall"],
    )


def train_kfold(
    cfg: dict,
    train_csv: str,
    data_dir: str,
    use_wandb: bool,
) -> None:
    """Run 5-fold cross-validation on the training split."""
    k: int = cfg.get("kfold", 5)
    image_size: int = cfg.get("image_size", 224)

    dataset = _build_dataset(train_csv, data_dir, image_size)
    logger.info("Full dataset: %d samples  |  %d-fold CV", len(dataset), k)

    # Create a dummy val loader for ViTTrainer init (unused in kfold_cv)
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_sub, val_sub = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )
    train_loader = _make_loader(train_sub, cfg["batch_size"], shuffle=True)
    val_loader = _make_loader(val_sub, cfg["batch_size"], shuffle=False)

    model = ExoplanetViT(
        pretrained=cfg.get("pretrained", True),
        regression_head=cfg.get("regression_head", True),
        freeze_backbone=cfg.get("freeze_backbone", False),
        aux_dim=cfg.get("aux_dim", 5),
    )

    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        checkpoint_dir=cfg.get("checkpoint_dir", "models/vit/"),
        use_wandb=use_wandb,
        use_class_weights=cfg.get("use_class_weights", True),
        use_regression=cfg.get("regression_head", True),
    )

    fold_metrics = trainer.kfold_cv(dataset, k=k, max_epochs=cfg.get("max_epochs"))
    _print_summary(f"{k}-FOLD CV", fold_metrics)

    # Persist per-fold metrics as JSON for paper tables
    out_path = Path(cfg.get("checkpoint_dir", "models/vit/")) / "kfold_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "fold": i,
            "planet_recall": m.planet_recall,
            "accuracy": m.accuracy,
            "val_loss": m.loss,
        }
        for i, m in enumerate(fold_metrics)
    ]
    out_path.write_text(json.dumps(records, indent=2))
    logger.info("Per-fold results saved → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 — Fine-tune ExoplanetViT on Kepler DR25"
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to vit_config.yaml (default: configs/vit_config.yaml)",
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        help="CSV with kepoi_name + koi_disposition for training samples",
    )
    parser.add_argument(
        "--val-csv",
        default=None,
        help="Validation split CSV (required unless --kfold is set)",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory of processed samples (contains per-KOI subdirs)",
    )
    parser.add_argument(
        "--kfold",
        action="store_true",
        help="Run 5-fold CV on train-csv instead of a single train/val split",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    set_seed(cfg.get("seed", 42))
    logger.info("Config loaded from %s", args.config)
    logger.info("Device: %s", "cuda" if torch.cuda.is_available() else "cpu")

    use_wandb = not args.no_wandb

    if args.kfold:
        train_kfold(cfg, args.train_csv, args.data_dir, use_wandb)
    else:
        if args.val_csv is None:
            parser_err = (
                "--val-csv is required for single-split training. "
                "Use --kfold to run cross-validation instead."
            )
            logger.error(parser_err)
            sys.exit(1)
        train_single_split(cfg, args.train_csv, args.val_csv, args.data_dir, use_wandb)


if __name__ == "__main__":
    main()
