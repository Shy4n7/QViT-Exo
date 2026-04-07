"""Phase 3 training script — hybrid quantum-classical ExoplanetQuantumViT.

Trains the quantum ViT on Kepler DR25 with:
  - Gradient variance monitoring per epoch (barren plateau detection)
  - W&B logging of quantum parameter gradient statistics
  - Identical CLI to train_vit.py for easy A/B comparison

Usage
-----
# VQC head variant (Unlu 2024), single split
python scripts/train_quantum_vit.py \\
    --train-csv data/splits/train.csv \\
    --val-csv   data/splits/val.csv \\
    --data-dir  data/processed \\
    --quantum-mode vqc_head

# QONN attention variant (Tesi 2024), 5-fold CV
python scripts/train_quantum_vit.py \\
    --train-csv data/splits/train.csv \\
    --data-dir  data/processed \\
    --quantum-mode qonn_attn \\
    --kfold

# Dry run (no W&B)
python scripts/train_quantum_vit.py ... --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import InterpolationMode, Resize

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ExoplanetDataset
from src.models.quantum_vit import ExoplanetQuantumViT
from src.training.vit_trainer import EpochMetrics, ViTTrainer
from src.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "quantum_vit_config.yaml"
_BARREN_PLATEAU_CONSECUTIVE = 3   # warn after N epochs below threshold


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class _ResizeTo224:
    def __init__(self, size: int = 224) -> None:
        self._resize = Resize(
            (size, size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._resize(x)


# ---------------------------------------------------------------------------
# Gradient variance monitoring
# ---------------------------------------------------------------------------

class BarrenPlateauMonitor:
    """Tracks quantum parameter gradient variance across epochs.

    Computes variance of gradients for quantum circuit parameters only.
    When variance stays below ``threshold`` for ``patience`` consecutive
    epochs, logs a warning (barren plateau suspected).

    Parameters
    ----------
    quantum_params : iterable of Parameter  Quantum-only parameters.
    threshold      : float  Gradient variance below this → plateau warning.
    patience       : int    Consecutive epochs below threshold to trigger warn.
    """

    def __init__(
        self,
        quantum_params: list[nn.Parameter],
        threshold: float = 1e-6,
        patience: int = _BARREN_PLATEAU_CONSECUTIVE,
    ) -> None:
        self._params = quantum_params
        self._threshold = threshold
        self._patience = patience
        self._below_count = 0
        self.history: list[float] = []

    def step(self) -> float:
        """Compute gradient variance after a backward pass.

        Returns
        -------
        float  Variance of all quantum parameter gradients, or 0.0 if none.
        """
        grads = [
            p.grad.flatten()
            for p in self._params
            if p.grad is not None
        ]
        if not grads:
            return 0.0

        all_grads = torch.cat(grads)
        variance = all_grads.var().item()
        self.history.append(variance)

        if variance < self._threshold:
            self._below_count += 1
            if self._below_count >= self._patience:
                logger.warning(
                    "Barren plateau suspected: quantum gradient variance "
                    "%.2e < %.2e for %d consecutive epochs. "
                    "Consider switching to qonn_attn or reducing n_qubits.",
                    variance,
                    self._threshold,
                    self._patience,
                )
        else:
            self._below_count = 0

        return variance


# ---------------------------------------------------------------------------
# Quantum-aware training loop
# ---------------------------------------------------------------------------

class QuantumViTTrainer:
    """Training loop for ExoplanetQuantumViT with barren plateau monitoring.

    Wraps the existing ViTTrainer for k-fold CV and checkpoint saving, but
    adds per-epoch gradient variance logging for the quantum parameters.

    The W&B log includes:
        quantum_grad_variance  — gradient variance of quantum circuit params
        train_loss             — total training loss
        val_loss / val_recall  — validation metrics
    """

    def __init__(
        self,
        model:           ExoplanetQuantumViT,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        config:          dict,
        checkpoint_dir:  str = "models/quantum_vit/",
        use_wandb:       bool = True,
        use_class_weights: bool = True,
        use_regression:  bool = True,
    ) -> None:
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._config = config
        self._checkpoint_dir = Path(checkpoint_dir)
        self._use_wandb = use_wandb
        self._use_class_weights = use_class_weights
        self._use_regression = use_regression
        self._regression_loss_weight: float = config.get("regression_loss_weight", 0.1)
        self._barren_threshold: float = config.get("barren_plateau_threshold", 1e-6)

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)

        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=config.get("lr", 5e-5)
        )

        if use_class_weights:
            weights = self._compute_class_weights().to(self._device)
            self._criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self._criterion = nn.CrossEntropyLoss()

        self._plateau_monitor = BarrenPlateauMonitor(
            quantum_params=model.quantum_parameters,
            threshold=self._barren_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, max_epochs: int | None = None) -> dict:
        """Run full training loop with gradient variance monitoring.

        Returns
        -------
        dict  {"epochs_trained": int, "best_val_loss": float,
               "final_quantum_grad_var": float}
        """
        n_epochs = max_epochs or self._config.get("max_epochs", 20)
        patience = self._config.get("patience", 5)
        min_delta = self._config.get("min_delta", 0.0)

        if self._use_wandb:
            import wandb
            wandb.init(config=self._config)
            wandb.watch(self._model, log="gradients")

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        final_grad_var = 0.0

        for epoch in range(1, n_epochs + 1):
            train_loss, grad_var = self._train_one_epoch()
            val_metrics = self._evaluate()

            if self._use_wandb:
                import wandb
                wandb.log({
                    "epoch":                  epoch,
                    "train_loss":             train_loss,
                    "val_loss":               val_metrics.loss,
                    "val_accuracy":           val_metrics.accuracy,
                    "val_planet_recall":      val_metrics.planet_recall,
                    "quantum_grad_variance":  grad_var,
                })

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "recall=%.4f | q_grad_var=%.2e",
                epoch, n_epochs, train_loss, val_metrics.loss,
                val_metrics.planet_recall, grad_var,
            )

            final_grad_var = grad_var
            improved = val_metrics.loss < best_val_loss - min_delta
            if improved:
                best_val_loss = val_metrics.loss
                epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics.loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        if self._use_wandb:
            import wandb
            wandb.finish()

        return {
            "epochs_trained":         n_epochs,
            "best_val_loss":          best_val_loss,
            "final_quantum_grad_var": final_grad_var,
        }

    def evaluate(self, loader: DataLoader) -> EpochMetrics:
        """Evaluate on a DataLoader — delegates to ViTTrainer.evaluate logic."""
        from sklearn.metrics import f1_score as sklearn_f1

        self._model.eval()
        total_loss = n_samples = n_correct = tp = fn = 0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for image, aux, labels in loader:
                image, aux = image.to(self._device), aux.to(self._device)
                labels = labels.to(self._device).long()

                logits, _ = self._model(image, aux)
                loss = self._criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                n_samples  += labels.size(0)
                preds = logits.argmax(dim=1)
                n_correct  += (preds == labels).sum().item()
                pos_mask = labels == 1
                tp += (preds[pos_mask] == 1).sum().item()
                fn += (preds[pos_mask] == 0).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        self._model.train()
        denom = tp + fn
        f1 = sklearn_f1(all_labels, all_preds, average="macro", zero_division=0.0)
        return EpochMetrics(
            loss=total_loss / max(n_samples, 1),
            accuracy=n_correct / max(n_samples, 1),
            planet_recall=tp / denom if denom > 0 else 0.0,
            f1=float(f1),
        )

    def kfold_cv(self, dataset: ExoplanetDataset, k: int = 5) -> list[EpochMetrics]:
        """Run k-fold cross-validation, returning one EpochMetrics per fold."""
        n = len(dataset)
        fold_size = n // k
        indices = list(range(n))
        results = []

        for fold in range(k):
            val_start = fold * fold_size
            val_end   = val_start + fold_size if fold < k - 1 else n
            val_idx   = indices[val_start:val_end]
            train_idx = indices[:val_start] + indices[val_end:]

            fold_train = DataLoader(
                Subset(dataset, train_idx),
                batch_size=self._config.get("batch_size", 8), shuffle=True,
            )
            fold_val = DataLoader(
                Subset(dataset, val_idx),
                batch_size=self._config.get("batch_size", 8), shuffle=False,
            )

            fold_model = deepcopy(self._model)
            fold_trainer = QuantumViTTrainer(
                model=fold_model,
                train_loader=fold_train,
                val_loader=fold_val,
                config=self._config,
                checkpoint_dir=str(self._checkpoint_dir / f"fold_{fold}"),
                use_wandb=False,
                use_class_weights=self._use_class_weights,
                use_regression=self._use_regression,
            )
            fold_trainer.train()
            results.append(fold_trainer.evaluate(fold_val))
            logger.info(
                "Fold %d/%d — recall=%.4f  acc=%.4f",
                fold + 1, k, results[-1].planet_recall, results[-1].accuracy,
            )

        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> tuple[float, float]:
        """Return (mean_train_loss, quantum_grad_variance)."""
        self._model.train()
        total_loss = n_samples = 0

        for image, aux, labels in self._train_loader:
            image, aux = image.to(self._device), aux.to(self._device)
            labels = labels.to(self._device).long()

            self._optimizer.zero_grad()
            logits, reg_preds = self._model(image, aux)
            loss = self._criterion(logits, labels)

            if self._use_regression and reg_preds is not None:
                num_reg = reg_preds.shape[1]
                proxy = labels.float().unsqueeze(-1).expand(-1, num_reg)
                loss = loss + self._config.get("regression_loss_weight", 0.1) * \
                    nn.functional.mse_loss(reg_preds, proxy)

            loss.backward()

            total_loss += loss.item() * labels.size(0)
            n_samples  += labels.size(0)
            self._optimizer.step()

        grad_var = self._plateau_monitor.step()
        return total_loss / max(n_samples, 1), grad_var

    def _evaluate(self) -> EpochMetrics:
        return self.evaluate(self._val_loader)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = self._checkpoint_dir / "best_model.pt"
        torch.save(self._model.state_dict(), str(path))
        logger.info("Checkpoint saved → %s  (epoch=%d  val_loss=%.4f)", path, epoch, val_loss)

    def _compute_class_weights(self) -> torch.Tensor:
        counts = torch.zeros(2, dtype=torch.float32)
        for _, _, labels in self._train_loader:
            for c in range(2):
                counts[c] += (labels == c).sum().item()
        return counts.sum() / (2.0 * counts.clamp(min=1.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_dataset(csv_path: str, data_dir: str, image_size: int, skip_missing: bool = True) -> ExoplanetDataset:
    transform = _ResizeTo224(image_size) if image_size != 64 else None
    return ExoplanetDataset(csv_path, data_dir, transform=transform, skip_missing=skip_missing)


def _print_summary(label: str, metrics_list: list[EpochMetrics]) -> None:
    import statistics
    recalls = [m.planet_recall for m in metrics_list]
    accs    = [m.accuracy for m in metrics_list]
    losses  = [m.loss for m in metrics_list]

    def _fmt(vals: list[float]) -> str:
        return f"{vals[0]:.4f}" if len(vals) == 1 else \
               f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"

    logger.info("=" * 55)
    logger.info("  %s RESULTS", label)
    logger.info("  Planet recall : %s", _fmt(recalls))
    logger.info("  Accuracy      : %s", _fmt(accs))
    logger.info("  Val loss      : %s", _fmt(losses))
    logger.info("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Train hybrid quantum-classical ExoplanetViT"
    )
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--quantum-mode",
        choices=["vqc_head", "qonn_attn"],
        default=None,
        help="Override quantum_mode in config",
    )
    parser.add_argument("--kfold", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = _load_config(args.config)
    if args.quantum_mode:
        cfg["quantum_mode"] = args.quantum_mode

    set_seed(cfg.get("seed", 42))
    quantum_mode = cfg["quantum_mode"]
    image_size   = cfg.get("image_size", 224)
    use_wandb    = not args.no_wandb

    logger.info("Quantum mode  : %s", quantum_mode)
    logger.info("n_qubits      : %d", cfg.get("n_qubits", 4))
    logger.info("n_layers      : %d", cfg.get("n_quantum_layers", 2))
    logger.info("Device        : %s", "cuda" if torch.cuda.is_available() else "cpu")

    model = ExoplanetQuantumViT(
        quantum_mode=quantum_mode,
        n_qubits=cfg.get("n_qubits", 4),
        n_quantum_layers=cfg.get("n_quantum_layers", 2),
        pretrained=cfg.get("pretrained", True),
        regression_head=cfg.get("regression_head", True),
        freeze_backbone=cfg.get("freeze_backbone", True),
        aux_dim=cfg.get("aux_dim", 5),
    )

    if args.kfold:
        dataset = _build_dataset(args.train_csv, args.data_dir, image_size)
        logger.info("Full dataset: %d samples  |  %d-fold CV", len(dataset), cfg["kfold"])

        val_size   = max(1, len(dataset) // 10)
        train_size = len(dataset) - val_size
        train_sub, val_sub = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
        )
        train_loader = DataLoader(train_sub, batch_size=cfg["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_sub,   batch_size=cfg["batch_size"], shuffle=False)

        trainer = QuantumViTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            checkpoint_dir=cfg["checkpoint_dir"],
            use_wandb=use_wandb,
            use_class_weights=cfg.get("use_class_weights", True),
            use_regression=cfg.get("regression_head", True),
        )
        fold_metrics = trainer.kfold_cv(dataset, k=cfg["kfold"])
        _print_summary(f"{cfg['kfold']}-FOLD CV ({quantum_mode})", fold_metrics)

        out_path = Path(cfg["checkpoint_dir"]) / "kfold_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([
            {"fold": i, "planet_recall": m.planet_recall, "accuracy": m.accuracy, "val_loss": m.loss}
            for i, m in enumerate(fold_metrics)
        ], indent=2))
        logger.info("Per-fold results saved → %s", out_path)

    else:
        if args.val_csv is None:
            logger.error("--val-csv required for single-split training (or use --kfold).")
            sys.exit(1)

        train_ds = _build_dataset(args.train_csv, args.data_dir, image_size)
        val_ds   = _build_dataset(args.val_csv,   args.data_dir, image_size)
        logger.info("Train: %d  |  Val: %d", len(train_ds), len(val_ds))

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

        trainer = QuantumViTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            checkpoint_dir=cfg["checkpoint_dir"],
            use_wandb=use_wandb,
            use_class_weights=cfg.get("use_class_weights", True),
            use_regression=cfg.get("regression_head", True),
        )
        result = trainer.train()
        final  = trainer.evaluate(val_loader)
        _print_summary(f"SINGLE SPLIT ({quantum_mode})", [final])
        logger.info(
            "Done — epochs: %d | best_val_loss: %.4f | final_q_grad_var: %.2e",
            result["epochs_trained"], result["best_val_loss"],
            result["final_quantum_grad_var"],
        )


if __name__ == "__main__":
    main()
