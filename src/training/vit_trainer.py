"""Training loop for ExoplanetViT — ViT-based exoplanet transit vetting.

Implements:
- Full training loop with early stopping
- Per-epoch validation with EpochMetrics
- Optional regression loss (MSE on reg_preds)
- Inverse-frequency class weight computation
- Best checkpoint saving
- W&B integration (optional, mockable)
- k-fold cross-validation

DataLoader contract
-------------------
Each batch yielded by train_loader / val_loader must be a 3-tuple:
    (image, aux, label)
where:
    image : torch.Tensor, shape (B, 2, 224, 224), dtype float32
    aux   : torch.Tensor, shape (B, 5),           dtype float32
    label : torch.Tensor, shape (B,),             dtype long

Model contract
--------------
model.forward(image, aux) -> (class_logits, reg_preds)
    class_logits : shape (B, 2)          — raw logits for CrossEntropyLoss
    reg_preds    : shape (B, num_reg) | None — regression outputs
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)

_PLANET_RECALL_THRESHOLD: float = 0.50
_DEFAULT_BATCH_SIZE: int = 4


# ---------------------------------------------------------------------------
# EpochMetrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochMetrics:
    """Immutable container for per-epoch evaluation metrics.

    Attributes
    ----------
    loss          : float  Mean cross-entropy loss over the evaluation set.
    accuracy      : float  Fraction of correctly classified samples, in [0, 1].
    planet_recall : float  Recall for the positive class (label=1), in [0, 1].
    f1            : float  Macro-averaged F1 score across both classes.
    """

    loss: float
    accuracy: float
    planet_recall: float
    f1: float


# ---------------------------------------------------------------------------
# ViTTrainer
# ---------------------------------------------------------------------------

class ViTTrainer:
    """Training loop for ViT-based exoplanet transit classifiers.

    Parameters
    ----------
    model            : nn.Module   Model with forward(image, aux) -> (logits, reg).
    train_loader     : DataLoader  Yields (image, aux, label) 3-tuples for training.
    val_loader       : DataLoader  Yields (image, aux, label) 3-tuples for validation.
    config           : dict        Must include: ``lr``, ``patience``, ``max_epochs``,
                                   ``min_delta``.  Optional: ``regression_loss_weight``.
    checkpoint_dir   : str         Directory for ``best_model.pt``.  Created if absent.
    use_wandb        : bool        Enable W&B logging.
    use_class_weights: bool        Apply inverse-frequency class weights to the loss.
    use_regression   : bool        Include MSE regression loss on reg_preds.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        checkpoint_dir: str = "models/",
        use_wandb: bool = True,
        use_class_weights: bool = False,
        use_regression: bool = False,
    ) -> None:
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._config = config
        self._checkpoint_dir = Path(checkpoint_dir)
        self._use_wandb = use_wandb
        self._use_class_weights: bool = use_class_weights or config.get(
            "use_class_weights", False
        )
        self._use_regression = use_regression
        self._regression_loss_weight: float = config.get("regression_loss_weight", 0.1)

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)

        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=config.get("lr", 1e-3)
        )

        if self._use_class_weights:
            weights = self._compute_class_weights().to(self._device)
            self._criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self._criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, max_epochs: int | None = None) -> dict:
        """Run the full training loop with early stopping.

        Returns
        -------
        dict
            ``{"epochs_trained": int, "best_val_loss": float}``
        """
        n_epochs: int = max_epochs if max_epochs is not None else self._config.get(
            "max_epochs", 10
        )
        patience: int = self._config.get("patience", 3)
        min_delta: float = self._config.get("min_delta", 0.0)

        if self._use_wandb:
            import wandb  # lazy import — optional dependency
            wandb.init(config=self._config)
            wandb.watch(self._model)

        best_f1: float = 0.0
        best_planet_recall: float = 0.0
        epochs_without_improvement: int = 0
        epochs_trained: int = 0

        for epoch in range(1, n_epochs + 1):
            epochs_trained = epoch
            train_loss = self._train_one_epoch()
            val_metrics = self.evaluate(self._val_loader)

            if self._use_wandb:
                import wandb  # noqa: PLC0415
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_metrics.loss,
                        "val_accuracy": val_metrics.accuracy,
                        "val_planet_recall": val_metrics.planet_recall,
                        "val_f1": val_metrics.f1,
                    }
                )

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_acc=%.4f | val_recall=%.4f | val_f1=%.4f",
                epoch,
                n_epochs,
                train_loss,
                val_metrics.loss,
                val_metrics.accuracy,
                val_metrics.planet_recall,
                val_metrics.f1,
            )

            if val_metrics.planet_recall < _PLANET_RECALL_THRESHOLD:
                logger.warning(
                    "Epoch %d: planet_recall=%.4f below threshold %.2f.",
                    epoch,
                    val_metrics.planet_recall,
                    _PLANET_RECALL_THRESHOLD,
                )

            improved = val_metrics.f1 > best_f1 + min_delta
            if improved:
                best_f1 = val_metrics.f1
                best_planet_recall = val_metrics.planet_recall
                epochs_without_improvement = 0
                self._save_checkpoint(epoch=epoch, val_loss=val_metrics.loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d).",
                        epoch,
                        patience,
                    )
                    break

        if self._use_wandb:
            import wandb  # noqa: PLC0415
            wandb.finish()

        return {"epochs_trained": epochs_trained, "best_f1": best_f1, "best_planet_recall": best_planet_recall}

    def evaluate(self, loader: DataLoader) -> EpochMetrics:
        """Run inference on *loader* and compute evaluation metrics.

        Temporarily switches the model to eval mode and restores training mode
        before returning.

        Parameters
        ----------
        loader : DataLoader  Yields (image, aux, label) 3-tuples.

        Returns
        -------
        EpochMetrics  Contains loss, accuracy, and planet_recall.
        """
        self._model.eval()

        total_loss = 0.0
        n_samples = 0
        n_correct = 0
        true_positives = 0
        false_negatives = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for image, aux, labels in loader:
                image = image.to(self._device)
                aux = aux.to(self._device)
                labels = labels.to(self._device).long()

                class_logits, _ = self._model(image, aux)  # (B, 2)

                loss = self._criterion(class_logits, labels)
                total_loss += loss.item() * labels.size(0)
                n_samples += labels.size(0)

                preds = class_logits.argmax(dim=1)  # (B,)
                n_correct += (preds == labels).sum().item()

                positive_mask = labels == 1
                true_positives += (preds[positive_mask] == 1).sum().item()
                false_negatives += (preds[positive_mask] == 0).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        import numpy as np
        from sklearn.metrics import f1_score

        mean_loss = total_loss / max(n_samples, 1)
        accuracy = n_correct / max(n_samples, 1)
        denom = true_positives + false_negatives
        planet_recall = true_positives / denom if denom > 0 else 0.0
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0.0)

        self._model.train()

        return EpochMetrics(
            loss=float(mean_loss),
            accuracy=float(accuracy),
            planet_recall=float(planet_recall),
            f1=float(f1),
        )

    def kfold_cv(
        self,
        dataset: Dataset,
        k: int = 5,
        max_epochs: int | None = None,
    ) -> list[EpochMetrics]:
        """Run k-fold cross-validation on *dataset*.

        Each fold trains an independently initialised copy of the model from
        scratch, evaluates on the held-out fold, and returns the val metrics.

        Parameters
        ----------
        dataset    : Dataset  Full dataset to split into k folds.
        k          : int      Number of folds.
        max_epochs : int | None  Per-fold training epochs (defaults to config value).

        Returns
        -------
        list[EpochMetrics]  One entry per fold, in fold order.
        """
        n = len(dataset)  # type: ignore[arg-type]
        fold_size = n // k
        indices = list(range(n))
        results: list[EpochMetrics] = []

        for fold in range(k):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else n

            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]

            train_ds = Subset(dataset, train_indices)
            val_ds = Subset(dataset, val_indices)

            fold_train_loader = DataLoader(
                train_ds, batch_size=_DEFAULT_BATCH_SIZE, shuffle=True
            )
            fold_val_loader = DataLoader(
                val_ds, batch_size=_DEFAULT_BATCH_SIZE, shuffle=False
            )

            # Fresh model copy for each fold
            fold_model = deepcopy(self._model)

            fold_trainer = ViTTrainer(
                model=fold_model,
                train_loader=fold_train_loader,
                val_loader=fold_val_loader,
                config=self._config,
                checkpoint_dir=str(self._checkpoint_dir / f"fold_{fold}"),
                use_wandb=False,
                use_class_weights=self._use_class_weights,
                use_regression=self._use_regression,
            )
            fold_trainer.train(max_epochs=max_epochs)
            fold_metrics = fold_trainer.evaluate(fold_val_loader)
            results.append(fold_metrics)

        return results

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency weights from train_loader labels.

        Returns
        -------
        torch.Tensor  Shape (2,), dtype float32.
        """
        counts = torch.zeros(2, dtype=torch.float32)

        for _, _, labels in self._train_loader:
            for c in range(2):
                counts[c] += (labels == c).sum().item()

        total = counts.sum()
        safe_counts = counts.clamp(min=1.0)
        return total / (2.0 * safe_counts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> float:
        """Run one full pass over self._train_loader.

        Returns
        -------
        float  Mean training loss for the epoch.
        """
        self._model.train()
        total_loss = 0.0
        n_samples = 0

        for image, aux, labels in self._train_loader:
            image = image.to(self._device)
            aux = aux.to(self._device)
            labels = labels.to(self._device).long()

            self._optimizer.zero_grad()
            class_logits, reg_preds = self._model(image, aux)  # (B, 2), (B, k) | None

            loss = self._criterion(class_logits, labels)

            if self._use_regression and reg_preds is not None:
                num_reg = reg_preds.shape[1]
                proxy = labels.float().unsqueeze(-1).expand(-1, num_reg)
                reg_loss = nn.functional.mse_loss(reg_preds, proxy)
                loss = loss + self._regression_loss_weight * reg_loss

            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

        return total_loss / max(n_samples, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model state dict to checkpoint_dir/best_model.pt."""
        checkpoint_path = self._checkpoint_dir / "best_model.pt"
        torch.save(self._model.state_dict(), str(checkpoint_path))
        logger.info(
            "Checkpoint saved to %s (epoch=%d, val_loss=%.4f)",
            checkpoint_path,
            epoch,
            val_loss,
        )
