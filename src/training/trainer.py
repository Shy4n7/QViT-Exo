"""Training loop for exoplanet transit vetting models.

Implements:
- Full training loop with early stopping
- Per-epoch validation with EpochMetrics
- Inverse-frequency class weight computation
- Best checkpoint saving
- W&B integration (optional, mockable)
- WARNING log when planet_recall < 0.85

DataLoader contract
-------------------
Each batch yielded by train_loader / val_loader must be a 2-tuple:
    (global_view, label)
where:
    global_view : torch.Tensor, shape (B, 1, 2001), dtype float32
    label       : torch.Tensor, shape (B,),         dtype long

Model contract
--------------
model(global_view) → logits, shape (B, 1) or (B,).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# wandb is imported lazily inside train() to avoid hard-import failures
# on machines without wandb installed when use_wandb=False.

logger = logging.getLogger(__name__)

# Threshold below which planet recall triggers a WARNING log.
_PLANET_RECALL_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# EpochMetrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochMetrics:
    """Immutable container for per-epoch evaluation metrics.

    Distinct from ``metrics.EpochMetrics`` which holds full sklearn metrics.
    This is a lightweight container used internally by the Trainer.

    Attributes
    ----------
    loss : float
        Mean cross-entropy loss over the evaluation set.
    accuracy : float
        Fraction of correctly classified samples, in [0, 1].
    planet_recall : float
        Recall for the positive class (CONFIRMED = 1), in [0, 1].
    """

    loss: float
    accuracy: float
    planet_recall: float


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Full training loop with early stopping and optional W&B logging.

    Parameters
    ----------
    model : nn.Module
        Model accepting (B, 1, 2001) tensors and returning (B, 1) or (B,) logits.
    train_loader : DataLoader
        Yields (global_view, label) batches for training.
    val_loader : DataLoader
        Yields (global_view, label) batches for validation.
    config : dict
        Must include keys: ``lr``, ``patience``, ``max_epochs``, ``min_delta``.
        Optional key: ``use_class_weights`` (bool, default False).
    checkpoint_dir : str
        Directory to write ``best_model.pt``.  Created if absent.
    use_wandb : bool
        When True, call wandb.init / wandb.log / wandb.finish.
    use_class_weights : bool
        When True, compute and apply inverse frequency weights to the loss.
        Overrides the ``use_class_weights`` key in *config* if provided.
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
    ) -> None:
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._config = config
        self._checkpoint_dir = Path(checkpoint_dir)
        self._use_wandb = use_wandb
        # use_class_weights: explicit kwarg takes precedence over config key
        self._use_class_weights: bool = use_class_weights or config.get(
            "use_class_weights", False
        )

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)

        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=config.get("lr", 1e-3)
        )

        # Build loss — CrossEntropyLoss matches the model's (B, 2) output.
        # Optionally weight by inverse class frequency to handle imbalance.
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

        Parameters
        ----------
        max_epochs : int | None
            Override ``config["max_epochs"]`` when provided.

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

        best_val_loss: float = float("inf")
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
                    }
                )

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_acc=%.4f | val_recall=%.4f",
                epoch,
                n_epochs,
                train_loss,
                val_metrics.loss,
                val_metrics.accuracy,
                val_metrics.planet_recall,
            )

            # Warn when planet recall falls below threshold — do NOT raise
            if val_metrics.planet_recall < _PLANET_RECALL_THRESHOLD:
                logger.warning(
                    "Epoch %d: planet_recall=%.4f is below threshold %.2f. "
                    "Consider adjusting class weights or decision threshold.",
                    epoch,
                    val_metrics.planet_recall,
                    _PLANET_RECALL_THRESHOLD,
                )

            # Save checkpoint if this is the best model so far
            improved = val_metrics.loss < best_val_loss - min_delta
            if improved:
                best_val_loss = val_metrics.loss
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

        return {"epochs_trained": epochs_trained, "best_val_loss": best_val_loss}

    def evaluate(self, loader: DataLoader) -> EpochMetrics:
        """Run inference on *loader* and compute evaluation metrics.

        Parameters
        ----------
        loader : DataLoader
            Yields (global_view, label) batches.

        Returns
        -------
        EpochMetrics
            Contains loss, accuracy, and planet_recall.
        """
        self._model.eval()

        total_loss = 0.0
        n_samples = 0
        n_correct = 0
        true_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for global_view, labels in loader:
                global_view = global_view.to(self._device)
                labels = labels.to(self._device).long()  # CrossEntropyLoss requires long

                logits = self._model(global_view)  # (B, 2) — no squeeze needed

                loss = self._criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                n_samples += labels.size(0)

                preds = logits.argmax(dim=1)  # (B,) predicted class indices
                n_correct += (preds == labels).sum().item()

                # Planet recall: TP / (TP + FN) for positive class (label=1)
                positive_mask = labels == 1
                true_positives += (preds[positive_mask] == 1).sum().item()
                false_negatives += (preds[positive_mask] == 0).sum().item()

        mean_loss = total_loss / max(n_samples, 1)
        accuracy = n_correct / max(n_samples, 1)
        denom = true_positives + false_negatives
        planet_recall = true_positives / denom if denom > 0 else 0.0

        self._model.train()

        return EpochMetrics(
            loss=float(mean_loss),
            accuracy=float(accuracy),
            planet_recall=float(planet_recall),
        )

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency weights from train_loader labels.

        Iterates over all batches in self._train_loader to count label
        occurrences, then returns weight[c] = total / (n_classes * count[c]).

        Returns
        -------
        torch.Tensor
            Shape (2,), dtype float32.  weights[0] for class 0, weights[1] for class 1.
        """
        counts = torch.zeros(2, dtype=torch.float32)

        for _, labels in self._train_loader:
            for c in range(2):
                counts[c] += (labels == c).sum().item()

        total = counts.sum()
        # Avoid division by zero for absent classes
        safe_counts = counts.clamp(min=1.0)
        weights = total / (2.0 * safe_counts)
        return weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> float:
        """Run one full pass over self._train_loader.

        Returns
        -------
        float
            Mean training loss for the epoch.
        """
        self._model.train()
        total_loss = 0.0
        n_samples = 0

        for global_view, labels in self._train_loader:
            global_view = global_view.to(self._device)
            labels = labels.to(self._device).long()  # CrossEntropyLoss requires long

            self._optimizer.zero_grad()
            logits = self._model(global_view)  # (B, 2) — no squeeze
            loss = self._criterion(logits, labels)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

        return total_loss / max(n_samples, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model state dict to checkpoint_dir/best_model.pt.

        Parameters
        ----------
        epoch : int
            Current epoch number (stored alongside the state dict).
        val_loss : float
            Validation loss at time of saving.
        """
        checkpoint_path = self._checkpoint_dir / "best_model.pt"
        torch.save(self._model.state_dict(), str(checkpoint_path))
        logger.info(
            "Checkpoint saved to %s (epoch=%d, val_loss=%.4f)",
            checkpoint_path,
            epoch,
            val_loss,
        )
