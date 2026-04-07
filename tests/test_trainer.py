"""Tests for src/training/trainer.py — Trainer class.

TDD: All tests written BEFORE implementation.

Test model contract:
    - DataLoader yields batches of (global_view, label)
      where global_view has shape (B, 1, 2001) and label has shape (B,).
    - Trainer accepts any nn.Module that takes a (B, 1, 2001) tensor
      and returns logits of shape (B, 1) or (B,).

Mock strategy:
    - wandb.init, wandb.log, wandb.finish are patched to prevent
      network calls during testing.
    - A tiny 2-layer MLP is used as the model for speed.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model() -> nn.Module:
    """Return a minimal 2-layer MLP: (B, 1, 2001) -> (B, 2).

    Output shape (B, 2) matches BaselineCNN and CrossEntropyLoss contract.
    """
    return nn.Sequential(
        nn.Flatten(),           # (B, 2001)
        nn.Linear(2001, 16),
        nn.ReLU(),
        nn.Linear(16, 2),       # logits, shape (B, 2) for CrossEntropyLoss
    )


def _make_loader(
    n_batches: int = 2,
    batch_size: int = 2,
    label_values: list[int] | None = None,
) -> DataLoader:
    """Build a DataLoader with n_batches * batch_size synthetic samples.

    Each sample: global_view shape (1, 2001), label scalar 0 or 1.
    If label_values is given its length must equal n_batches * batch_size.
    """
    n_samples = n_batches * batch_size
    global_views = torch.randn(n_samples, 1, 2001)

    if label_values is not None:
        assert len(label_values) == n_samples
        labels = torch.tensor(label_values, dtype=torch.long)
    else:
        # Alternate 0 / 1 for balanced classes
        labels = torch.tensor(
            [i % 2 for i in range(n_samples)], dtype=torch.long
        )

    ds = TensorDataset(global_views, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _default_config() -> dict:
    return {
        "lr": 1e-3,
        "patience": 3,
        "max_epochs": 10,
        "min_delta": 0.0,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_model() -> nn.Module:
    return _make_tiny_model()


@pytest.fixture()
def train_loader() -> DataLoader:
    return _make_loader(n_batches=2, batch_size=2)


@pytest.fixture()
def val_loader() -> DataLoader:
    return _make_loader(n_batches=2, batch_size=2)


@pytest.fixture()
def config() -> dict:
    return _default_config()


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _patch_wandb():
    """Return patches that mock wandb for lazy-import usage in Trainer.

    Because wandb is lazily imported inside train(), we patch the wandb
    module itself in sys.modules so any `import wandb` inside the function
    receives a MagicMock instead of the real module.
    """
    import sys
    import unittest.mock as mock
    mock_wandb = mock.MagicMock()
    return [mock.patch.dict("sys.modules", {"wandb": mock_wandb})]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainerOneEpoch:
    """test_trainer_runs_one_epoch: training completes without raising."""

    def test_trainer_runs_one_epoch(
        self,
        tiny_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        tmp_path: Path,
    ) -> None:
        """Trainer.train(max_epochs=1) must complete without error."""
        from src.training.trainer import Trainer

        mocks = _patch_wandb()
        started = [m.start() for m in mocks]
        try:
            trainer = Trainer(
                model=tiny_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                checkpoint_dir=str(tmp_path),
                use_wandb=True,
            )
            result = trainer.train(max_epochs=1)
        finally:
            for m in mocks:
                m.stop()

        assert isinstance(result, dict), (
            f"train() must return a dict, got {type(result)}"
        )


class TestTrainerValMetrics:
    """test_trainer_computes_val_metrics: evaluate() returns EpochMetrics."""

    def test_trainer_computes_val_metrics(
        self,
        tiny_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        tmp_path: Path,
    ) -> None:
        """After 1 epoch, evaluate(val_loader) must return a EpochMetrics."""
        from src.training.trainer import EpochMetrics, Trainer

        mocks = _patch_wandb()
        started = [m.start() for m in mocks]
        try:
            trainer = Trainer(
                model=tiny_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                checkpoint_dir=str(tmp_path),
                use_wandb=False,
            )
            trainer.train(max_epochs=1)
            metrics = trainer.evaluate(val_loader)
        finally:
            for m in mocks:
                m.stop()

        assert isinstance(metrics, EpochMetrics), (
            f"evaluate() must return EpochMetrics, got {type(metrics)}"
        )
        # EpochMetrics must expose loss, accuracy, planet_recall
        assert hasattr(metrics, "loss"), "EpochMetrics must have .loss"
        assert hasattr(metrics, "accuracy"), "EpochMetrics must have .accuracy"
        assert hasattr(metrics, "planet_recall"), (
            "EpochMetrics must have .planet_recall"
        )
        assert isinstance(metrics.loss, float), ".loss must be float"
        assert 0.0 <= metrics.accuracy <= 1.0, (
            f".accuracy must be in [0, 1], got {metrics.accuracy}"
        )
        assert 0.0 <= metrics.planet_recall <= 1.0, (
            f".planet_recall must be in [0, 1], got {metrics.planet_recall}"
        )


class TestEarlyStopping:
    """test_early_stopping_triggers: training halts when val loss stops improving."""

    def test_early_stopping_triggers(
        self,
        tiny_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """With patience=2, training must stop at epoch 3 (not epoch 10).

        Strategy: override Trainer.evaluate() to return strictly increasing
        val losses so early stopping fires after patience epochs.
        """
        from src.training.trainer import EpochMetrics, Trainer

        epoch_counter: list[int] = []

        # Increasing val losses: 1.0, 2.0, 3.0 … so no improvement ever
        increasing_losses = iter([float(i + 1) for i in range(20)])

        def _fake_evaluate(loader: DataLoader) -> EpochMetrics:
            return EpochMetrics(
                loss=next(increasing_losses),
                accuracy=0.5,
                planet_recall=0.9,
            )

        config = {
            "lr": 1e-3,
            "patience": 2,
            "max_epochs": 10,
            "min_delta": 0.0,
        }

        mocks = _patch_wandb()
        started = [m.start() for m in mocks]
        try:
            trainer = Trainer(
                model=tiny_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                checkpoint_dir=str(tmp_path),
                use_wandb=False,
            )
            # Monkey-patch evaluate so val loss always increases
            trainer.evaluate = _fake_evaluate  # type: ignore[method-assign]
            result = trainer.train(max_epochs=10)
        finally:
            for m in mocks:
                m.stop()

        # patience=2 means stop after 2 non-improving epochs → epoch 3 total
        epochs_trained = result.get("epochs_trained")
        assert epochs_trained == 3, (
            f"Expected early stopping at epoch 3 with patience=2, "
            f"got epochs_trained={epochs_trained}"
        )


class TestBestCheckpointSaved:
    """test_best_checkpoint_saved: a checkpoint file exists after training."""

    def test_best_checkpoint_saved(
        self,
        tiny_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        tmp_path: Path,
    ) -> None:
        """After training completes, best_model.pt must exist in checkpoint_dir."""
        from src.training.trainer import Trainer

        mocks = _patch_wandb()
        started = [m.start() for m in mocks]
        try:
            trainer = Trainer(
                model=tiny_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                checkpoint_dir=str(tmp_path),
                use_wandb=False,
            )
            trainer.train(max_epochs=2)
        finally:
            for m in mocks:
                m.stop()

        checkpoint_path = tmp_path / "best_model.pt"
        assert checkpoint_path.exists(), (
            f"Expected checkpoint at {checkpoint_path}, but file not found"
        )
        # Verify the file is a valid torch state dict
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        assert isinstance(state, dict), "Checkpoint must be a state dict (dict)"


class TestClassWeightsComputed:
    """test_class_weights_computed: inverse frequency weights used when enabled."""

    def test_class_weights_computed(
        self,
        tiny_model: nn.Module,
        tmp_path: Path,
    ) -> None:
        """When use_class_weights=True the loss uses inverse frequency weights.

        Strategy:
        - Create an imbalanced loader: 3 positives + 1 negative.
        - Instantiate Trainer with use_class_weights=True.
        - Call _compute_class_weights() directly and check that the
          weight for the minority class (label=0) is larger.
        """
        from src.training.trainer import Trainer

        # 3 positives (label=1), 1 negative (label=0)
        global_views = torch.randn(4, 1, 2001)
        labels = torch.tensor([1, 1, 1, 0], dtype=torch.long)
        from torch.utils.data import TensorDataset
        ds = TensorDataset(global_views, labels)
        imbalanced_loader = DataLoader(ds, batch_size=4, shuffle=False)

        config = {**_default_config(), "use_class_weights": True}

        mocks = _patch_wandb()
        started = [m.start() for m in mocks]
        try:
            trainer = Trainer(
                model=tiny_model,
                train_loader=imbalanced_loader,
                val_loader=imbalanced_loader,
                config=config,
                checkpoint_dir=str(tmp_path),
                use_wandb=False,
                use_class_weights=True,
            )
            weights = trainer._compute_class_weights()
        finally:
            for m in mocks:
                m.stop()

        assert isinstance(weights, torch.Tensor), (
            f"_compute_class_weights() must return torch.Tensor, got {type(weights)}"
        )
        assert weights.shape == torch.Size([2]), (
            f"Expected weights shape [2], got {weights.shape}"
        )
        # Weight for minority class (label=0, 1 sample) > majority class (label=1, 3 samples)
        assert weights[0].item() > weights[1].item(), (
            f"Minority class weight ({weights[0].item():.3f}) must exceed "
            f"majority class weight ({weights[1].item():.3f})"
        )


class TestStopConditionLogged:
    """test_stop_condition_logged: low planet_recall emits WARNING, not raises."""

    def test_stop_condition_logged(
        self,
        tiny_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If planet_recall < 0.85, a WARNING must be logged (training still completes).

        Strategy: patch evaluate() to return planet_recall=0.5 so the
        threshold is always breached, then check caplog for a WARNING.
        """
        from src.training.trainer import EpochMetrics, Trainer

        # Decreasing loss so early stopping does NOT fire in 2 epochs
        losses = iter([1.0, 0.9, 0.8])

        def _fake_evaluate(loader: DataLoader) -> EpochMetrics:
            return EpochMetrics(
                loss=next(losses),
                accuracy=0.5,
                planet_recall=0.5,  # below 0.85 threshold
            )

        config = {
            "lr": 1e-3,
            "patience": 5,
            "max_epochs": 2,
            "min_delta": 0.0,
        }

        mocks = _patch_wandb()
        started = [m.start() for m in mocks]
        try:
            trainer = Trainer(
                model=tiny_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                checkpoint_dir=str(tmp_path),
                use_wandb=False,
            )
            trainer.evaluate = _fake_evaluate  # type: ignore[method-assign]

            with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
                result = trainer.train(max_epochs=2)
        finally:
            for m in mocks:
                m.stop()

        # Training must have completed without raising
        assert isinstance(result, dict), (
            "train() must return dict even when planet_recall is low"
        )
        # At least one WARNING must mention planet_recall / recall
        warning_msgs = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any(
            "recall" in str(m).lower() or "planet" in str(m).lower()
            for m in warning_msgs
        ), (
            f"Expected a WARNING about planet_recall, "
            f"got warning messages: {warning_msgs}"
        )
