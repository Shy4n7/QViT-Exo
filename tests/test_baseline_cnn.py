"""Tests for the Shallue & Vanderburg 2018 baseline 1D CNN.

TDD: All tests written before implementation.
Architecture reference: https://arxiv.org/abs/1712.05044
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def global_batch() -> torch.Tensor:
    """Batch of 4 global-view light curves, shape (4, 1, 2001)."""
    torch.manual_seed(42)
    return torch.randn(4, 1, 2001)


@pytest.fixture
def local_batch() -> torch.Tensor:
    """Batch of 4 local-view light curves, shape (4, 1, 201)."""
    torch.manual_seed(43)
    return torch.randn(4, 1, 201)


@pytest.fixture
def model_global_only():
    """BaselineCNN with use_local_view=False (default)."""
    from src.models.baseline_cnn import BaselineCNN
    return BaselineCNN(use_local_view=False, dropout=0.5)


@pytest.fixture
def model_with_local():
    """BaselineCNN with use_local_view=True."""
    from src.models.baseline_cnn import BaselineCNN
    return BaselineCNN(use_local_view=True, dropout=0.5)


# ---------------------------------------------------------------------------
# Shape / forward-pass tests
# ---------------------------------------------------------------------------

class TestForwardShape:
    def test_global_view_forward_shape(self, model_global_only, global_batch):
        """Global-only forward: (4, 1, 2001) -> (4, 2) logits."""
        model_global_only.eval()
        with torch.no_grad():
            output = model_global_only(global_batch)
        assert output.shape == (4, 2), (
            f"Expected output shape (4, 2), got {output.shape}"
        )

    def test_local_view_disabled_forward(self, model_global_only, global_batch):
        """Model with use_local_view=False accepts only global input and succeeds."""
        model_global_only.eval()
        with torch.no_grad():
            output = model_global_only(global_batch)
        # Should not raise; output must be 2D
        assert output.ndim == 2

    def test_local_view_enabled_forward(self, model_with_local, global_batch, local_batch):
        """Model with use_local_view=True: (global [4,1,2001], local [4,1,201]) -> (4, 2)."""
        model_with_local.eval()
        with torch.no_grad():
            output = model_with_local(global_batch, local_batch)
        assert output.shape == (4, 2), (
            f"Expected output shape (4, 2), got {output.shape}"
        )


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------

class TestNumerics:
    def test_output_is_finite(self, model_global_only, global_batch):
        """No NaN or Inf in forward pass output."""
        model_global_only.eval()
        with torch.no_grad():
            output = model_global_only(global_batch)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_output_is_finite_with_local_view(self, model_with_local, global_batch, local_batch):
        """No NaN or Inf with both views active."""
        model_with_local.eval()
        with torch.no_grad():
            output = model_with_local(global_batch, local_batch)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf (local view)"

    def test_different_inputs_different_outputs(self, model_global_only):
        """Two clearly distinct inputs must produce different logit vectors."""
        model_global_only.eval()
        input_a = torch.zeros(2, 1, 2001)
        input_b = torch.ones(2, 1, 2001)
        with torch.no_grad():
            out_a = model_global_only(input_a)
            out_b = model_global_only(input_b)
        assert not torch.allclose(out_a, out_b), (
            "Model produced identical outputs for zero-input and one-input batches"
        )


# ---------------------------------------------------------------------------
# Training dynamics
# ---------------------------------------------------------------------------

class TestTrainingDynamics:
    def test_gradient_flows(self, model_global_only, global_batch):
        """loss.backward() produces non-zero gradients on all named parameters."""
        model_global_only.train()
        output = model_global_only(global_batch)
        labels = torch.randint(0, 2, (4,))
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()

        zero_grad_params = [
            name
            for name, param in model_global_only.named_parameters()
            if param.grad is None or param.grad.abs().sum().item() == 0.0
        ]
        assert len(zero_grad_params) == 0, (
            f"Zero or missing gradients on: {zero_grad_params}"
        )

    def test_gradient_flows_with_local_view(self, model_with_local, global_batch, local_batch):
        """Gradients flow through both branches when use_local_view=True."""
        model_with_local.train()
        output = model_with_local(global_batch, local_batch)
        labels = torch.randint(0, 2, (4,))
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()

        zero_grad_params = [
            name
            for name, param in model_with_local.named_parameters()
            if param.grad is None or param.grad.abs().sum().item() == 0.0
        ]
        assert len(zero_grad_params) == 0, (
            f"Zero or missing gradients on: {zero_grad_params}"
        )


# ---------------------------------------------------------------------------
# Model properties
# ---------------------------------------------------------------------------

class TestModelProperties:
    def test_parameter_count_reasonable(self, model_global_only):
        """Total param count must be between 10_000 and 500_000."""
        total = sum(p.numel() for p in model_global_only.parameters())
        assert 10_000 <= total <= 500_000, (
            f"Parameter count {total} outside expected range [10_000, 500_000]"
        )

    def test_parameter_count_reasonable_with_local_view(self, model_with_local):
        """Local-view model param count between 10_000 and 1_000_000."""
        total = sum(p.numel() for p in model_with_local.parameters())
        assert 10_000 <= total <= 1_000_000, (
            f"Parameter count {total} outside expected range [10_000, 1_000_000]"
        )

    def test_dropout_inactive_eval_mode(self, model_global_only, global_batch):
        """In eval mode, two forward passes on the same input produce identical output."""
        model_global_only.eval()
        with torch.no_grad():
            out1 = model_global_only(global_batch)
            out2 = model_global_only(global_batch)
        assert torch.allclose(out1, out2), (
            "Eval-mode forward passes are not deterministic — dropout may be active"
        )

    def test_uses_cross_entropy_compatible_output(self, model_global_only, global_batch):
        """Output logits must be compatible with nn.CrossEntropyLoss (no softmax applied)."""
        model_global_only.eval()
        with torch.no_grad():
            output = model_global_only(global_batch)
        # If softmax were applied, all values would be in (0, 1) and rows sum to 1.
        # Raw logits can have values outside (0, 1) or rows that don't sum to 1.
        row_sums = output.sum(dim=1)
        # At least some rows should NOT sum to ~1.0 (would fail for softmax output)
        not_all_one = not torch.allclose(row_sums, torch.ones(4), atol=1e-3)
        # Allow: if the model happens to output values near 1.0, check range instead
        has_out_of_range = (output < 0).any() or (output > 1).any()
        assert not_all_one or has_out_of_range, (
            "Output looks like softmax probabilities — model should return raw logits"
        )

    def test_model_has_correct_output_classes(self, model_global_only, global_batch):
        """Output second dimension must be exactly 2 (binary classification)."""
        model_global_only.eval()
        with torch.no_grad():
            output = model_global_only(global_batch)
        assert output.shape[1] == 2, (
            f"Expected 2 output classes, got {output.shape[1]}"
        )
