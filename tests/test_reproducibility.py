"""Tests for src/utils/reproducibility.py — written BEFORE implementation (TDD RED phase)."""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# NOTE: These imports will FAIL until src/utils/reproducibility.py exists.
# That failure confirms we are in the RED state.
# ---------------------------------------------------------------------------
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestSetSeedNumpy:
    """set_seed produces deterministic NumPy output."""

    def test_same_seed_produces_same_sequence(self):
        set_seed(42)
        a = np.random.rand(10)

        set_seed(42)
        b = np.random.rand(10)

        np.testing.assert_array_equal(a, b, err_msg="set_seed(42) must yield identical numpy sequences")

    def test_seed_affects_subsequent_calls(self):
        """Multiple np.random calls after a single set_seed are all deterministic."""
        set_seed(42)
        first_call = np.random.rand(5)
        second_call = np.random.rand(5)

        set_seed(42)
        first_call_2 = np.random.rand(5)
        second_call_2 = np.random.rand(5)

        np.testing.assert_array_equal(first_call, first_call_2)
        np.testing.assert_array_equal(second_call, second_call_2)

    def test_numpy_integers_deterministic(self):
        set_seed(7)
        a = np.random.randint(0, 100, size=20)

        set_seed(7)
        b = np.random.randint(0, 100, size=20)

        np.testing.assert_array_equal(a, b)


class TestSetSeedTorch:
    """set_seed produces deterministic PyTorch output."""

    def test_same_seed_produces_same_tensor(self):
        set_seed(42)
        a = torch.rand(10)

        set_seed(42)
        b = torch.rand(10)

        assert torch.equal(a, b), "set_seed(42) must yield identical torch tensors"

    def test_seed_affects_subsequent_torch_calls(self):
        set_seed(42)
        first = torch.rand(5)
        second = torch.rand(5)

        set_seed(42)
        first_2 = torch.rand(5)
        second_2 = torch.rand(5)

        assert torch.equal(first, first_2)
        assert torch.equal(second, second_2)

    def test_torch_randn_deterministic(self):
        set_seed(99)
        a = torch.randn(8)

        set_seed(99)
        b = torch.randn(8)

        assert torch.equal(a, b)

    def test_torch_randint_deterministic(self):
        set_seed(13)
        a = torch.randint(0, 100, (15,))

        set_seed(13)
        b = torch.randint(0, 100, (15,))

        assert torch.equal(a, b)


class TestDifferentSeedsDiffer:
    """Different seeds must produce different random sequences."""

    def test_numpy_different_seeds_differ(self):
        set_seed(42)
        a = np.random.rand(50)

        set_seed(99)
        b = np.random.rand(50)

        assert not np.array_equal(a, b), (
            "seed=42 and seed=99 must not produce identical numpy arrays"
        )

    def test_torch_different_seeds_differ(self):
        set_seed(42)
        a = torch.rand(50)

        set_seed(99)
        b = torch.rand(50)

        assert not torch.equal(a, b), (
            "seed=42 and seed=99 must not produce identical torch tensors"
        )

    def test_zero_seed_differs_from_nonzero(self):
        set_seed(0)
        a = np.random.rand(20)

        set_seed(1)
        b = np.random.rand(20)

        assert not np.array_equal(a, b)


class TestSetSeedReturnValue:
    """set_seed is a pure procedure — it returns None."""

    def test_returns_none(self):
        result = set_seed(42)
        assert result is None, "set_seed must return None (side-effect only)"


class TestSetSeedEdgeCases:
    """Boundary and invalid input handling."""

    def test_seed_zero_is_valid(self):
        """seed=0 is a legitimate value and must not raise."""
        set_seed(0)
        val = np.random.rand()
        assert 0.0 <= val <= 1.0

    def test_large_seed_is_valid(self):
        """Seeds up to 2**32 - 1 must be accepted."""
        set_seed(2**32 - 1)
        val = np.random.rand()
        assert 0.0 <= val <= 1.0

    def test_negative_seed_raises(self):
        """Negative seeds are outside the valid range and must raise ValueError."""
        with pytest.raises((ValueError, OverflowError)):
            set_seed(-1)

    def test_non_integer_seed_raises(self):
        """Float seeds must raise TypeError."""
        with pytest.raises(TypeError):
            set_seed(3.14)  # type: ignore[arg-type]

    def test_none_seed_raises(self):
        """None seed must raise TypeError."""
        with pytest.raises(TypeError):
            set_seed(None)  # type: ignore[arg-type]


class TestCudaDeterminism:
    """Verify CUDA determinism flags are set (CPU-only environments tolerated)."""

    def test_cudnn_deterministic_flag_set(self):
        set_seed(42)
        assert torch.backends.cudnn.deterministic is True

    def test_cudnn_benchmark_flag_unset(self):
        set_seed(42)
        assert torch.backends.cudnn.benchmark is False
