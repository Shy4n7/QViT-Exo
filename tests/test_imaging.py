"""Tests for src/data/imaging.py — RED phase written before implementation."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flux(n: int, seed: int = 42) -> np.ndarray:
    """Return a 1D float64 flux array of length n with injected transit."""
    rng = np.random.default_rng(seed=seed)
    flux = rng.normal(loc=1.0, scale=0.001, size=n)
    mid = n // 2
    half = max(1, n // 10)
    flux[mid - half : mid + half] -= 0.01
    return flux


# ---------------------------------------------------------------------------
# compute_recurrence_plot
# ---------------------------------------------------------------------------

class TestComputeRecurrencePlot:
    def test_recurrence_plot_shape(self):
        """200-point flux → output shape (64, 64), dtype float32."""
        from src.data.imaging import compute_recurrence_plot

        flux = _make_flux(200)
        result = compute_recurrence_plot(flux)

        assert result.shape == (64, 64), f"Expected (64, 64), got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_recurrence_plot_values_range(self):
        """All values in the recurrence plot must lie in [0.0, 1.0]."""
        from src.data.imaging import compute_recurrence_plot

        flux = _make_flux(200)
        result = compute_recurrence_plot(flux)

        assert float(result.min()) >= 0.0, f"Min value {result.min()} < 0.0"
        assert float(result.max()) <= 1.0, f"Max value {result.max()} > 1.0"

    def test_recurrence_plot_diagonal_is_zero(self):
        """Main diagonal must be all 0.0 (self-recurrence = zero distance)."""
        from src.data.imaging import compute_recurrence_plot

        flux = _make_flux(200)
        result = compute_recurrence_plot(flux)

        diagonal = np.diag(result)
        assert np.all(diagonal == 0.0), (
            f"Diagonal contains non-zero values: {diagonal[diagonal != 0.0]}"
        )


# ---------------------------------------------------------------------------
# compute_gadf
# ---------------------------------------------------------------------------

class TestComputeGadf:
    def test_gadf_shape(self):
        """200-point flux → output shape (64, 64), dtype float32."""
        from src.data.imaging import compute_gadf

        flux = _make_flux(200)
        result = compute_gadf(flux)

        assert result.shape == (64, 64), f"Expected (64, 64), got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_gadf_values_range(self):
        """All values in the GADF must lie in [-1.0, 1.0]."""
        from src.data.imaging import compute_gadf

        flux = _make_flux(200)
        result = compute_gadf(flux)

        assert float(result.min()) >= -1.0, f"Min value {result.min()} < -1.0"
        assert float(result.max()) <= 1.0, f"Max value {result.max()} > 1.0"


# ---------------------------------------------------------------------------
# generate_image_pair
# ---------------------------------------------------------------------------

class TestGenerateImagePair:
    def test_generate_image_pair_shape(self):
        """Output shape must be (2, 64, 64), dtype float32."""
        from src.data.imaging import generate_image_pair

        flux = _make_flux(200)
        result = generate_image_pair(flux)

        assert result.shape == (2, 64, 64), f"Expected (2, 64, 64), got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_generate_image_pair_channel_0_is_rp(self):
        """Channel 0 of image pair must match standalone compute_recurrence_plot output."""
        from src.data.imaging import compute_recurrence_plot, generate_image_pair

        flux = _make_flux(200)
        pair = generate_image_pair(flux)
        rp = compute_recurrence_plot(flux)

        np.testing.assert_array_equal(
            pair[0], rp, err_msg="Channel 0 does not match compute_recurrence_plot"
        )

    def test_generate_image_pair_channel_1_is_gadf(self):
        """Channel 1 of image pair must match standalone compute_gadf output."""
        from src.data.imaging import compute_gadf, generate_image_pair

        flux = _make_flux(200)
        pair = generate_image_pair(flux)
        gadf = compute_gadf(flux)

        np.testing.assert_array_equal(
            pair[1], gadf, err_msg="Channel 1 does not match compute_gadf"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_degenerate_constant_input(self):
        """All-same-value flux must not crash and must return valid float32 arrays."""
        from src.data.imaging import compute_gadf, compute_recurrence_plot

        flux = np.ones(200, dtype=np.float64)

        rp = compute_recurrence_plot(flux)
        gadf = compute_gadf(flux)

        assert rp.shape == (64, 64)
        assert rp.dtype == np.float32
        assert np.all(np.isfinite(rp)), "RP contains non-finite values for constant input"

        assert gadf.shape == (64, 64)
        assert gadf.dtype == np.float32
        assert np.all(np.isfinite(gadf)), "GADF contains non-finite values for constant input"

    def test_short_input_resampled(self):
        """10-point flux must still produce (64, 64) output after resampling."""
        from src.data.imaging import compute_gadf, compute_recurrence_plot

        flux = _make_flux(10)

        rp = compute_recurrence_plot(flux)
        gadf = compute_gadf(flux)

        assert rp.shape == (64, 64), f"RP shape {rp.shape} != (64, 64)"
        assert gadf.shape == (64, 64), f"GADF shape {gadf.shape} != (64, 64)"

    def test_custom_size(self):
        """size=32 parameter must produce (32, 32) output for both transforms."""
        from src.data.imaging import compute_gadf, compute_recurrence_plot

        flux = _make_flux(200)

        rp = compute_recurrence_plot(flux, size=32)
        gadf = compute_gadf(flux, size=32)

        assert rp.shape == (32, 32), f"RP shape {rp.shape} != (32, 32)"
        assert gadf.shape == (32, 32), f"GADF shape {gadf.shape} != (32, 32)"
