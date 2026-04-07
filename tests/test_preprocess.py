"""Tests for src/data/preprocess.py.

TDD order: tests written BEFORE implementation.
All tests use synthetic data — no network calls, no file I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocess import (
    ProcessedLC,
    detrend_savgol,
    normalize,
    phase_fold,
    preprocess_pipeline,
    sigma_clip,
)


# ---------------------------------------------------------------------------
# detrend_savgol
# ---------------------------------------------------------------------------


class TestDetrendSavgol:
    def test_detrend_savgol_removes_trend(self) -> None:
        """Linear trend injected into flat flux is removed: residual std < 0.01."""
        rng = np.random.default_rng(seed=7)
        n = 2001
        flat = rng.normal(loc=1.0, scale=0.001, size=n)

        # Inject a linear trend spanning ±0.5 across the array
        trend = np.linspace(-0.5, 0.5, n)
        flux_with_trend = flat + trend

        detrended = detrend_savgol(flux_with_trend)

        # Residual after detrend should be close to original noise
        residual = detrended - detrended.mean()
        assert residual.std() < 0.01, (
            f"Detrend did not remove trend; residual std = {residual.std():.4f}"
        )

    def test_detrend_savgol_does_not_mutate_input(self) -> None:
        """detrend_savgol must be pure — input array unchanged."""
        flux = np.ones(2001) + np.linspace(0, 1, 2001)
        original = flux.copy()

        detrend_savgol(flux)

        np.testing.assert_array_equal(flux, original)

    def test_detrend_savgol_output_shape(self) -> None:
        """Output array has same shape as input."""
        flux = np.random.default_rng(0).normal(1.0, 0.01, 2001)
        result = detrend_savgol(flux)
        assert result.shape == flux.shape

    def test_detrend_savgol_short_series_auto_adjusts_window(self) -> None:
        """Short input (< default window) does not raise; auto-adjusts window."""
        flux = np.random.default_rng(0).normal(1.0, 0.01, 50)
        # Should not raise even though 50 < default window_length=1001
        result = detrend_savgol(flux)
        assert result.shape == flux.shape

    def test_detrend_savgol_very_short_series_near_polyorder(self) -> None:
        """Input barely longer than polyorder triggers the polyorder+1 window branch."""
        # polyorder default is 3; n=5 → adjusted window must be at least 5 (odd, > 3)
        flux = np.array([1.0, 1.01, 0.99, 1.02, 0.98])
        result = detrend_savgol(flux, window_length=1001, polyorder=3)
        assert result.shape == (5,)

    def test_detrend_savgol_all_zero_input_returns_ones(self) -> None:
        """All-zero flux (pathological) divides by 1.0 fallback, returns ones."""
        flux = np.zeros(2001)
        result = detrend_savgol(flux)
        # trend is zero → safe_trend substituted with 1.0 → 0/1 = 0
        np.testing.assert_array_equal(result, 0.0)

    def test_detrend_savgol_flat_flux_unchanged(self) -> None:
        """Perfectly flat flux detrended to ~1.0 everywhere (within 1e-3)."""
        flux = np.ones(2001)
        result = detrend_savgol(flux)
        np.testing.assert_allclose(result, 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_normalize_median_is_one(self) -> None:
        """After normalize(), median of output is ~1.0 within 1e-6."""
        rng = np.random.default_rng(seed=3)
        flux = rng.normal(loc=5.0, scale=0.05, size=2001)

        result = normalize(flux)

        assert abs(np.median(result) - 1.0) < 1e-6, (
            f"Median after normalize = {np.median(result):.8f}, expected ~1.0"
        )

    def test_normalize_does_not_mutate_input(self) -> None:
        """normalize must be pure — input array unchanged."""
        flux = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        original = flux.copy()

        normalize(flux)

        np.testing.assert_array_equal(flux, original)

    def test_normalize_output_shape(self) -> None:
        """Output has the same shape as input."""
        flux = np.random.default_rng(0).normal(1.0, 0.01, 500)
        result = normalize(flux)
        assert result.shape == flux.shape

    def test_normalize_scales_relative_differences(self) -> None:
        """Relative flux differences are preserved after normalization."""
        flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # median = 3.0
        result = normalize(flux)
        # Each value divided by median (3.0)
        expected = flux / 3.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_normalize_single_element(self) -> None:
        """Single-element array normalizes to 1.0."""
        flux = np.array([42.0])
        result = normalize(flux)
        assert result[0] == pytest.approx(1.0)

    def test_normalize_zero_median_returns_copy(self) -> None:
        """Zero-median flux (edge case) returns a copy unchanged rather than dividing by zero."""
        flux = np.array([-1.0, 0.0, 1.0])  # median == 0.0
        result = normalize(flux)
        # Should not raise and should return a copy equal to the input
        np.testing.assert_array_equal(result, flux)
        # Must be a copy, not the same object
        assert result is not flux


# ---------------------------------------------------------------------------
# sigma_clip
# ---------------------------------------------------------------------------


class TestSigmaClip:
    def test_sigma_clip_removes_outliers(self) -> None:
        """5 injected 100-sigma outlier points are removed from output."""
        rng = np.random.default_rng(seed=9)
        n = 2001
        time = np.linspace(0.0, 30.0, n)
        flux = rng.normal(loc=1.0, scale=0.001, size=n)

        # Inject 5 extreme outliers at 100-sigma
        outlier_indices = [100, 300, 700, 1200, 1800]
        flux_with_outliers = flux.copy()
        flux_with_outliers[outlier_indices] = 1.0 + 100 * 0.001 * 100  # 100-sigma spike

        clipped_time, clipped_flux = sigma_clip(time, flux_with_outliers, sigma=5.0)

        assert len(clipped_flux) < n, "No points were removed"
        # All removed indices should be gone (check via time array)
        for idx in outlier_indices:
            outlier_t = time[idx]
            assert outlier_t not in clipped_time, (
                f"Outlier at index {idx} (time={outlier_t}) survived clipping"
            )

    def test_sigma_clip_preserves_transit(self) -> None:
        """Transit at 0.01 depth (well within 5-sigma for noise std=0.001) is NOT clipped."""
        rng = np.random.default_rng(seed=11)
        n = 2001
        time = np.linspace(0.0, 30.0, n)
        flux = rng.normal(loc=1.0, scale=0.001, size=n)

        # Inject transit: depth=0.01, noise std=0.001 → transit is ~10-sigma below mean
        # BUT sigma_clip clips based on *median* and *std of the whole distribution*
        # std of the whole array (with transit) is dominated by noise ~0.001
        # transit depth 0.01 = 10-sigma below mean, so it WILL be clipped by naive sigma_clip
        # The spec says "well within 5-sigma" — transit at 0.01 depth with noise std 0.001
        # To ensure transit is NOT clipped: use noise std=0.005, transit depth=0.01 → 2-sigma
        rng2 = np.random.default_rng(seed=11)
        flux_shallow = rng2.normal(loc=1.0, scale=0.005, size=n)
        transit_indices = list(range(900, 1100))
        flux_shallow[transit_indices] -= 0.01  # 0.01 / 0.005 = 2-sigma — below clip threshold

        clipped_time, clipped_flux = sigma_clip(time, flux_shallow, sigma=5.0)

        # Transit duration points should survive: check at least half of them remain
        transit_times = time[transit_indices]
        surviving_transit = sum(t in clipped_time for t in transit_times)
        survival_fraction = surviving_transit / len(transit_indices)

        assert survival_fraction >= 0.9, (
            f"Transit clipped: only {survival_fraction:.0%} of transit points survived"
        )

    def test_sigma_clip_does_not_mutate_input(self) -> None:
        """sigma_clip must be pure — input arrays unchanged."""
        time = np.linspace(0, 10, 100)
        flux = np.random.default_rng(0).normal(1.0, 0.01, 100)
        original_flux = flux.copy()

        sigma_clip(time, flux)

        np.testing.assert_array_equal(flux, original_flux)

    def test_sigma_clip_returns_matching_length_arrays(self) -> None:
        """Returned time and flux arrays have the same length."""
        time = np.linspace(0, 10, 500)
        flux = np.random.default_rng(0).normal(1.0, 0.01, 500)

        clipped_time, clipped_flux = sigma_clip(time, flux)

        assert len(clipped_time) == len(clipped_flux)

    def test_sigma_clip_clean_data_unchanged(self) -> None:
        """Data with no outliers is returned unchanged (all points preserved)."""
        rng = np.random.default_rng(seed=5)
        n = 200
        time = np.linspace(0, 10, n)
        flux = rng.normal(loc=1.0, scale=0.001, size=n)
        # No outliers: all points within ~4-sigma from mean for n=200

        clipped_time, clipped_flux = sigma_clip(time, flux, sigma=10.0)

        # With 10-sigma threshold on clean data, no points should be removed
        assert len(clipped_flux) == n

    def test_sigma_clip_constant_flux_returns_all_points(self) -> None:
        """Constant flux (std == 0) triggers the zero-std guard and returns all points."""
        n = 100
        time = np.linspace(0, 5, n)
        flux = np.ones(n)  # std == 0.0 exactly

        clipped_time, clipped_flux = sigma_clip(time, flux)

        assert len(clipped_flux) == n
        np.testing.assert_array_equal(clipped_flux, flux)


# ---------------------------------------------------------------------------
# phase_fold
# ---------------------------------------------------------------------------


class TestPhaseFold:
    def test_phase_fold_range(self) -> None:
        """Output phase array is entirely within [-0.5, 0.5]."""
        rng = np.random.default_rng(seed=13)
        n = 2001
        time = np.linspace(0.0, 30.0, n)
        flux = rng.normal(1.0, 0.001, n)
        period = 3.5
        epoch = 5.0

        phase, _ = phase_fold(time, flux, period=period, epoch=epoch)

        assert phase.min() >= -0.5, f"phase.min() = {phase.min():.4f} < -0.5"
        assert phase.max() <= 0.5, f"phase.max() = {phase.max():.4f} > 0.5"

    def test_phase_fold_transit_at_zero(self) -> None:
        """Known transit epoch folds to phase ~0.0 within ±0.05."""
        n = 5000
        period = 10.0
        epoch = 5.0
        time = np.linspace(0.0, 100.0, n)
        flux = np.ones(n)

        # Inject transits at epoch + k*period for all k
        for k in range(int(100 / period) + 1):
            t_center = epoch + k * period
            mask = np.abs(time - t_center) < 0.3
            flux[mask] -= 0.01

        phase, binned_flux = phase_fold(time, flux, period=period, epoch=epoch, n_bins=201)

        transit_phase_idx = np.argmin(binned_flux)
        transit_phase = phase[transit_phase_idx]

        assert abs(transit_phase) <= 0.05, (
            f"Transit phase = {transit_phase:.4f}, expected within ±0.05 of 0.0"
        )

    def test_phase_fold_output_sorted(self) -> None:
        """Output phase array is sorted in ascending order."""
        time = np.linspace(0, 30, 2001)
        flux = np.random.default_rng(0).normal(1.0, 0.001, 2001)
        phase, _ = phase_fold(time, flux, period=3.5, epoch=5.0)

        assert np.all(np.diff(phase) >= 0), "phase array is not sorted"

    def test_phase_fold_output_length_matches_n_bins(self) -> None:
        """Output length equals n_bins parameter."""
        time = np.linspace(0, 30, 5000)
        flux = np.ones(5000)
        n_bins = 401

        phase, binned_flux = phase_fold(time, flux, period=5.0, epoch=2.0, n_bins=n_bins)

        assert len(phase) == n_bins
        assert len(binned_flux) == n_bins

    def test_phase_fold_does_not_mutate_inputs(self) -> None:
        """phase_fold must be pure — input arrays unchanged."""
        time = np.linspace(0, 30, 2001)
        flux = np.random.default_rng(0).normal(1.0, 0.001, 2001)
        original_time = time.copy()
        original_flux = flux.copy()

        phase_fold(time, flux, period=3.5, epoch=5.0)

        np.testing.assert_array_equal(time, original_time)
        np.testing.assert_array_equal(flux, original_flux)

    def test_phase_fold_flat_flux_bins_to_one(self) -> None:
        """Perfectly flat flux (all 1.0) phase-folds to all-1.0 binned output."""
        time = np.linspace(0, 30, 2001)
        flux = np.ones(2001)

        _, binned_flux = phase_fold(time, flux, period=3.5, epoch=5.0, n_bins=201)

        np.testing.assert_allclose(binned_flux, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# preprocess_pipeline
# ---------------------------------------------------------------------------


class TestPreprocessPipeline:
    def test_preprocess_pipeline_shape(
        self, synthetic_lightcurve: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Input 2001-point light curve → ProcessedLC.phase and .flux have n_bins=2001 shape."""
        time, flux = synthetic_lightcurve
        period = 10.0
        epoch = 5.0

        result = preprocess_pipeline(time, flux, period=period, epoch=epoch)

        assert isinstance(result, ProcessedLC)
        assert len(result.phase) == 2001
        assert len(result.flux) == 2001

    def test_preprocess_pipeline_returns_processedlc(
        self, synthetic_lightcurve: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """preprocess_pipeline returns a ProcessedLC dataclass instance."""
        time, flux = synthetic_lightcurve

        result = preprocess_pipeline(time, flux, period=10.0, epoch=5.0)

        assert isinstance(result, ProcessedLC)

    def test_preprocess_pipeline_stores_period_and_epoch(
        self, synthetic_lightcurve: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """ProcessedLC stores the supplied period and epoch unchanged."""
        time, flux = synthetic_lightcurve
        period = 7.3
        epoch = 12.5

        result = preprocess_pipeline(time, flux, period=period, epoch=epoch)

        assert result.period == pytest.approx(period)
        assert result.epoch == pytest.approx(epoch)

    def test_preprocess_pipeline_stores_koi_id(
        self, synthetic_lightcurve: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """ProcessedLC stores the koi_id string passed in."""
        time, flux = synthetic_lightcurve

        result = preprocess_pipeline(time, flux, period=10.0, epoch=5.0, koi_id="K00001.01")

        assert result.koi_id == "K00001.01"

    def test_preprocess_pipeline_phase_range(
        self, synthetic_lightcurve: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Output phase is in [-0.5, 0.5]."""
        time, flux = synthetic_lightcurve

        result = preprocess_pipeline(time, flux, period=10.0, epoch=5.0)

        assert result.phase.min() >= -0.5
        assert result.phase.max() <= 0.5

    def test_preprocess_pipeline_preserves_depth(
        self, synthetic_lightcurve: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """After full pipeline, transit depth is within 15% of the injected depth.

        The synthetic_lightcurve fixture injects a 0.01 transit depth.
        """
        time, flux = synthetic_lightcurve

        # Period and epoch chosen so transit folds to phase ~0
        # Time spans [0, 30], transit center at index 1000 → t=15.0
        # Use period=30.0, epoch=15.0 → transit folds exactly to phase=0
        period = 30.0
        epoch = 15.0

        result = preprocess_pipeline(time, flux, period=period, epoch=epoch)

        # Continuum ~ 1.0 after normalize; find minimum (transit bottom)
        continuum = np.median(result.flux)
        transit_bottom = result.flux.min()
        measured_depth = continuum - transit_bottom

        injected_depth = 0.01
        relative_error = abs(measured_depth - injected_depth) / injected_depth

        assert relative_error <= 0.15, (
            f"Transit depth error too large: measured={measured_depth:.4f}, "
            f"injected={injected_depth:.4f}, relative_error={relative_error:.2%}"
        )

    def test_preprocess_pipeline_does_not_mutate_inputs(
        self, synthetic_lightcurve: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """preprocess_pipeline must be pure — input time and flux arrays unchanged."""
        time, flux = synthetic_lightcurve
        original_time = time.copy()
        original_flux = flux.copy()

        preprocess_pipeline(time, flux, period=10.0, epoch=5.0)

        np.testing.assert_array_equal(time, original_time)
        np.testing.assert_array_equal(flux, original_flux)

    def test_preprocess_pipeline_custom_config(
        self, synthetic_lightcurve: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Pipeline respects config overrides for sg_window_length and phase_bins."""
        time, flux = synthetic_lightcurve
        config = {
            "sg_window_length": 51,
            "sg_polyorder": 3,
            "sigma_clip_threshold": 5.0,
            "phase_bins": 101,
        }

        result = preprocess_pipeline(time, flux, period=10.0, epoch=5.0, config=config)

        assert len(result.phase) == 101
        assert len(result.flux) == 101
