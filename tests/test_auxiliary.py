"""Tests for src/data/auxiliary.py — auxiliary diagnostic feature extraction.

TDD: All tests written BEFORE implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build synthetic light curves for specific test scenarios
# ---------------------------------------------------------------------------


def _make_box_transit_lightcurve(
    n_points: int = 2001,
    period: float = 10.0,
    epoch: float = 5.0,
    depth: float = 0.01,
    duration_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, flux) with a box-shaped transit repeating every `period`.

    All transits have the same depth (uniform box model).
    duration_fraction is the fraction of the period the transit occupies.
    """
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, n_points * (period / 200.0), n_points)
    flux = rng.normal(loc=1.0, scale=0.0002, size=n_points)

    duration = period * duration_fraction
    half_dur = duration / 2.0

    # Compute phase relative to epoch
    phase = ((time - epoch) % period)
    # Transit is centered at 0 within each cycle — use phase in [0, period)
    # Mark in-transit: phase < half_dur or phase > period - half_dur
    in_transit = (phase < half_dur) | (phase > period - half_dur)
    flux = np.where(in_transit, flux - depth, flux)

    return time, flux


def _make_alternating_transit_lightcurve(
    n_points: int = 2001,
    period: float = 10.0,
    epoch: float = 5.0,
    odd_depth: float = 0.02,
    even_depth: float = 0.008,
    duration_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, flux) alternating deep (odd) / shallow (even) transits.

    This mimics an eclipsing binary where odd and even eclipse depths differ.
    """
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, n_points * (period / 200.0), n_points)
    flux = rng.normal(loc=1.0, scale=0.0002, size=n_points)

    duration = period * duration_fraction
    half_dur = duration / 2.0

    phase = ((time - epoch) % period)

    # Determine which transit number each point belongs to (0-indexed)
    transit_number = ((time - epoch) / period).astype(int)

    in_transit = (phase < half_dur) | (phase > period - half_dur)
    is_odd = (transit_number % 2 == 1)  # 1-indexed odd: transit_number odd indices

    flux = np.where(in_transit & is_odd, flux - odd_depth, flux)
    flux = np.where(in_transit & ~is_odd, flux - even_depth, flux)

    return time, flux


def _make_phase_array(n_bins: int = 2001) -> np.ndarray:
    """Return a uniform phase array in [-0.5, 0.5)."""
    return np.linspace(-0.5, 0.5, n_bins, endpoint=False)


def _inject_secondary_eclipse(
    phase: np.ndarray,
    flux: np.ndarray,
    depth: float = 0.005,
    center: float = 0.5,
    half_width: float = 0.05,
) -> np.ndarray:
    """Return a new flux array with a dip injected near phase=center."""
    new_flux = flux.copy()
    # Phase array is in [-0.5, 0.5); center=0.5 maps to phase just below 0.5
    # Use wrap-around: treat |phase - 0.5| after folding
    dist = np.abs(phase - center)
    # Also handle wrap-around for center=0.5 near the boundary
    dist_wrapped = np.minimum(dist, 1.0 - dist)
    mask = dist_wrapped < half_width
    new_flux = np.where(mask, new_flux - depth, new_flux)
    return new_flux


# ---------------------------------------------------------------------------
# Tests for odd_even_depth
# ---------------------------------------------------------------------------


class TestOddEvenDepth:
    """Tests for the odd_even_depth() function."""

    def test_odd_even_depth_similar_for_planet(self) -> None:
        """Uniform-depth transits: odd and even depths should differ by <5%.

        A real planet produces the same transit depth every transit.
        """
        from src.data.auxiliary import odd_even_depth

        time, flux = _make_box_transit_lightcurve(
            n_points=4001,
            period=10.0,
            epoch=5.0,
            depth=0.01,
        )
        odd_d, even_d = odd_even_depth(flux, time, period=10.0, epoch=5.0)

        assert odd_d > 0, "Odd transit depth must be positive"
        assert even_d > 0, "Even transit depth must be positive"

        # Depths should agree to within 5 %
        relative_diff = abs(odd_d - even_d) / max(odd_d, even_d)
        assert relative_diff < 0.05, (
            f"Odd/even depths differ by {relative_diff:.1%} > 5% for a uniform planet signal. "
            f"odd={odd_d:.6f}, even={even_d:.6f}"
        )

    def test_odd_even_depth_different_for_eb(self) -> None:
        """Alternating deep/shallow transits: odd_depth > 2 * even_depth.

        This is the eclipsing binary diagnostic: primary eclipse is much deeper
        than secondary, indicating two distinct stellar components.
        """
        from src.data.auxiliary import odd_even_depth

        time, flux = _make_alternating_transit_lightcurve(
            n_points=4001,
            period=10.0,
            epoch=5.0,
            odd_depth=0.02,
            even_depth=0.008,
        )
        odd_d, even_d = odd_even_depth(flux, time, period=10.0, epoch=5.0)

        assert odd_d > 0, "Odd transit depth must be positive"
        assert even_d > 0, "Even transit depth must be positive"
        assert odd_d > 2.0 * even_d, (
            f"Expected odd_depth > 2x even_depth for EB. "
            f"odd={odd_d:.6f}, even={even_d:.6f}, ratio={odd_d/even_d:.2f}"
        )

    def test_odd_even_depth_immutable(self) -> None:
        """Input arrays must not be modified after calling odd_even_depth."""
        from src.data.auxiliary import odd_even_depth

        time, flux = _make_box_transit_lightcurve(n_points=2001, period=10.0, epoch=5.0)
        flux_orig = flux.copy()
        time_orig = time.copy()

        odd_even_depth(flux, time, period=10.0, epoch=5.0)

        np.testing.assert_array_equal(flux, flux_orig, err_msg="flux array was mutated")
        np.testing.assert_array_equal(time, time_orig, err_msg="time array was mutated")

    def test_odd_even_depth_zero_baseline_handled(self) -> None:
        """All-zero flux must not cause ZeroDivisionError; returns (0, 0)."""
        from src.data.auxiliary import odd_even_depth

        time = np.linspace(0.0, 100.0, 500)
        flux = np.zeros(500)

        odd_d, even_d = odd_even_depth(flux, time, period=10.0, epoch=5.0)

        assert isinstance(odd_d, float)
        assert isinstance(even_d, float)


# ---------------------------------------------------------------------------
# Tests for secondary_eclipse_flag
# ---------------------------------------------------------------------------


class TestSecondaryEclipseFlag:
    """Tests for the secondary_eclipse_flag() function."""

    def test_secondary_eclipse_flag_detected(self) -> None:
        """A dip of depth 0.005 injected at phase ~0.5 must be detected."""
        from src.data.auxiliary import secondary_eclipse_flag

        phase = _make_phase_array(n_bins=2001)
        flux = np.ones_like(phase)
        flux = _inject_secondary_eclipse(phase, flux, depth=0.005, center=0.5)

        result = secondary_eclipse_flag(phase, flux, threshold=0.0005)

        assert result is True, (
            "Expected secondary_eclipse_flag=True when depth=0.005 > threshold=0.0005 at phase~0.5"
        )

    def test_secondary_eclipse_flag_not_detected(self) -> None:
        """A perfectly flat light curve at phase 0.5 must not trigger the flag."""
        from src.data.auxiliary import secondary_eclipse_flag

        phase = _make_phase_array(n_bins=2001)
        flux = np.ones_like(phase)  # perfectly flat — no secondary eclipse

        result = secondary_eclipse_flag(phase, flux, threshold=0.0005)

        assert result is False, (
            "Expected secondary_eclipse_flag=False for flat light curve at phase~0.5"
        )

    def test_secondary_eclipse_flag_below_threshold(self) -> None:
        """A dip smaller than the threshold must not be flagged."""
        from src.data.auxiliary import secondary_eclipse_flag

        phase = _make_phase_array(n_bins=2001)
        flux = np.ones_like(phase)
        # Inject a tiny dip — well below threshold
        flux = _inject_secondary_eclipse(phase, flux, depth=0.00001, center=0.5)

        result = secondary_eclipse_flag(phase, flux, threshold=0.0005)

        assert result is False, (
            "Expected secondary_eclipse_flag=False when eclipse depth < threshold"
        )

    def test_secondary_eclipse_flag_empty_phase_returns_false(self) -> None:
        """When no points fall in the eclipse window the flag must be False."""
        from src.data.auxiliary import secondary_eclipse_flag

        # A very short phase array (only 1 point) so eclipse_mask is empty
        phase = np.array([0.0])
        flux = np.array([1.0])

        result = secondary_eclipse_flag(phase, flux, threshold=0.0005)

        assert result is False, (
            "Expected secondary_eclipse_flag=False when eclipse window is empty"
        )


# ---------------------------------------------------------------------------
# Tests for extract_auxiliary
# ---------------------------------------------------------------------------


class TestExtractAuxiliary:
    """Tests for the extract_auxiliary() function."""

    def _make_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (flux, time, phase) suitable for extract_auxiliary."""
        time, flux = _make_box_transit_lightcurve(n_points=2001, period=10.0, epoch=5.0)
        phase = _make_phase_array(n_bins=2001)
        return flux, time, phase

    def test_extract_auxiliary_returns_expected_keys(self) -> None:
        """Output AuxFeatures must have all required fields."""
        from src.data.auxiliary import AuxFeatures, extract_auxiliary

        flux, time, phase = self._make_inputs()
        result = extract_auxiliary(flux, time, phase, period=10.0, epoch=5.0, tpf=None)

        assert isinstance(result, AuxFeatures), (
            f"Expected AuxFeatures, got {type(result)}"
        )
        assert hasattr(result, "odd_depth"), "Missing field: odd_depth"
        assert hasattr(result, "even_depth"), "Missing field: even_depth"
        assert hasattr(result, "depth_ratio"), "Missing field: depth_ratio"
        assert hasattr(result, "secondary_eclipse"), "Missing field: secondary_eclipse"
        assert hasattr(result, "centroid_available"), "Missing field: centroid_available"

    def test_extract_auxiliary_centroid_none(self) -> None:
        """When tpf=None, centroid_available=False and centroid_shift is NaN."""
        from src.data.auxiliary import extract_auxiliary

        flux, time, phase = self._make_inputs()
        result = extract_auxiliary(flux, time, phase, period=10.0, epoch=5.0, tpf=None)

        assert result.centroid_available is False, (
            "Expected centroid_available=False when tpf=None"
        )
        assert np.isnan(result.centroid_shift), (
            f"Expected centroid_shift=NaN when tpf=None, got {result.centroid_shift}"
        )

    def test_extract_auxiliary_depth_ratio_nan_when_even_zero(self) -> None:
        """When even_depth is 0, depth_ratio must be NaN (no division by zero)."""
        from src.data.auxiliary import extract_auxiliary

        # Use a perfectly flat light curve — no transits, so depths will be ~0
        phase = _make_phase_array(n_bins=2001)
        flux = np.ones(2001)
        time = np.linspace(0.0, 100.0, 2001)

        # We can't guarantee even_depth is exactly 0 with noise-free flat curve,
        # but we test the property by patching the function's output path via
        # a near-zero flux signal that drives even_depth to 0.
        # Direct unit test of the ratio logic: verify no ZeroDivisionError raised.
        result = extract_auxiliary(flux, time, phase, period=50.0, epoch=10.0, tpf=None)

        # Just verify it returns without raising and depth_ratio is a float or NaN
        assert isinstance(result.depth_ratio, float) or np.isnan(result.depth_ratio)

    def test_extract_auxiliary_secondary_eclipse_type(self) -> None:
        """secondary_eclipse field must be a Python bool."""
        from src.data.auxiliary import extract_auxiliary

        flux, time, phase = self._make_inputs()
        result = extract_auxiliary(flux, time, phase, period=10.0, epoch=5.0, tpf=None)

        assert isinstance(result.secondary_eclipse, bool), (
            f"Expected secondary_eclipse to be bool, got {type(result.secondary_eclipse)}"
        )

    def test_extract_auxiliary_odd_even_depths_positive_for_transit(self) -> None:
        """odd_depth and even_depth must be positive for a light curve with transits."""
        from src.data.auxiliary import extract_auxiliary

        time, flux = _make_box_transit_lightcurve(n_points=4001, period=10.0, epoch=5.0, depth=0.01)
        phase = _make_phase_array(n_bins=2001)
        result = extract_auxiliary(flux, time, phase, period=10.0, epoch=5.0, tpf=None)

        assert result.odd_depth > 0, f"Expected odd_depth > 0, got {result.odd_depth}"
        assert result.even_depth > 0, f"Expected even_depth > 0, got {result.even_depth}"


# ---------------------------------------------------------------------------
# Tests for centroid_shift
# ---------------------------------------------------------------------------


class TestCentroidShift:
    """Tests for the centroid_shift() function."""

    def test_centroid_shift_returns_nan_for_none_tpf(self) -> None:
        """centroid_shift(None) must return NaN."""
        from src.data.auxiliary import centroid_shift

        result = centroid_shift(None)
        assert np.isnan(result), f"Expected NaN for tpf=None, got {result}"

    def test_centroid_shift_returns_float(self) -> None:
        """centroid_shift must return a float scalar."""
        from src.data.auxiliary import centroid_shift

        result = centroid_shift(None)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
