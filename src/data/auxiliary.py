"""Auxiliary diagnostic feature extraction for exoplanet transit vetting.

All public functions are PURE — they return new values and never mutate
their inputs.

Features computed:
    odd_even_depth      — compare transit depths for odd vs even transit numbers
    secondary_eclipse_flag — detect a flux dip at phase ~0.5 (EB diagnostic)
    centroid_shift      — mean in-/out-of-transit centroid displacement (NaN if unavailable)
    extract_auxiliary   — convenience wrapper returning an AuxFeatures dataclass
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AuxFeatures:
    """Container for all auxiliary diagnostic features.

    Attributes
    ----------
    odd_depth:
        Median flux suppression during odd-numbered transits (fractional depth).
    even_depth:
        Median flux suppression during even-numbered transits (fractional depth).
    depth_ratio:
        odd_depth / even_depth.  NaN when even_depth is zero or near-zero.
    secondary_eclipse:
        True when a flux dip exceeding the threshold is detected at phase ~0.5.
    centroid_shift:
        Mean displacement of the photometric centroid in-transit vs out-of-transit.
        NaN when a TPF is unavailable or centroid computation fails.
    centroid_available:
        False when tpf=None or centroid computation failed.
    """

    odd_depth: float
    even_depth: float
    depth_ratio: float        # odd/even, or NaN if even_depth == 0
    secondary_eclipse: bool
    centroid_shift: float     # NaN if TPF unavailable
    centroid_available: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Minimum absolute value of even_depth below which depth_ratio is set to NaN
# to avoid division by near-zero values.
_MIN_DEPTH_FOR_RATIO: float = 1e-10


def odd_even_depth(
    flux: np.ndarray,
    time: np.ndarray,
    period: float,
    epoch: float,
) -> tuple[float, float]:
    """Measure transit depth for odd and even transit numbers separately.

    Transit numbers are assigned using floor division of elapsed periods since
    the reference epoch.  Transit 0 is even; transit 1 is odd; etc.

    The transit window is determined adaptively by phase-folding the light
    curve onto a 200-bin grid and identifying bins that are more than 2σ
    below the median.  This avoids hard-coding a fixed fraction of the period
    and correctly handles both narrow and wide transits without contaminating
    adjacent out-of-transit cadences.

    Depth is defined as:
        1 − median(in-transit normalised flux)
    where the light curve is normalised so the out-of-transit median is 1.

    Parameters
    ----------
    flux:
        Flux array.  Not mutated.
    time:
        Time array with the same length as *flux*.  Not mutated.
    period:
        Orbital period (same units as *time*).
    epoch:
        Reference transit epoch (same units as *time*).

    Returns
    -------
    tuple[float, float]
        (odd_depth, even_depth) — both non-negative floats.
        Returns (0.0, 0.0) when fewer than two transits are found or when
        no transit signal is detected in the phase-folded light curve.
    """
    # Work on copies so we never mutate the caller's arrays
    _time = np.array(time, copy=True, dtype=float)
    _flux = np.array(flux, copy=True, dtype=float)

    # --- Adaptive transit window via phase folding ---
    # Phase in [0, 1)
    phase_01 = ((_time - epoch) / period) % 1.0
    # Floor-division transit number (0 = first transit at/after epoch)
    transit_number = np.floor((_time - epoch) / period).astype(int)
    # Phase centred at primary transit: map to [-0.5, 0.5)
    phase_centered = np.where(phase_01 >= 0.5, phase_01 - 1.0, phase_01)

    # Build a coarse phase-folded light curve to detect the transit window
    n_phase_bins = 200
    bin_edges = np.linspace(-0.5, 0.5, n_phase_bins + 1)
    bin_idx = np.digitize(phase_centered, bin_edges[1:-1])  # 0 … n_phase_bins-1

    global_mean = float(np.mean(_flux))
    binned = np.full(n_phase_bins, global_mean)
    for b in range(n_phase_bins):
        pts = _flux[bin_idx == b]
        if len(pts) > 0:
            binned[b] = float(np.mean(pts))

    median_f = float(np.median(binned))
    std_f = float(np.std(binned))

    # Identify in-transit phase bins (> 2σ below median)
    if std_f > 0.0:
        transit_bin_centres = bin_edges[:-1][binned < median_f - 2.0 * std_f] + (
            0.5 * (bin_edges[1] - bin_edges[0])
        )
    else:
        transit_bin_centres = np.array([])

    if len(transit_bin_centres) == 0:
        # No transit detected — fallback to 5% of period as half-width
        half_width = 0.05
    else:
        span = float(transit_bin_centres.max() - transit_bin_centres.min())
        half_width = span / 2.0 + (bin_edges[1] - bin_edges[0])

    # Classify cadences as in-transit / out-of-transit
    in_transit_mask = np.abs(phase_centered) <= half_width

    out_of_transit_flux = _flux[~in_transit_mask]
    if len(out_of_transit_flux) == 0:
        return 0.0, 0.0

    baseline = float(np.median(out_of_transit_flux))
    if baseline == 0.0:
        baseline = 1.0

    norm_flux = _flux / baseline

    # Split by parity of transit number
    odd_mask = in_transit_mask & (np.abs(transit_number) % 2 == 1)
    even_mask = in_transit_mask & (np.abs(transit_number) % 2 == 0)

    odd_in = norm_flux[odd_mask]
    even_in = norm_flux[even_mask]

    odd_d = float(1.0 - np.median(odd_in)) if len(odd_in) > 0 else 0.0
    even_d = float(1.0 - np.median(even_in)) if len(even_in) > 0 else 0.0

    return max(odd_d, 0.0), max(even_d, 0.0)


def secondary_eclipse_flag(
    phase: np.ndarray,
    flux: np.ndarray,
    threshold: float = 0.0005,
) -> bool:
    """Check for a flux dip at phase ~0.5 (±0.1) exceeding *threshold* depth.

    A dip at phase 0.5 indicates a secondary eclipse — the companion passes
    behind the primary star.  This is a diagnostic signature of an eclipsing
    binary rather than a planetary transit.

    Parameters
    ----------
    phase:
        Phase array in [-0.5, 0.5).  Not mutated.
    flux:
        Flux array of the same length as *phase*.  Not mutated.
    threshold:
        Minimum fractional flux dip to count as a secondary eclipse.

    Returns
    -------
    bool
        True when the mean flux in the secondary-eclipse window is suppressed
        below the out-of-window baseline by more than *threshold*.
    """
    _phase = np.array(phase, copy=True, dtype=float)
    _flux = np.array(flux, copy=True, dtype=float)

    # Secondary eclipse window: |phase - 0.5| < 0.1 with wrap-around
    # Phase is in [-0.5, 0.5) so 0.5 wraps to just below -0.5.
    # Compute circular distance to 0.5.
    dist = np.abs(_phase - 0.5)
    dist_wrapped = np.minimum(dist, 1.0 - dist)
    eclipse_mask = dist_wrapped < 0.1

    # Out-of-eclipse window: also exclude primary transit window (|phase| < 0.1)
    primary_mask = np.abs(_phase) < 0.1
    baseline_mask = ~eclipse_mask & ~primary_mask

    if eclipse_mask.sum() == 0 or baseline_mask.sum() == 0:
        return False

    baseline = float(np.median(_flux[baseline_mask]))
    eclipse_level = float(np.median(_flux[eclipse_mask]))

    depth = baseline - eclipse_level
    return bool(depth > threshold)


def centroid_shift(tpf: object) -> float:
    """Compute mean in-transit minus out-of-transit centroid displacement.

    Currently returns NaN when *tpf* is None or centroid computation fails.
    A full implementation would use ``lightkurve.TessTargetPixelFile`` or
    ``lightkurve.KeplerTargetPixelFile`` centroid methods.

    Parameters
    ----------
    tpf:
        A lightkurve TargetPixelFile object, or None.

    Returns
    -------
    float
        Mean centroid displacement in pixels, or NaN if unavailable.
    """
    if tpf is None:
        return float("nan")

    try:
        # Attempt to compute centroid from the TPF
        col, row = tpf.estimate_centroids()
        # Return mean absolute displacement as a scalar summary
        col_arr = np.array(col, dtype=float)
        row_arr = np.array(row, dtype=float)
        shift = float(np.nanmean(np.sqrt(col_arr**2 + row_arr**2)))
        return shift if math.isfinite(shift) else float("nan")
    except Exception:
        return float("nan")


def extract_auxiliary(
    flux: np.ndarray,
    time: np.ndarray,
    phase: np.ndarray,
    period: float,
    epoch: float,
    tpf: object = None,
) -> AuxFeatures:
    """Extract all auxiliary diagnostic features into a single AuxFeatures object.

    All inputs are treated as immutable; no arrays are modified in place.

    Parameters
    ----------
    flux:
        Flux array.  Not mutated.
    time:
        Time array.  Not mutated.
    phase:
        Phase array in [-0.5, 0.5), same length as *flux* or independently sized.
        Not mutated.
    period:
        Orbital period (same units as *time*).
    epoch:
        Transit epoch (same units as *time*).
    tpf:
        Optional lightkurve TargetPixelFile for centroid analysis.  Pass None
        when unavailable.

    Returns
    -------
    AuxFeatures
        All computed auxiliary diagnostic features.
    """
    odd_d, even_d = odd_even_depth(flux, time, period=period, epoch=epoch)

    if even_d > _MIN_DEPTH_FOR_RATIO:
        ratio = odd_d / even_d
    else:
        ratio = float("nan")

    sec_eclipse = secondary_eclipse_flag(phase, flux[: len(phase)] if len(flux) >= len(phase) else flux)

    c_shift = centroid_shift(tpf)
    c_available = tpf is not None and not math.isnan(c_shift)

    return AuxFeatures(
        odd_depth=odd_d,
        even_depth=even_d,
        depth_ratio=ratio,
        secondary_eclipse=bool(sec_eclipse),
        centroid_shift=c_shift,
        centroid_available=c_available,
    )
