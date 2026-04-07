"""Light-curve preprocessing pipeline.

All public functions are PURE — they return new arrays and never mutate
their inputs.  This makes the pipeline safe for concurrent use and easy
to test.

Pipeline order:
    detrend_savgol  →  normalize  →  sigma_clip  →  phase_fold
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

# Default pipeline hyper-parameters (can be overridden via config dict).
#
# _DEFAULT_SG_WINDOW is intentionally large (1001 cadences ≈ 20 days for
# Kepler long-cadence data) so that typical transit durations (≤ 15 % of the
# window) are not absorbed by the SG baseline fit.  Using a window that is
# shorter than ~5 × transit_duration suppresses transit depth by >15 %.
_DEFAULT_SG_WINDOW: int = 1001
_DEFAULT_SG_POLYORDER: int = 3
_DEFAULT_SIGMA: float = 5.0
_DEFAULT_N_BINS: int = 2001


@dataclass
class ProcessedLC:
    """Container for a phase-folded, preprocessed light curve.

    Attributes
    ----------
    phase:
        Phase values in [-0.5, 0.5], sorted ascending.
    flux:
        Normalised, detrended, sigma-clipped flux binned onto `phase`.
    period:
        Orbital period used for phase folding (days).
    epoch:
        Transit epoch used for phase folding (BJD - 2454833, Kepler convention).
    koi_id:
        KOI identifier string (e.g. "K00001.01").  Empty string if not supplied.
    """

    phase: np.ndarray
    flux: np.ndarray
    period: float
    epoch: float
    koi_id: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detrend_savgol(
    flux: np.ndarray,
    window_length: int = _DEFAULT_SG_WINDOW,
    polyorder: int = _DEFAULT_SG_POLYORDER,
) -> np.ndarray:
    """Remove a slowly-varying trend using a Savitzky-Golay filter.

    The trend estimate is divided out (flux / trend) rather than
    subtracted, so the result is a fractional deviation around 1.0.

    Parameters
    ----------
    flux:
        Input flux array. Not mutated.
    window_length:
        SG filter window length (must be odd and > polyorder).
        Automatically reduced to the largest valid odd value if the
        input is shorter than the default window.
    polyorder:
        SG filter polynomial order.

    Returns
    -------
    np.ndarray
        Detrended flux array of the same shape as *flux*.
    """
    n = len(flux)

    # Auto-adjust window to fit short series
    adjusted_window = min(window_length, n)
    # Window must be odd
    if adjusted_window % 2 == 0:
        adjusted_window -= 1
    # Window must be > polyorder
    if adjusted_window <= polyorder:
        adjusted_window = polyorder + 1
        if adjusted_window % 2 == 0:
            adjusted_window += 1

    trend = savgol_filter(flux, window_length=adjusted_window, polyorder=polyorder)
    # Avoid division by zero for pathological (all-zero) inputs
    safe_trend = np.where(trend == 0.0, 1.0, trend)
    return flux / safe_trend


def normalize(flux: np.ndarray) -> np.ndarray:
    """Divide flux by its median so that the continuum sits at 1.0.

    Parameters
    ----------
    flux:
        Input flux array. Not mutated.

    Returns
    -------
    np.ndarray
        Normalised flux array of the same shape.
    """
    med = np.median(flux)
    if med == 0.0:
        return flux.copy()
    return flux / med


def sigma_clip(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float = _DEFAULT_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove flux points that deviate beyond *sigma* × std from the median.

    Parameters
    ----------
    time:
        Time array. Not mutated.
    flux:
        Flux array. Not mutated.
    sigma:
        Clipping threshold in units of the flux standard deviation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        New (time, flux) arrays with outlier points removed.
    """
    median = np.median(flux)
    std = np.std(flux)

    if std == 0.0:
        return time.copy(), flux.copy()

    mask = np.abs(flux - median) <= sigma * std
    return time[mask].copy(), flux[mask].copy()


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    n_bins: int = _DEFAULT_N_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-fold a light curve and bin it onto a uniform phase grid.

    The transit is centred at phase 0.  Each bin is the mean of all flux
    points that fall within it; empty bins are filled with the global
    flux mean.

    Parameters
    ----------
    time:
        Time array (arbitrary units matching *epoch*). Not mutated.
    flux:
        Flux array of the same length as *time*. Not mutated.
    period:
        Orbital period.
    epoch:
        Reference transit epoch (same units as *time*).
    n_bins:
        Number of phase bins in the output.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (phase, binned_flux) both length *n_bins*, sorted by ascending phase.
        Phase values are the bin-centre positions in [-0.5, 0.5).
    """
    # Map time → phase in [-0.5, 0.5)
    raw_phase = ((time - epoch) / period) % 1.0  # [0, 1)
    raw_phase = np.where(raw_phase >= 0.5, raw_phase - 1.0, raw_phase)  # [-0.5, 0.5)

    # Uniform phase bin edges and centres
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Bin indices for each data point  (values in 0 … n_bins-1)
    bin_indices = np.digitize(raw_phase, edges[1:-1])  # right-open bins

    global_mean = float(np.mean(flux))

    # Vectorised binning via np.bincount — O(N) instead of O(N × n_bins)
    counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float64)
    sums = np.bincount(bin_indices, weights=flux.astype(np.float64), minlength=n_bins)
    populated = counts > 0
    binned = np.where(populated, sums / np.where(populated, counts, 1.0), global_mean)
    binned = binned.astype(np.float64)

    # Sort by phase (centres are already sorted; binned matches)
    return centres.copy(), binned.copy()


def preprocess_pipeline(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    koi_id: str = "",
    config: dict[str, Any] | None = None,
) -> ProcessedLC:
    """Chain detrend → normalize → sigma_clip → phase_fold into one call.

    Parameters
    ----------
    time:
        Raw time array. Not mutated.
    flux:
        Raw flux array. Not mutated.
    period:
        Orbital period (same units as *time*).
    epoch:
        Transit epoch (same units as *time*).
    koi_id:
        Optional KOI identifier stored in the result.
    config:
        Optional dict to override default hyper-parameters:
            sg_window_length  (int,   default 1001)
            sg_polyorder      (int,   default 3)
            sigma_clip_threshold (float, default 5.0)
            phase_bins        (int,   default 2001)

    Returns
    -------
    ProcessedLC
        Phase-folded, preprocessed light curve.
    """
    cfg = config or {}
    sg_window: int = int(cfg.get("sg_window_length", _DEFAULT_SG_WINDOW))
    sg_order: int = int(cfg.get("sg_polyorder", _DEFAULT_SG_POLYORDER))
    sigma: float = float(cfg.get("sigma_clip_threshold", _DEFAULT_SIGMA))
    n_bins: int = int(cfg.get("phase_bins", _DEFAULT_N_BINS))

    # Step 1 — detrend (returns new array)
    flux_detrended = detrend_savgol(flux, window_length=sg_window, polyorder=sg_order)

    # Step 2 — normalize (returns new array)
    flux_normalized = normalize(flux_detrended)

    # Step 3 — sigma clip (returns new arrays)
    time_clipped, flux_clipped = sigma_clip(time, flux_normalized, sigma=sigma)

    # Step 4 — phase fold & bin (returns new arrays)
    phase_arr, binned_flux = phase_fold(
        time_clipped, flux_clipped, period=period, epoch=epoch, n_bins=n_bins
    )

    return ProcessedLC(
        phase=phase_arr,
        flux=binned_flux,
        period=period,
        epoch=epoch,
        koi_id=koi_id,
    )
