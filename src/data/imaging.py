"""Light-curve to image transforms for the exoplanet transit vetting pipeline.

Two transforms are provided:
- Recurrence Plot (RP): encodes temporal recurrence structure as a distance matrix.
- Gramian Angular Difference Field (GADF): encodes temporal correlations in polar coordinates.

Both operate on 1D flux arrays and return (size, size) float32 images.
``generate_image_pair`` stacks both into a (2, size, size) tensor ready for a
2-channel CNN / Vision Transformer input.

All functions are pure (no mutation of inputs).
"""

from __future__ import annotations

import numpy as np
from pyts.image import GramianAngularField, RecurrencePlot
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample(flux: np.ndarray, target_size: int) -> np.ndarray:
    """Resample a 1D flux array to *target_size* points via linear interpolation.

    Parameters
    ----------
    flux:
        1D float array of arbitrary length >= 2.
    target_size:
        Number of output points.

    Returns
    -------
    np.ndarray
        New 1D array of length *target_size*; input is never mutated.
    """
    n = len(flux)
    if n == target_size:
        return flux.copy()

    x_original = np.linspace(0.0, 1.0, n)
    x_target = np.linspace(0.0, 1.0, target_size)
    interpolator = interp1d(x_original, flux, kind="linear", assume_sorted=True)
    return interpolator(x_target)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_recurrence_plot(flux: np.ndarray, size: int = 64) -> np.ndarray:
    """Convert a 1D flux array to a Recurrence Plot distance image.

    Steps
    -----
    1. Resample *flux* to *size* points.
    2. Apply ``pyts.image.RecurrencePlot`` (dimension=1, threshold='point',
       percentage=20) which returns values in [0, 1] where 1 = recurrent
       (small distance).
    3. Invert: ``distance = 1 - recurrence`` so that the main diagonal
       (self-recurrence) becomes 0.0.
    4. Cast to float32.

    Parameters
    ----------
    flux:
        1D array of flux values.
    size:
        Target image dimension; output shape is (size, size).

    Returns
    -------
    np.ndarray
        Shape (size, size), dtype float32, values in [0.0, 1.0].
        Main diagonal is 0.0 (zero self-distance).
    """
    resampled = _resample(flux, size)
    # pyts expects (n_samples, n_timestamps)
    x = resampled.reshape(1, -1)

    transformer = RecurrencePlot(dimension=1, threshold="point", percentage=20)
    rp = transformer.fit_transform(x)[0]  # shape (size, size), float64, values in [0,1]

    # Invert so that the diagonal represents zero distance (self-similarity)
    distance_matrix = 1.0 - rp
    return distance_matrix.astype(np.float32)


def compute_gadf(flux: np.ndarray, size: int = 64) -> np.ndarray:
    """Convert a 1D flux array to a Gramian Angular Difference Field image.

    Steps
    -----
    1. Resample *flux* to *size* points.
    2. Apply ``pyts.image.GramianAngularField(method='difference', image_size=size)``.
    3. Clip to [-1.0, 1.0] to guard against floating-point overshoot.
    4. Cast to float32.

    Parameters
    ----------
    flux:
        1D array of flux values.
    size:
        Target image dimension; output shape is (size, size).

    Returns
    -------
    np.ndarray
        Shape (size, size), dtype float32, values in [-1.0, 1.0].
    """
    resampled = _resample(flux, size)
    x = resampled.reshape(1, -1)

    transformer = GramianAngularField(method="difference", image_size=size)
    gadf = transformer.fit_transform(x)[0]  # shape (size, size), float64

    clipped = np.clip(gadf, -1.0, 1.0)
    return clipped.astype(np.float32)


def generate_image_pair(flux: np.ndarray, size: int = 64) -> np.ndarray:
    """Generate a 2-channel image from a 1D flux array.

    Channel 0: Recurrence Plot (via ``compute_recurrence_plot``).
    Channel 1: Gramian Angular Difference Field (via ``compute_gadf``).

    Parameters
    ----------
    flux:
        1D array of flux values.
    size:
        Target spatial dimension; output shape is (2, size, size).

    Returns
    -------
    np.ndarray
        Shape (2, size, size), dtype float32.
    """
    rp = compute_recurrence_plot(flux, size=size)
    gadf = compute_gadf(flux, size=size)
    return np.stack([rp, gadf], axis=0)
