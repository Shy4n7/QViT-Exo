"""Shared pytest fixtures for the exoplanet detection test suite."""

import importlib
import numpy as np
import pandas as pd
import pytest
import torch

# Pre-import heavyweight modules before pytest's AssertionRewritingHook can intercept
# them. These modules register C++ extensions and TORCH_LIBRARY backends at import
# time; if pytest rewrites and re-executes them on a second import, the registration
# fails with "FAKE c10d backend already exist" or "Only a single TORCH_LIBRARY..."
# on PyTorch 2.5.1 + Windows.
for _mod in (
    "torch._inductor.test_operators",
    "torchvision",
    "timm",
):
    try:
        importlib.import_module(_mod)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_torch_determinism():
    """Reset torch.use_deterministic_algorithms after each test.

    torch.use_deterministic_algorithms is process-global state. Tests in
    test_reproducibility.py set it to True, which causes optimizer.step()
    to trigger the torch._inductor double-registration bug on PyTorch 2.5.1
    + Windows. Resetting after each test isolates the side effect.
    """
    yield
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


@pytest.fixture
def synthetic_lightcurve() -> tuple[np.ndarray, np.ndarray]:
    """Return (time, flux) arrays simulating a single Kepler quarter.

    Spec:
    - 2001 cadence points (matching phase_bins in data_config.yaml)
    - Flat continuum at 1.0 with Gaussian noise (std=0.001)
    - Box-shaped transit injected at the center: depth=0.01, duration=200 pts
    - Random seed fixed to 0 for determinism across the suite

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (time, flux) both shape (2001,), dtype float64
    """
    rng = np.random.default_rng(seed=0)
    n_points = 2001
    time = np.linspace(0.0, 30.0, n_points)  # ~30 days, Kepler long-cadence scale

    flux = rng.normal(loc=1.0, scale=0.001, size=n_points)

    # Inject box transit centered at midpoint
    center = n_points // 2       # index 1000
    half_duration = 100          # 200-point total duration
    transit_depth = 0.01
    start = center - half_duration
    stop = center + half_duration
    flux[start:stop] -= transit_depth

    return time, flux


@pytest.fixture
def mock_koi_catalog() -> pd.DataFrame:
    """Return a minimal KOI catalog DataFrame with 10 rows.

    Columns match the subset used by the data pipeline:
        kepoi_name, koi_disposition, koi_period, koi_time0bk,
        koi_depth, koi_duration, koi_ror, ra, dec

    Rows: 5 CONFIRMED, 5 FALSE POSITIVE — deterministic values.
    """
    n_confirmed = 5
    n_false_positive = 5
    total = n_confirmed + n_false_positive

    rng = np.random.default_rng(seed=1)

    kepoi_names = [f"K{str(i).zfill(8)}.01" for i in range(1, total + 1)]
    dispositions = ["CONFIRMED"] * n_confirmed + ["FALSE POSITIVE"] * n_false_positive

    data = {
        "kepoi_name": kepoi_names,
        "koi_disposition": dispositions,
        "koi_period": rng.uniform(1.0, 400.0, total),
        "koi_time0bk": rng.uniform(100.0, 200.0, total),
        "koi_depth": rng.uniform(100.0, 10_000.0, total),   # ppm
        "koi_duration": rng.uniform(1.0, 15.0, total),      # hours
        "koi_ror": rng.uniform(0.01, 0.25, total),
        "ra": rng.uniform(280.0, 300.0, total),
        "dec": rng.uniform(40.0, 52.0, total),
    }

    return pd.DataFrame(data)
