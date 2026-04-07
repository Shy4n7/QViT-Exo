"""Seed management for deterministic training and data processing.

Responsibilities
----------------
- Set the global random state for the Python ``random`` module, NumPy, and
  PyTorch (CPU and CUDA) to a given integer seed.
- Configure PyTorch cuDNN to run deterministically.

Design notes
------------
- ``set_seed`` is a deliberate side-effect function (it mutates global RNG
  state); it is nonetheless kept narrow: no return value, no hidden branching.
- Input validation is performed up-front so callers receive clear errors
  rather than cryptic failures deep inside a framework.
- CUDA seeding calls are always executed; they are no-ops on CPU-only builds,
  which is the desired behaviour.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible experiments.

    Parameters
    ----------
    seed:
        Non-negative integer seed value.  Must satisfy ``0 <= seed <= 2**32 - 1``
        to remain compatible with NumPy's legacy ``RandomState`` interface.

    Raises
    ------
    TypeError
        If *seed* is not an ``int`` (or ``int`` subclass).
    ValueError
        If *seed* is negative.
    """
    if not isinstance(seed, int):
        raise TypeError(
            f"seed must be an int, got {type(seed).__name__!r}."
        )

    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}.")

    # Python built-in RNG
    random.seed(seed)

    # NumPy legacy RNG (used by most scientific code and sklearn)
    np.random.seed(seed)

    # PyTorch CPU RNG
    torch.manual_seed(seed)

    # PyTorch CUDA RNG (no-op on CPU-only builds)
    torch.cuda.manual_seed_all(seed)

    # Ensure cuDNN uses deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Note: torch.use_deterministic_algorithms(True) is intentionally NOT called
    # here because it triggers the torch._inductor TORCH_LIBRARY double-registration
    # bug on PyTorch 2.5.1 + Windows when optimizer.step() is called. Call it
    # explicitly in training scripts if needed:
    #   torch.use_deterministic_algorithms(True, warn_only=True)
