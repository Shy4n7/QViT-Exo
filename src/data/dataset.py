"""PyTorch Dataset for exoplanet transit vetting.

Each sample consists of:
    - A 2-channel phase-folded image:  shape (2, 64, 64), dtype float32
    - An auxiliary feature vector:     shape (5,), dtype float32
      [odd_depth, even_depth, depth_ratio, secondary_eclipse (0/1), centroid_shift]
    - A binary label:                  int  (1 = CONFIRMED, 0 = FALSE POSITIVE)

On-disk layout expected under *processed_dir*::

    processed_dir/
        <safe_koi_name>/          # dots replaced with underscores (Windows compat)
            image.npy             # numpy array, shape (2, 64, 64), float32
            features.npy               # numpy array, shape (5,), float32
                                  # [odd_depth, even_depth, depth_ratio,
                                  #  secondary_eclipse (0.0/1.0), centroid_shift]
        ...

Note: aux data is stored as ``features.npy`` (not JSON) because Python's ``open()``
for text-mode file creation fails on certain directory name patterns inside
Windows temp directories, while numpy's C-level fopen works reliably.

The split CSV must have at least two columns: ``kepoi_name`` and ``koi_disposition``.
Valid disposition values: ``"CONFIRMED"`` (→ 1) and ``"FALSE POSITIVE"`` (→ 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, int] = {
    "CONFIRMED": 1,
    "FALSE POSITIVE": 0,
}

# Number of auxiliary features per sample (must match features.npy shape[0]).
_N_AUX_FEATURES: int = 5

# Feature order within features.npy (for documentation purposes):
#   0: odd_depth
#   1: even_depth
#   2: depth_ratio
#   3: secondary_eclipse  (0.0 = False, 1.0 = True)
#   4: centroid_shift     (0.0 when unavailable)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_dir_name(koi_name: str) -> str:
    """Convert a KOI name to a filesystem-safe directory name.

    Directories whose names contain a dot followed by digits (e.g. K00001.01)
    cause ``open()`` failures on Windows because the OS interprets the suffix
    as a file extension.  Replacing the dot with an underscore avoids this.
    """
    return koi_name.replace(".", "_")


_NAN_SENTINELS: dict[int, float] = {
    2: 1.0,   # depth_ratio: 1.0 = unavailable (odd/even indistinguishable)
    4: -1.0,  # centroid_shift: -1.0 = not measured (physically impossible value)
}
"""Sentinel values for NaN features.

Using 0.0 for centroid_shift would be ambiguous (0.0 = no centroid motion,
which is the expected value for a real planet). Using -1.0 as a sentinel
allows the model to distinguish "measured and zero" from "not measured".
Similarly, depth_ratio=1.0 is the neutral value (equal odd/even depths).
"""


def _load_aux_tensor(aux_path: Path) -> torch.Tensor:
    """Load features.npy and return a float32 tensor of shape (5,).

    NaN values are replaced with physically-motivated sentinels:
    - depth_ratio (index 2): NaN → 1.0  (neutral / unmeasured)
    - centroid_shift (index 4): NaN → -1.0  (impossible value = not measured)

    The aux array must have exactly ``_N_AUX_FEATURES`` elements in the
    documented order: [odd_depth, even_depth, depth_ratio,
    secondary_eclipse (0.0/1.0), centroid_shift].
    """
    arr: np.ndarray = np.load(str(aux_path)).astype(np.float32)
    for idx, sentinel in _NAN_SENTINELS.items():
        if idx < len(arr) and np.isnan(arr[idx]):
            arr[idx] = np.float32(sentinel)
    # Replace any remaining NaNs (e.g. odd_depth, even_depth) with 0.0
    arr = np.where(np.isnan(arr), np.float32(0.0), arr)
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ExoplanetDataset(Dataset):
    """PyTorch Dataset for exoplanet transit vetting.

    Parameters
    ----------
    split_csv_path:
        Path to a CSV with columns ``kepoi_name`` and ``koi_disposition``.
    processed_dir:
        Root directory containing one sub-directory per sample (named after
        the KOI, with dots replaced by underscores for filesystem safety).
    transform:
        Optional callable applied to the image tensor after loading.
    skip_missing:
        If True, silently skip samples whose ``image.npy`` or ``features.npy``
        are not yet on disk (useful when the dataset build is still running).
        If False (default), raise FileNotFoundError on first access.
    """

    def __init__(
        self,
        split_csv_path: str,
        processed_dir: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        skip_missing: bool = False,
    ) -> None:
        self._processed_dir = Path(processed_dir)
        self._transform = transform

        df = pd.read_csv(split_csv_path)
        records = []
        for _, row in df.iterrows():
            disp = row["koi_disposition"]
            if disp not in LABEL_MAP:
                raise ValueError(
                    f"Unknown disposition '{disp}' for KOI '{row['kepoi_name']}'. "
                    f"Expected one of {list(LABEL_MAP.keys())}. "
                    "Run filter_dispositions() on the catalog before constructing the dataset."
                )
            if skip_missing:
                safe = _safe_dir_name(row["kepoi_name"])
                sample_dir = self._processed_dir / safe
                if not (sample_dir / "image.npy").exists() or not (sample_dir / "features.npy").exists():
                    continue
            records.append((row["kepoi_name"], LABEL_MAP[disp]))
        self._records: list[tuple[str, int]] = records

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (image_tensor, aux_tensor, label).

        Parameters
        ----------
        idx:
            Integer index into the dataset.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, int]
            - image_tensor: shape (2, 64, 64), dtype float32
            - aux_tensor:   shape (5,), dtype float32
            - label:        0 or 1

        Raises
        ------
        FileNotFoundError
            When the image.npy or features.npy files are missing for the
            requested sample.  The error message includes the KOI name for
            easy diagnosis.
        """
        koi_name, label = self._records[idx]
        safe_name = _safe_dir_name(koi_name)
        sample_dir = self._processed_dir / safe_name

        image_path = sample_dir / "image.npy"
        aux_path = sample_dir / "features.npy"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found for KOI '{koi_name}': {image_path}"
            )
        if not aux_path.exists():
            raise FileNotFoundError(
                f"Aux file not found for KOI '{koi_name}': {aux_path}"
            )

        # Load image — shape (2, 64, 64), dtype float32
        image_np: np.ndarray = np.load(str(image_path)).astype(np.float32)
        image_tensor = torch.from_numpy(image_np)

        if self._transform is not None:
            image_tensor = self._transform(image_tensor)

        aux_tensor = _load_aux_tensor(aux_path)

        return image_tensor, aux_tensor, label
