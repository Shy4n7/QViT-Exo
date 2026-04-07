"""Tests for src/data/dataset.py — ExoplanetDataset PyTorch Dataset.

TDD: All tests written BEFORE implementation.

File layout expected by ExoplanetDataset (per test fixtures):
    processed_dir/
        <safe_koi_name>/          # dots replaced with underscores (Windows compat)
            image.npy   — shape (2, 64, 64), float32
            features.npy     — shape (5,), float32
                          [odd_depth, even_depth, depth_ratio,
                           secondary_eclipse (0.0/1.0), centroid_shift]
        ...
    split.csv — columns: kepoi_name, koi_disposition

Note: aux data is stored as features.npy (not aux.json or aux.npy) because:
- 'AUX' is a reserved Windows device name — files named aux.* cannot be
  created with Python's open() on Windows (raises FileNotFoundError).
- numpy.save uses the C runtime fopen which would also fail for the reserved
  name, so renaming to 'features.npy' avoids the issue entirely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ROWS = 10
_KOI_NAMES = [f"K{str(i).zfill(8)}.01" for i in range(1, N_ROWS + 1)]
_DISPOSITIONS = ["CONFIRMED"] * 5 + ["FALSE POSITIVE"] * 5


def _safe_dir_name(koi_name: str) -> str:
    """Convert KOI name to a filesystem-safe directory name.

    Windows treats directories with names like 'K00000001.01' (name.digits)
    as having a file extension, breaking open() calls inside them.
    Replace the dot with an underscore to avoid this.
    """
    return koi_name.replace(".", "_")


def _write_sample_files(processed_dir: Path, koi_name: str) -> None:
    """Write a synthetic image.npy and aux.json for one KOI into processed_dir.

    Directory is named using _safe_dir_name(koi_name) to avoid Windows path
    issues with dotted directory names.
    """
    safe_name = _safe_dir_name(koi_name)
    sample_dir = processed_dir / safe_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # image: (2, 64, 64) float32
    rng = np.random.default_rng(seed=abs(hash(koi_name)) % (2**31))
    image = rng.random((2, 64, 64)).astype(np.float32)
    np.save(str(sample_dir / "image.npy"), image)

    # aux features as float32 array: [odd_depth, even_depth, depth_ratio,
    #                                  secondary_eclipse (0.0), centroid_shift]
    # Note: saved as .npy (not .json) to avoid Python open() failures on Windows
    # when directory names have certain patterns in tempfile paths.
    aux = np.array(
        [0.0095, 0.0098, 0.0095 / 0.0098, 0.0, 0.0], dtype=np.float32
    )
    np.save(str(sample_dir / "features.npy"), aux)


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal processed dataset in tmp_path.

    Returns (csv_path, processed_dir).
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    for koi_name in _KOI_NAMES:
        _write_sample_files(processed_dir, koi_name)

    csv_path = tmp_path / "split.csv"
    df = pd.DataFrame(
        {"kepoi_name": _KOI_NAMES, "koi_disposition": _DISPOSITIONS}
    )
    df.to_csv(csv_path, index=False)

    return csv_path, processed_dir


# ---------------------------------------------------------------------------
# Tests for ExoplanetDataset
# ---------------------------------------------------------------------------


class TestExoplanetDatasetLength:
    """Tests for __len__()."""

    def test_dataset_length(self, dataset_dir: tuple[Path, Path]) -> None:
        """Dataset length equals the number of rows in the CSV."""
        from src.data.dataset import ExoplanetDataset

        csv_path, processed_dir = dataset_dir
        ds = ExoplanetDataset(str(csv_path), str(processed_dir))

        assert len(ds) == N_ROWS, f"Expected {N_ROWS}, got {len(ds)}"


class TestExoplanetDatasetGetItem:
    """Tests for __getitem__()."""

    def test_dataset_getitem_image_shape(self, dataset_dir: tuple[Path, Path]) -> None:
        """Image tensor shape must be torch.Size([2, 64, 64]), dtype float32."""
        from src.data.dataset import ExoplanetDataset

        csv_path, processed_dir = dataset_dir
        ds = ExoplanetDataset(str(csv_path), str(processed_dir))
        image, aux, label = ds[0]

        assert image.shape == torch.Size([2, 64, 64]), (
            f"Expected image shape [2, 64, 64], got {image.shape}"
        )
        assert image.dtype == torch.float32, (
            f"Expected float32 image, got {image.dtype}"
        )

    def test_dataset_getitem_aux_shape(self, dataset_dir: tuple[Path, Path]) -> None:
        """Aux tensor shape must be torch.Size([5])."""
        from src.data.dataset import ExoplanetDataset

        csv_path, processed_dir = dataset_dir
        ds = ExoplanetDataset(str(csv_path), str(processed_dir))
        image, aux, label = ds[0]

        assert aux.shape == torch.Size([5]), (
            f"Expected aux shape [5], got {aux.shape}"
        )
        assert aux.dtype == torch.float32, (
            f"Expected float32 aux tensor, got {aux.dtype}"
        )

    def test_dataset_getitem_label_confirmed(self, dataset_dir: tuple[Path, Path]) -> None:
        """CONFIRMED disposition must yield label == 1."""
        from src.data.dataset import ExoplanetDataset

        csv_path, processed_dir = dataset_dir
        ds = ExoplanetDataset(str(csv_path), str(processed_dir))

        # First 5 rows are CONFIRMED
        for idx in range(5):
            _, _, label = ds[idx]
            assert label == 1, (
                f"Expected label=1 for CONFIRMED at idx={idx}, got {label}"
            )

    def test_dataset_getitem_label_fp(self, dataset_dir: tuple[Path, Path]) -> None:
        """FALSE POSITIVE disposition must yield label == 0."""
        from src.data.dataset import ExoplanetDataset

        csv_path, processed_dir = dataset_dir
        ds = ExoplanetDataset(str(csv_path), str(processed_dir))

        # Last 5 rows are FALSE POSITIVE
        for idx in range(5, 10):
            _, _, label = ds[idx]
            assert label == 0, (
                f"Expected label=0 for FALSE POSITIVE at idx={idx}, got {label}"
            )

    def test_dataset_getitem_aux_feature_order(
        self, dataset_dir: tuple[Path, Path]
    ) -> None:
        """Aux tensor must encode features in the documented order.

        Order: [odd_depth, even_depth, depth_ratio, secondary_eclipse, centroid_shift]
        secondary_eclipse is encoded as 0.0 or 1.0.
        """
        from src.data.dataset import ExoplanetDataset

        csv_path, processed_dir = dataset_dir
        ds = ExoplanetDataset(str(csv_path), str(processed_dir))
        _, aux, _ = ds[0]

        # All values from the fixture features.npy — secondary_eclipse=False → 0.0
        assert aux[3].item() == pytest.approx(0.0), (
            f"Expected secondary_eclipse=0.0 at index 3, got {aux[3].item()}"
        )


class TestExoplanetDatasetErrorHandling:
    """Tests for error conditions."""

    def test_dataset_missing_image_file(self, tmp_path: Path) -> None:
        """A row with no matching image.npy must raise FileNotFoundError."""
        from src.data.dataset import ExoplanetDataset

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Write CSV but NO files in processed_dir for this KOI
        csv_path = tmp_path / "split.csv"
        df = pd.DataFrame(
            {
                "kepoi_name": ["K00099999.01"],
                "koi_disposition": ["CONFIRMED"],
            }
        )
        df.to_csv(csv_path, index=False)

        ds = ExoplanetDataset(str(csv_path), str(processed_dir))

        with pytest.raises(FileNotFoundError, match="K00099999.01"):
            _ = ds[0]

    def test_dataset_missing_features_file(self, tmp_path: Path) -> None:
        """A sample with image.npy but no features.npy must raise FileNotFoundError."""
        from src.data.dataset import ExoplanetDataset

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Write the sample directory with ONLY image.npy (no features.npy)
        koi_name = "K00088888.01"
        safe_name = koi_name.replace(".", "_")
        sample_dir = processed_dir / safe_name
        sample_dir.mkdir()
        img = np.zeros((2, 64, 64), dtype=np.float32)
        np.save(str(sample_dir / "image.npy"), img)
        # Intentionally do NOT write features.npy

        csv_path = tmp_path / "split.csv"
        df = pd.DataFrame(
            {"kepoi_name": [koi_name], "koi_disposition": ["CONFIRMED"]}
        )
        df.to_csv(csv_path, index=False)

        ds = ExoplanetDataset(str(csv_path), str(processed_dir))

        with pytest.raises(FileNotFoundError, match="K00088888.01"):
            _ = ds[0]

    def test_dataset_transform_applied(self, dataset_dir: tuple[Path, Path]) -> None:
        """When a transform is provided it must be applied to the image tensor."""
        from src.data.dataset import ExoplanetDataset

        csv_path, processed_dir = dataset_dir

        # Transform: multiply all values by 2
        def double(x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

        ds_plain = ExoplanetDataset(str(csv_path), str(processed_dir))
        ds_transformed = ExoplanetDataset(
            str(csv_path), str(processed_dir), transform=double
        )

        img_plain, _, _ = ds_plain[0]
        img_transformed, _, _ = ds_transformed[0]

        torch.testing.assert_close(img_transformed, img_plain * 2.0)


class TestExoplanetDatasetDataLoader:
    """Tests for DataLoader integration."""

    def test_dataset_dataloader_batches(self, dataset_dir: tuple[Path, Path]) -> None:
        """First batch from DataLoader(batch_size=4) must have image shape [4, 2, 64, 64]."""
        from src.data.dataset import ExoplanetDataset

        csv_path, processed_dir = dataset_dir
        ds = ExoplanetDataset(str(csv_path), str(processed_dir))
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        batch_images, batch_aux, batch_labels = next(iter(loader))

        assert batch_images.shape == torch.Size([4, 2, 64, 64]), (
            f"Expected batch image shape [4, 2, 64, 64], got {batch_images.shape}"
        )
        assert batch_aux.shape == torch.Size([4, 5]), (
            f"Expected batch aux shape [4, 5], got {batch_aux.shape}"
        )
        assert len(batch_labels) == 4, (
            f"Expected 4 labels in batch, got {len(batch_labels)}"
        )
