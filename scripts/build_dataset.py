"""Phase 1 — Download and preprocess the full Kepler DR25 dataset.

Steps
-----
1. Fetch KOI DR25 catalog from NASA Exoplanet Archive TAP.
2. Filter to CONFIRMED + FALSE POSITIVE dispositions.
3. Create stratified train / val / test splits.
4. For each KOI: download Kepler LC → preprocess → generate RP+GADF image
   → extract auxiliary features → save image.npy + features.npy.

Usage
-----
python scripts/build_dataset.py
python scripts/build_dataset.py --config configs/data_config.yaml --workers 4
python scripts/build_dataset.py --dry-run  # process only first 20 KOIs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.catalog import fetch_koi_catalog, filter_dispositions, create_splits
from src.data.download import download_lightcurve
from src.data.preprocess import preprocess_pipeline
from src.data.imaging import generate_image_pair
from src.data.auxiliary import extract_auxiliary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "data_config.yaml"


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def _process_koi(
    row: pd.Series,
    cache_dir: str,
    output_dir: str,
    config: dict,
) -> bool:
    """Download, preprocess, and save one KOI.

    Returns True on success, False on any failure.
    """
    koi_id   = str(row["kepoi_name"])
    kepid    = row.get("kepid")
    period   = float(row.get("koi_period", 0) or 0)
    epoch    = float(row.get("koi_time0bk", 0) or 0)  # BKJD (BJD - 2454833)

    if period <= 0 or epoch <= 0:
        logger.debug("KOI %s: missing period/epoch, skipping.", koi_id)
        return False
    if not kepid or (isinstance(kepid, float) and np.isnan(kepid)):
        logger.debug("KOI %s: missing kepid, skipping.", koi_id)
        return False

    # lightkurve uses KIC ID for Kepler searches
    lk_target = f"KIC {int(kepid)}"

    # Determine output directory
    safe_name = koi_id.replace(".", "_")
    sample_dir = Path(output_dir) / safe_name
    image_path = sample_dir / "image.npy"
    feat_path  = sample_dir / "features.npy"
    if image_path.exists() and feat_path.exists():
        return True  # already processed

    # Download light curve (search by KIC ID; cache keyed by KOI ID)
    lc = download_lightcurve(lk_target, cache_dir=cache_dir,
                             timeout=config.get("download_timeout", 60))
    if lc is None:
        return False

    try:
        time_arr = np.array(lc.time.value,  dtype=np.float64)
        flux_arr = np.array(lc.flux.value,  dtype=np.float64)
    except Exception as exc:
        logger.debug("KOI %s: LC attribute error: %s", koi_id, exc)
        return False

    # Remove NaNs
    finite = np.isfinite(flux_arr) & np.isfinite(time_arr)
    time_arr, flux_arr = time_arr[finite], flux_arr[finite]
    if len(flux_arr) < 100:
        return False

    # Preprocess
    try:
        plc = preprocess_pipeline(
            time_arr, flux_arr, period=period, epoch=epoch,
            koi_id=koi_id, config=config,
        )
    except Exception as exc:
        logger.debug("KOI %s: preprocess failed: %s", koi_id, exc)
        return False

    # Image pair
    try:
        image_size = int(config.get("image_size", 64))
        image_np = generate_image_pair(plc.flux, size=image_size)
    except Exception as exc:
        logger.debug("KOI %s: imaging failed: %s", koi_id, exc)
        return False

    # Auxiliary features
    try:
        aux = extract_auxiliary(
            flux=flux_arr, time=time_arr,
            phase=plc.phase, period=period, epoch=epoch,
        )
        feat_np = np.array([
            aux.odd_depth, aux.even_depth, aux.depth_ratio,
            float(aux.secondary_eclipse), aux.centroid_shift,
        ], dtype=np.float32)
    except Exception:
        feat_np = np.zeros(5, dtype=np.float32)

    # Save
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(image_path), image_np)
    np.save(str(feat_path),  feat_np)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Kepler DR25 processed dataset")
    p.add_argument("--config",   default=str(_DEFAULT_CONFIG))
    p.add_argument("--workers",  type=int, default=None,
                   help="Override download_workers from config.")
    p.add_argument("--dry-run",  action="store_true",
                   help="Process only first 20 KOIs (testing).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config) as f:
        config: dict = yaml.safe_load(f)

    if args.workers is not None:
        config["download_workers"] = args.workers

    cache_dir  = config.get("cache_dir",  "data/raw")
    output_dir = config.get("output_dir", "data/processed")
    splits_dir = config.get("splits_dir", "data/splits")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(splits_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Fetch catalog ────────────────────────────────────────────────
    catalog_path = Path(splits_dir) / "koi_catalog_raw.csv"
    if catalog_path.exists():
        logger.info("Loading cached catalog from %s…", catalog_path)
        raw_df = pd.read_csv(catalog_path)
    else:
        logger.info("Fetching KOI DR25 catalog from NASA Exoplanet Archive…")
        raw_df = fetch_koi_catalog(str(catalog_path), config)

    logger.info("Raw catalog: %d rows.", len(raw_df))

    # ── Step 2: Filter ────────────────────────────────────────────────────────
    df = filter_dispositions(raw_df)
    logger.info("After filtering (CONFIRMED + FALSE POSITIVE): %d rows.", len(df))

    # ── Step 3: Splits ────────────────────────────────────────────────────────
    train_df, val_df, test_df = create_splits(df, config)
    train_df.to_csv(Path(splits_dir) / "train.csv", index=False)
    val_df.to_csv(  Path(splits_dir) / "val.csv",   index=False)
    test_df.to_csv( Path(splits_dir) / "test.csv",  index=False)
    logger.info(
        "Splits saved → train=%d  val=%d  test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Step 4: Process all KOIs ─────────────────────────────────────────────
    all_rows = df.to_dict("records")
    if args.dry_run:
        all_rows = all_rows[:20]
        logger.info("Dry run: processing first 20 KOIs only.")

    n_ok = n_fail = 0
    for i, row_dict in enumerate(all_rows, 1):
        row = pd.Series(row_dict)
        koi_id = row.get("kepoi_name", f"row_{i}")
        ok = _process_koi(row, cache_dir=cache_dir,
                          output_dir=output_dir, config=config)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
        if i % 100 == 0:
            logger.info("[%d/%d]  ok=%d  fail=%d", i, len(all_rows), n_ok, n_fail)

    logger.info("Dataset build complete. ok=%d  fail=%d", n_ok, n_fail)
    logger.info("Processed samples → %s", output_dir)
    logger.info("Split CSVs       → %s", splits_dir)


if __name__ == "__main__":
    main()
