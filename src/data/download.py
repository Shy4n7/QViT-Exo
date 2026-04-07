"""Download Kepler light curves for exoplanet transit vetting.

Public API
----------
download_lightcurve(koi_id, cache_dir, timeout) -> LightCurve | None
    Download a single Kepler long-cadence PDCSAP light curve.
    Checks local cache first; logs WARNING and returns None when not found.

batch_download(koi_ids, cache_dir, n_workers, timeout) -> (dict, list[str])
    Download multiple light curves in parallel.
    Returns (results dict, failed koi_id list).

Cache layout
------------
    cache_dir/
        <koi_id>/              # e.g. "K00001_01/"  (dots → underscores)
            lightcurve.fits    # FITS format via lightkurve (safe, no pickle)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import lightkurve as lk

logger = logging.getLogger(__name__)

# FITS is used instead of pickle to avoid arbitrary code execution on cache load.
_CACHE_FILENAME: str = "lightcurve.fits"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir_for(koi_id: str, cache_dir: str) -> Path:
    """Return the per-KOI subdirectory inside *cache_dir*."""
    safe_name = koi_id.replace("/", "_").replace("\\", "_").replace(".", "_")
    return Path(cache_dir) / safe_name


def _load_from_cache(koi_id: str, cache_dir: str) -> Any | None:
    """Return a cached LightCurve from FITS if it exists, else None."""
    fits_path = _cache_dir_for(koi_id, cache_dir) / _CACHE_FILENAME
    if fits_path.exists():
        try:
            return lk.read(str(fits_path))
        except Exception as exc:
            logger.warning("Failed to read cached FITS for KOI %s: %s", koi_id, exc)
    return None


def _save_to_cache(lc: Any, koi_id: str, cache_dir: str) -> None:
    """Persist *lc* to the local cache as FITS for *koi_id*."""
    koi_dir = _cache_dir_for(koi_id, cache_dir)
    koi_dir.mkdir(parents=True, exist_ok=True)
    fits_path = koi_dir / _CACHE_FILENAME
    lc.to_fits(str(fits_path), overwrite=True)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def download_lightcurve(
    koi_id: str,
    cache_dir: str,
    timeout: int = 60,
) -> Any | None:
    """Download Kepler long-cadence PDCSAP flux for *koi_id*.

    Checks local cache first.  On a cache miss, calls
    ``lightkurve.search_lightcurve`` and downloads the first result.

    Parameters
    ----------
    koi_id : str
        Kepler Object of Interest identifier, e.g. ``"K00001.01"``.
    cache_dir : str
        Root directory for the local cache.
    timeout : int
        Reserved for future use (lightkurve does not expose a timeout
        parameter at the module level, but the argument is accepted for
        API consistency).

    Returns
    -------
    LightCurve | None
        The downloaded (or cached) light curve, or ``None`` if not found.
        Never raises; logs a WARNING when the KOI cannot be retrieved.
    """
    # 1. Cache hit — skip search entirely
    cached = _load_from_cache(koi_id, cache_dir)
    if cached is not None:
        logger.debug("Cache hit for KOI %s", koi_id)
        return cached

    # 2. Search via lightkurve
    try:
        search_result = lk.search_lightcurve(
            koi_id,
            mission="Kepler",
            exptime="long",
        )
    except Exception as exc:
        logger.warning(
            "search_lightcurve raised an exception for KOI %s: %s",
            koi_id,
            exc,
        )
        return None

    if len(search_result) == 0:
        logger.warning(
            "No Kepler light curves found for KOI %s. "
            "Check that the KOI identifier is correct.",
            koi_id,
        )
        return None

    # 3. Download the first result
    try:
        lc = search_result.download()
    except Exception as exc:
        logger.warning(
            "Failed to download light curve for KOI %s: %s",
            koi_id,
            exc,
        )
        return None

    # 4. Persist to cache for future calls
    try:
        _save_to_cache(lc, koi_id, cache_dir)
    except Exception as exc:
        # Cache write failure is non-fatal
        logger.debug("Could not cache light curve for KOI %s: %s", koi_id, exc)

    return lc


def batch_download(
    koi_ids: list[str],
    cache_dir: str,
    n_workers: int = 4,
    timeout: int = 60,
) -> tuple[dict[str, Any | None], list[str]]:
    """Download light curves for all *koi_ids* in parallel.

    Each KOI is processed by ``download_lightcurve``.  Failures (exceptions
    or missing results) are collected in *failed* and logged; they do not
    interrupt processing of other KOIs.

    Parameters
    ----------
    koi_ids : list[str]
        KOI identifiers to download.
    cache_dir : str
        Root directory for the local cache.
    n_workers : int
        Number of parallel worker threads.
    timeout : int
        Per-download timeout forwarded to ``download_lightcurve``.

    Returns
    -------
    tuple[dict[str, LightCurve | None], list[str]]
        ``(results, failed)`` where:
        - ``results`` maps every koi_id to its LightCurve (or None).
        - ``failed`` contains koi_ids for which retrieval failed or returned None.
    """
    results: dict[str, Any | None] = {kid: None for kid in koi_ids}
    failed: list[str] = []

    def _worker(kid: str) -> tuple[str, Any | None]:
        try:
            lc = download_lightcurve(koi_id=kid, cache_dir=cache_dir, timeout=timeout)
            return kid, lc
        except Exception as exc:
            logger.warning("Unexpected error downloading KOI %s: %s", kid, exc)
            return kid, None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, kid): kid for kid in koi_ids}
        for future in as_completed(futures):
            kid = futures[future]
            try:
                returned_kid, lc = future.result()
                results[returned_kid] = lc
                if lc is None:
                    failed.append(returned_kid)
            except Exception as exc:
                logger.warning(
                    "Worker future raised for KOI %s: %s", kid, exc
                )
                results[kid] = None
                failed.append(kid)

    return results, failed
