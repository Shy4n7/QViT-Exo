"""Download TESS light curves for exoplanet transit vetting.

Public API
----------
download_tess_lightcurve(tic_id, cache_dir, sectors, exptime, timeout) -> LightCurve | None
    Download TESS PDCSAP light curve(s) for a TIC target.  Multiple sectors
    are stitched into a single LightCurve.  Checks local cache first.

batch_download_tess(tic_ids, cache_dir, sectors, exptime, n_workers, timeout)
    -> (dict[str, LightCurve | None], list[str])
    Parallel download for a list of TIC IDs.

Cache layout
------------
    cache_dir/
        TIC_<id>_all/           # all available sectors
            lightcurve.fits
        TIC_<id>_s<lo>_<hi>/    # sector range [lo, hi]
            lightcurve.fits
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import lightkurve as lk

logger = logging.getLogger(__name__)

_CACHE_FILENAME: str = "lightcurve.fits"


# ---------------------------------------------------------------------------
# TIC ID normalisation
# ---------------------------------------------------------------------------

def _tic_target(tic_id: int | str) -> str:
    """Normalise a TIC ID to the ``'TIC <integer>'`` string lightkurve expects.

    Accepted input formats: ``123456789``, ``"123456789"``, ``"TIC 123456789"``,
    ``"tic123456789"`` (case-insensitive, leading/trailing whitespace ignored).
    """
    raw = str(tic_id).strip().upper().replace("TIC", "").strip()
    return f"TIC {int(raw)}"


def _cache_key(tic_id: int | str, sectors: Sequence[int] | None) -> str:
    """Return a filesystem-safe cache directory name for this (TIC, sectors) pair."""
    raw = str(tic_id).strip().upper().replace("TIC", "").strip()
    tic_int = int(raw)
    if sectors is None:
        return f"TIC_{tic_int}_all"
    lo = min(sectors)
    hi = max(sectors)
    return f"TIC_{tic_int}_s{lo}_{hi}"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir_for(
    tic_id: int | str,
    cache_dir: str,
    sectors: Sequence[int] | None,
) -> Path:
    return Path(cache_dir) / _cache_key(tic_id, sectors)


def _load_from_cache(
    tic_id: int | str,
    cache_dir: str,
    sectors: Sequence[int] | None,
) -> Any | None:
    """Return cached LightCurve from FITS if it exists, else None."""
    fits_path = _cache_dir_for(tic_id, cache_dir, sectors) / _CACHE_FILENAME
    if fits_path.exists():
        try:
            return lk.read(str(fits_path))
        except Exception as exc:
            logger.warning("Failed to read cached FITS for TIC %s: %s", tic_id, exc)
    return None


def _save_to_cache(
    lc: Any,
    tic_id: int | str,
    cache_dir: str,
    sectors: Sequence[int] | None,
) -> None:
    """Persist *lc* to the local cache as FITS."""
    target_dir = _cache_dir_for(tic_id, cache_dir, sectors)
    target_dir.mkdir(parents=True, exist_ok=True)
    fits_path = target_dir / _CACHE_FILENAME
    lc.to_fits(str(fits_path), overwrite=True)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def download_tess_lightcurve(
    tic_id: int | str,
    cache_dir: str,
    sectors: Sequence[int] | None = None,
    exptime: str = "short",
    timeout: int = 60,
) -> Any | None:
    """Download TESS PDCSAP light curve(s) for a TIC target.

    All available matching sectors are downloaded and stitched into a single
    ``LightCurve``.  NaN-flux cadences are removed before caching.

    Parameters
    ----------
    tic_id : int | str
        TIC identifier.  Accepted formats: integer, ``"123456789"``,
        ``"TIC 123456789"`` (case-insensitive).
    cache_dir : str
        Root directory for the local FITS cache.
    sectors : sequence of int | None
        When given, only the specified TESS sectors are searched.  Pass
        ``None`` to retrieve all available sectors.
    exptime : str
        Exposure-time selector: ``"short"`` (2-min), ``"long"`` (10-min FFI),
        or ``"fast"`` (20-sec).  Defaults to ``"short"``.
    timeout : int
        Reserved for API consistency; lightkurve does not expose a timeout.

    Returns
    -------
    LightCurve | None
        Stitched, NaN-cleaned light curve, or ``None`` when no data are found.
        Never raises; logs a WARNING on failure.
    """
    cached = _load_from_cache(tic_id, cache_dir, sectors)
    if cached is not None:
        logger.debug("Cache hit for TIC %s (sectors=%s)", tic_id, sectors)
        return cached

    target = _tic_target(tic_id)

    # Search via lightkurve
    try:
        search_kwargs: dict[str, Any] = {"mission": "TESS", "exptime": exptime}
        if sectors is not None:
            search_kwargs["sector"] = list(sectors)
        search_result = lk.search_lightcurve(target, **search_kwargs)
    except Exception as exc:
        logger.warning("search_lightcurve raised for TIC %s: %s", tic_id, exc)
        return None

    if len(search_result) == 0:
        logger.warning(
            "No TESS %s-cadence light curves found for TIC %s (sectors=%s).",
            exptime, tic_id, sectors,
        )
        return None

    # Download all matching sectors and stitch
    try:
        lc_collection = search_result.download_all(flux_column="pdcsap_flux")
        lc = lc_collection.stitch()
    except Exception as exc:
        logger.warning("Download/stitch failed for TIC %s: %s", tic_id, exc)
        return None

    # Remove NaN flux cadences before caching
    try:
        lc = lc.remove_nans()
    except Exception:
        pass  # Non-fatal: proceed with un-cleaned LC

    try:
        _save_to_cache(lc, tic_id, cache_dir, sectors)
    except Exception as exc:
        logger.debug("Could not cache light curve for TIC %s: %s", tic_id, exc)

    return lc


def batch_download_tess(
    tic_ids: list[int | str],
    cache_dir: str,
    sectors: Sequence[int] | None = None,
    exptime: str = "short",
    n_workers: int = 4,
    timeout: int = 60,
) -> tuple[dict[str, Any | None], list[str]]:
    """Download TESS light curves for all *tic_ids* in parallel.

    Parameters
    ----------
    tic_ids : list[int | str]
        TIC identifiers to download.
    cache_dir : str
        Root directory for the local cache.
    sectors : sequence of int | None
        Sector filter forwarded to ``download_tess_lightcurve``.
    exptime : str
        Exposure time selector forwarded to ``download_tess_lightcurve``.
    n_workers : int
        Number of parallel download threads.
    timeout : int
        Per-download timeout.

    Returns
    -------
    tuple[dict[str, LightCurve | None], list[str]]
        ``(results, failed)`` where ``results`` maps ``str(tic_id)`` to its
        LightCurve (or None) and ``failed`` lists ids that could not be fetched.
    """
    str_ids = [str(t) for t in tic_ids]
    results: dict[str, Any | None] = {tid: None for tid in str_ids}
    failed: list[str] = []

    def _worker(tid: str) -> tuple[str, Any | None]:
        try:
            lc = download_tess_lightcurve(
                tic_id=tid,
                cache_dir=cache_dir,
                sectors=sectors,
                exptime=exptime,
                timeout=timeout,
            )
            return tid, lc
        except Exception as exc:
            logger.warning("Unexpected error downloading TIC %s: %s", tid, exc)
            return tid, None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, tid): tid for tid in str_ids}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                returned_tid, lc = future.result()
                results[returned_tid] = lc
                if lc is None:
                    failed.append(returned_tid)
            except Exception as exc:
                logger.warning("Worker future raised for TIC %s: %s", tid, exc)
                results[tid] = None
                failed.append(tid)

    return results, failed
