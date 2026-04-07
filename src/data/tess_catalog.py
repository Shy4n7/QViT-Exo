"""TESS TOI catalog download, filtering, and cross-matching.

All public functions are pure (return new objects, never mutate inputs).
Network calls use exponential backoff.

TOI TFOPWG dispositions
-----------------------
PC   — Planet Candidate (unvetted; primary target for model screening)
APC  — Ambiguous Planet Candidate (also unvetted)
CP   — Confirmed Planet (skip; already in confirmed catalog)
KP   — Known Planet (skip; identified before TESS)
FP   — False Positive (skip; already rejected by follow-up)
FA   — False Alarm (skip; instrumental artifact)
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dispositions that represent unvetted/ambiguous candidates worth screening
_UNVETTED_DISPOSITIONS: frozenset[str] = frozenset({"PC", "APC", ""})

# ExoFOP TOI CSV download endpoint
_EXOFOP_URL: str = "https://exofop.ipac.caltech.edu/tess/download_toi.php"
_EXOFOP_PARAMS: dict[str, str] = {"sort": "toi", "output": "csv"}

# NASA Exoplanet Archive TAP endpoint
_NASA_TAP_URL: str = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# ADQL query: TIC IDs of confirmed / known TESS planets
_NASA_CONFIRMED_QUERY: str = (
    "SELECT tic_id,toi,tfopwg_disp FROM toi "
    "WHERE tfopwg_disp IN ('CP','KP')"
)

# BTJD offset: TESS BJD = BJD - BTJD_OFFSET
BTJD_OFFSET: float = 2457000.0


# ---------------------------------------------------------------------------
# Catalog download
# ---------------------------------------------------------------------------

def fetch_toi_catalog(
    output_path: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Download the full TESS TOI catalog from ExoFOP.

    Parameters
    ----------
    output_path : str
        Filesystem path where the CSV will be written.  Parent directories are
        created if they do not exist.
    max_retries : int
        Maximum number of HTTP retry attempts.

    Returns
    -------
    pd.DataFrame
        Raw TOI catalog DataFrame with stripped column names.

    Raises
    ------
    RuntimeError
        When all retry attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            logger.info(
                "Fetching ExoFOP TOI catalog (attempt %d/%d)…",
                attempt + 1, max_retries,
            )
            response = requests.get(_EXOFOP_URL, params=_EXOFOP_PARAMS, timeout=120)
            response.raise_for_status()
            # ExoFOP CSVs sometimes begin with comment lines starting with '#'
            df = pd.read_csv(StringIO(response.text), comment="#")
            df.columns = [c.strip() for c in df.columns]
            _save_csv(df, output_path)
            logger.info("TOI catalog saved → %s (%d rows)", output_path, len(df))
            return df
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Attempt %d failed: %s. Retrying in %ds…", attempt + 1, exc, wait
            )
            if attempt < max_retries - 1:
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to download TOI catalog after {max_retries} attempts. "
        f"Last error: {last_exc}"
    )


def fetch_confirmed_tic_ids(max_retries: int = 3) -> frozenset[int]:
    """Query NASA Exoplanet Archive for TIC IDs of confirmed/known TESS planets.

    Returns an empty frozenset (with a WARNING) when the query fails, so the
    calling code can continue without cross-matching rather than crashing.

    Parameters
    ----------
    max_retries : int
        Maximum number of HTTP retry attempts.

    Returns
    -------
    frozenset[int]
        TIC IDs of already-confirmed planets (CP + KP dispositions).
    """
    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": _NASA_CONFIRMED_QUERY,
    }
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            logger.info(
                "Querying NASA TAP for confirmed TESS planets (attempt %d/%d)…",
                attempt + 1, max_retries,
            )
            response = requests.get(_NASA_TAP_URL, params=params, timeout=120)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df.columns = [c.strip().lower() for c in df.columns]
            tic_col = _find_column(df, ["tic_id", "tic id", "tic"])
            confirmed = frozenset(int(x) for x in df[tic_col].dropna())
            logger.info("Found %d confirmed/known TIC IDs via NASA TAP.", len(confirmed))
            return confirmed
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Attempt %d failed: %s. Retrying in %ds…", attempt + 1, exc, wait
            )
            if attempt < max_retries - 1:
                time.sleep(wait)

    logger.warning(
        "Could not fetch confirmed TIC IDs after %d attempts (%s). "
        "Skipping cross-match — all unvetted candidates will be retained.",
        max_retries, last_exc,
    )
    return frozenset()


# ---------------------------------------------------------------------------
# Filtering and cross-matching
# ---------------------------------------------------------------------------

def filter_unvetted(df: pd.DataFrame) -> pd.DataFrame:
    """Return only PC/APC rows that have valid period and epoch values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw TOI catalog DataFrame. Not mutated.

    Returns
    -------
    pd.DataFrame
        Filtered copy: only unvetted candidates with usable transit parameters.
    """
    disp_col   = _find_column(df, ["TFOPWG Disp", "tfopwg_disp", "disp", "Disp"])
    period_col = _find_column(df, ["Period (days)", "period_days", "Period (Days)", "Period"])
    epoch_col  = _find_column(df, ["Epoch (BJD)", "epoch_bjd", "Epoch (BJD-2457000)", "Epoch"])
    tic_col    = _find_column(df, ["TIC ID", "tic_id", "TIC"])

    disp_series = df[disp_col].fillna("").str.strip()
    unvetted_mask = disp_series.isin(_UNVETTED_DISPOSITIONS)

    valid_params = (
        df[period_col].notna() & (df[period_col] > 0)
        & df[epoch_col].notna()
        & df[tic_col].notna()
    )

    filtered = df.loc[unvetted_mask & valid_params].copy()
    return filtered.reset_index(drop=True)


def remove_confirmed(
    df: pd.DataFrame,
    confirmed_tic_ids: frozenset[int],
) -> pd.DataFrame:
    """Remove rows whose TIC IDs already appear in the confirmed-planet catalog.

    Parameters
    ----------
    df : pd.DataFrame
        Unvetted candidate DataFrame. Not mutated.
    confirmed_tic_ids : frozenset[int]
        TIC IDs returned by ``fetch_confirmed_tic_ids``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with confirmed planets removed.
    """
    if not confirmed_tic_ids:
        return df.copy()
    tic_col = _find_column(df, ["TIC ID", "tic_id", "TIC"])
    mask = ~df[tic_col].astype(int).isin(confirmed_tic_ids)
    return df.loc[mask].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def extract_toi_params(row: pd.Series) -> dict[str, Any]:
    """Extract transit parameters from a single TOI catalog row.

    Converts the ExoFOP epoch from BJD to BTJD (BJD − 2457000) to match the
    time system used by lightkurve for TESS light curves.

    Parameters
    ----------
    row : pd.Series
        A row from the TOI catalog DataFrame.

    Returns
    -------
    dict with keys:
        tic_id          int    — TIC identifier
        toi_id          float  — TOI number (e.g. 1234.01)
        period_days     float  — orbital period in days
        epoch_btjd      float  — transit epoch in BTJD (BJD − 2457000)
        duration_hours  float  — transit duration in hours (default 2.0 if missing)
    """
    def _get(keys: list[str]) -> Any:
        for k in keys:
            if k in row.index and pd.notna(row[k]):
                return row[k]
        return None

    epoch_raw = float(_get(["Epoch (BJD)", "epoch_bjd", "Epoch (BJD-2457000)", "Epoch"]) or 0.0)
    # ExoFOP epochs are in BJD; convert to BTJD for lightkurve compatibility.
    # If the value is already small (< 10000), it is likely already in BTJD.
    epoch_btjd = epoch_raw - BTJD_OFFSET if epoch_raw > BTJD_OFFSET else epoch_raw

    return {
        "tic_id":         int(_get(["TIC ID", "tic_id", "TIC"])),
        "toi_id":         float(_get(["TOI", "toi"]) or 0.0),
        "period_days":    float(_get(["Period (days)", "period_days", "Period"]) or 0.0),
        "epoch_btjd":     epoch_btjd,
        "duration_hours": float(_get(["Duration (hours)", "duration_hours", "Duration"]) or 2.0),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name from *candidates* found in *df*.

    Raises
    ------
    KeyError
        When none of the candidate column names are present.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of the expected columns {candidates} found in DataFrame. "
        f"Available columns: {list(df.columns)}"
    )


def _save_csv(df: pd.DataFrame, output_path: str) -> None:
    """Write *df* to CSV, creating parent directories as needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
