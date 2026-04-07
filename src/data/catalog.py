"""KOI DR25 catalog download, filtering, and train/val/test splitting.

All functions are pure (return new objects, never mutate inputs).
Network calls use exponential backoff with configurable retry count.
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Dispositions treated as labelled examples for binary classification
_KEEP_DISPOSITIONS = frozenset({"CONFIRMED", "FALSE POSITIVE"})

# ADQL query template for the NASA Exoplanet Archive TAP service
_TAP_QUERY_TEMPLATE = (
    "SELECT {columns} FROM {table} "
    "WHERE koi_disposition IN ('CONFIRMED','FALSE POSITIVE')"
)


def fetch_koi_catalog(output_path: str, config: dict[str, Any]) -> pd.DataFrame:
    """Download KOI DR25 catalog from the NASA Exoplanet Archive TAP service.

    Parameters
    ----------
    output_path:
        Filesystem path where the CSV will be written.
        Parent directories are created if they do not exist.
    config:
        Must contain keys:
            catalog_url       – TAP service base URL
            catalog_table     – Table name (e.g. "cumulative")
            catalog_columns   – List of column names to SELECT
            download_retries  – Maximum number of attempts (default 3)

    Returns
    -------
    pd.DataFrame
        DataFrame with the downloaded catalog rows.

    Raises
    ------
    RuntimeError
        When all retry attempts are exhausted.
    """
    url: str = config["catalog_url"]
    table: str = config["catalog_table"]
    columns: list[str] = config["catalog_columns"]
    max_retries: int = config.get("download_retries", 3)

    adql = _TAP_QUERY_TEMPLATE.format(
        columns=",".join(columns),
        table=table,
    )
    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": adql,
    }

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            logger.info("TAP request attempt %d/%d …", attempt + 1, max_retries)
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            _save_csv(df, output_path)
            logger.info("Catalog saved to %s (%d rows)", output_path, len(df))
            return df
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 2**attempt  # exponential backoff: 1 s, 2 s, 4 s, …
            logger.warning(
                "Attempt %d failed: %s. Retrying in %ds …", attempt + 1, exc, wait
            )
            if attempt < max_retries - 1:
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to download KOI catalog after {max_retries} attempts. "
        f"Last error: {last_exc}"
    )


def filter_dispositions(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame containing only CONFIRMED and FALSE POSITIVE rows.

    Parameters
    ----------
    df:
        Raw KOI catalog DataFrame. Not mutated.

    Returns
    -------
    pd.DataFrame
        Filtered copy with the same columns as the input.
    """
    mask = df["koi_disposition"].isin(_KEEP_DISPOSITIONS)
    return df.loc[mask].copy()


def create_splits(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split preserving class ratio.

    Parameters
    ----------
    df:
        Filtered catalog (only CONFIRMED and FALSE POSITIVE). Not mutated.
    config:
        Must contain:
            test_size    – Fraction of total data for the test set (e.g. 0.15)
            val_size     – Fraction of total data for the val set (e.g. 0.15)
            random_seed  – Integer seed for reproducibility

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    test_size: float = config["test_size"]
    val_size: float = config["val_size"]
    seed: int = config["random_seed"]
    stratify_col = "koi_disposition"

    # First split: carve off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[stratify_col],
    )

    # Second split: carve val from the remaining train+val pool
    # val_size is a fraction of the *original* dataset; adjust relative to train_val size
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_df[stratify_col],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_csv(df: pd.DataFrame, output_path: str) -> None:
    """Write DataFrame to CSV, creating parent directories as needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
