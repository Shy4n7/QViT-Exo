"""Tests for src/data/catalog.py.

TDD order: tests written BEFORE implementation.
All unit tests mock network calls — no real HTTP in CI.
The @pytest.mark.slow test is excluded via -k "not slow".
"""

from __future__ import annotations

import os
import textwrap
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.catalog import create_splits, fetch_koi_catalog, filter_dispositions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "kepoi_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk",
    "koi_depth",
    "koi_duration",
    "koi_ror",
    "ra",
    "dec",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal catalog DataFrame from a list of row dicts."""
    base = {col: None for col in EXPECTED_COLUMNS}
    records = [{**base, **row} for row in rows]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# filter_dispositions tests
# ---------------------------------------------------------------------------


class TestFilterDispositions:
    def test_filter_dispositions_keeps_confirmed_and_fp(self) -> None:
        """Rows with CONFIRMED and FALSE POSITIVE survive; CANDIDATE is dropped."""
        df = _make_catalog(
            [
                {"kepoi_name": "K001", "koi_disposition": "CONFIRMED"},
                {"kepoi_name": "K002", "koi_disposition": "FALSE POSITIVE"},
                {"kepoi_name": "K003", "koi_disposition": "CANDIDATE"},
            ]
        )

        result = filter_dispositions(df)

        assert len(result) == 2
        assert set(result["koi_disposition"].unique()) == {"CONFIRMED", "FALSE POSITIVE"}
        assert "K003" not in result["kepoi_name"].values

    def test_filter_dispositions_output_columns(self, mock_koi_catalog: pd.DataFrame) -> None:
        """Output DataFrame contains at least the expected pipeline columns."""
        result = filter_dispositions(mock_koi_catalog)

        for col in EXPECTED_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_filter_dispositions_empty_input(self) -> None:
        """Empty DataFrame returns empty DataFrame without error."""
        empty = pd.DataFrame(columns=EXPECTED_COLUMNS)

        result = filter_dispositions(empty)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == EXPECTED_COLUMNS

    def test_filter_dispositions_does_not_mutate_input(self) -> None:
        """filter_dispositions must be pure — original DataFrame unchanged."""
        df = _make_catalog(
            [
                {"kepoi_name": "K001", "koi_disposition": "CONFIRMED"},
                {"kepoi_name": "K002", "koi_disposition": "CANDIDATE"},
            ]
        )
        original_len = len(df)
        original_dispositions = df["koi_disposition"].tolist()

        filter_dispositions(df)

        assert len(df) == original_len
        assert df["koi_disposition"].tolist() == original_dispositions

    def test_filter_dispositions_all_candidate_returns_empty(self) -> None:
        """All CANDIDATE rows → empty result."""
        df = _make_catalog(
            [
                {"kepoi_name": "K001", "koi_disposition": "CANDIDATE"},
                {"kepoi_name": "K002", "koi_disposition": "CANDIDATE"},
            ]
        )

        result = filter_dispositions(df)

        assert len(result) == 0

    def test_filter_dispositions_preserves_row_values(self) -> None:
        """Surviving rows keep their original field values intact."""
        df = _make_catalog(
            [
                {
                    "kepoi_name": "K007",
                    "koi_disposition": "CONFIRMED",
                    "koi_period": 3.14,
                    "ra": 291.5,
                },
                {"kepoi_name": "K008", "koi_disposition": "CANDIDATE"},
            ]
        )

        result = filter_dispositions(df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["kepoi_name"] == "K007"
        assert row["koi_period"] == pytest.approx(3.14)
        assert row["ra"] == pytest.approx(291.5)


# ---------------------------------------------------------------------------
# create_splits tests
# ---------------------------------------------------------------------------


class TestCreateSplits:
    _config = {
        "test_size": 0.15,
        "val_size": 0.15,
        "random_seed": 42,
    }

    def test_create_splits_returns_three_dataframes(
        self, mock_koi_catalog: pd.DataFrame
    ) -> None:
        """create_splits returns a 3-tuple of DataFrames."""
        result = create_splits(mock_koi_catalog, self._config)

        assert isinstance(result, tuple)
        assert len(result) == 3
        for part in result:
            assert isinstance(part, pd.DataFrame)

    def test_create_splits_total_row_count(self, mock_koi_catalog: pd.DataFrame) -> None:
        """Sum of all split rows equals the input row count."""
        train_df, val_df, test_df = create_splits(mock_koi_catalog, self._config)

        assert len(train_df) + len(val_df) + len(test_df) == len(mock_koi_catalog)

    def test_create_splits_no_overlap(self, mock_koi_catalog: pd.DataFrame) -> None:
        """No kepoi_name appears in more than one split."""
        train_df, val_df, test_df = create_splits(mock_koi_catalog, self._config)

        train_ids = set(train_df["kepoi_name"])
        val_ids = set(val_df["kepoi_name"])
        test_ids = set(test_df["kepoi_name"])

        assert train_ids.isdisjoint(val_ids), "Overlap between train and val"
        assert train_ids.isdisjoint(test_ids), "Overlap between train and test"
        assert val_ids.isdisjoint(test_ids), "Overlap between val and test"

    def test_create_splits_stratified_class_ratio(
        self, mock_koi_catalog: pd.DataFrame
    ) -> None:
        """Each split preserves the original CONFIRMED / FALSE POSITIVE ratio (±20%)."""
        original_ratio = (
            mock_koi_catalog["koi_disposition"].value_counts(normalize=True)
        )
        train_df, val_df, test_df = create_splits(mock_koi_catalog, self._config)

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(split_df) == 0:
                continue
            split_ratio = split_df["koi_disposition"].value_counts(normalize=True)
            for label in original_ratio.index:
                if label in split_ratio.index:
                    diff = abs(split_ratio[label] - original_ratio[label])
                    assert diff <= 0.20, (
                        f"{split_name} split ratio for {label} deviates by {diff:.2f}"
                    )

    def test_create_splits_deterministic(self, mock_koi_catalog: pd.DataFrame) -> None:
        """Same config produces identical splits on repeated calls."""
        result_a = create_splits(mock_koi_catalog, self._config)
        result_b = create_splits(mock_koi_catalog, self._config)

        for df_a, df_b in zip(result_a, result_b):
            pd.testing.assert_frame_equal(
                df_a.reset_index(drop=True), df_b.reset_index(drop=True)
            )

    def test_create_splits_does_not_mutate_input(
        self, mock_koi_catalog: pd.DataFrame
    ) -> None:
        """create_splits must not modify the input DataFrame."""
        original_len = len(mock_koi_catalog)
        original_cols = list(mock_koi_catalog.columns)

        create_splits(mock_koi_catalog, self._config)

        assert len(mock_koi_catalog) == original_len
        assert list(mock_koi_catalog.columns) == original_cols


# ---------------------------------------------------------------------------
# fetch_koi_catalog tests (mocked network)
# ---------------------------------------------------------------------------


class TestFetchKoiCatalogMocked:
    """Unit tests for fetch_koi_catalog using mocked HTTP responses."""

    _config = {
        "catalog_url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        "catalog_table": "cumulative",
        "catalog_columns": EXPECTED_COLUMNS,
        "download_retries": 3,
    }

    def _make_csv_response(self, n_confirmed: int = 3, n_fp: int = 3) -> str:
        """Build a minimal CSV string that mimics the TAP service response."""
        rows = []
        for i in range(n_confirmed):
            rows.append(
                f"K{str(i).zfill(8)}.01,CONFIRMED,"
                f"{1.0 + i},{100.0 + i},{500.0},{5.0},{0.1},{290.0},{45.0}"
            )
        for i in range(n_fp):
            rows.append(
                f"K{str(100 + i).zfill(8)}.01,FALSE POSITIVE,"
                f"{2.0 + i},{110.0 + i},{300.0},{4.0},{0.08},{291.0},{46.0}"
            )
        header = ",".join(EXPECTED_COLUMNS)
        return header + "\n" + "\n".join(rows) + "\n"

    @patch("src.data.catalog.requests.get")
    def test_fetch_saves_csv_to_output_path(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """fetch_koi_catalog writes a CSV at the specified output_path."""
        csv_text = self._make_csv_response()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = csv_text
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        output_path = str(tmp_path / "catalog.csv")
        fetch_koi_catalog(output_path, self._config)

        assert Path(output_path).exists(), "CSV file was not created"

    @patch("src.data.catalog.requests.get")
    def test_fetch_returns_dataframe(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """fetch_koi_catalog returns a pandas DataFrame."""
        csv_text = self._make_csv_response()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = csv_text
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        output_path = str(tmp_path / "catalog.csv")
        result = fetch_koi_catalog(output_path, self._config)

        assert isinstance(result, pd.DataFrame)

    @patch("src.data.catalog.requests.get")
    def test_fetch_returned_dataframe_has_expected_columns(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """Returned DataFrame contains all expected columns."""
        csv_text = self._make_csv_response()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = csv_text
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        output_path = str(tmp_path / "catalog.csv")
        result = fetch_koi_catalog(output_path, self._config)

        for col in EXPECTED_COLUMNS:
            assert col in result.columns

    @patch("src.data.catalog.requests.get")
    def test_fetch_retries_on_failure_then_succeeds(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """fetch_koi_catalog retries after transient HTTP errors and succeeds."""
        import requests as req

        csv_text = self._make_csv_response()
        good_response = MagicMock()
        good_response.status_code = 200
        good_response.text = csv_text
        good_response.raise_for_status = MagicMock()

        # First call raises, second succeeds
        mock_get.side_effect = [req.RequestException("timeout"), good_response]

        output_path = str(tmp_path / "catalog.csv")
        result = fetch_koi_catalog(output_path, self._config)

        assert isinstance(result, pd.DataFrame)
        assert mock_get.call_count == 2

    @patch("src.data.catalog.requests.get")
    def test_fetch_raises_after_all_retries_exhausted(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """fetch_koi_catalog raises RuntimeError when all retries fail."""
        import requests as req

        mock_get.side_effect = req.RequestException("timeout")

        output_path = str(tmp_path / "catalog.csv")
        with pytest.raises(RuntimeError, match="Failed to download"):
            fetch_koi_catalog(output_path, self._config)

    @patch("src.data.catalog.requests.get")
    def test_fetch_creates_parent_directories(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """fetch_koi_catalog creates missing parent directories for output_path."""
        csv_text = self._make_csv_response()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = csv_text
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        output_path = str(tmp_path / "nested" / "deep" / "catalog.csv")
        fetch_koi_catalog(output_path, self._config)

        assert Path(output_path).exists()


# ---------------------------------------------------------------------------
# fetch_koi_catalog — real network (slow, excluded from CI)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fetch_koi_catalog_saves_csv(tmp_path: Path) -> None:
    """Real TAP network call: saves CSV with >5000 rows.

    Skipped in CI via: pytest -k "not slow"
    """
    import yaml

    config_path = Path(__file__).parent.parent / "configs" / "data_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_path = str(tmp_path / "koi_catalog.csv")
    result = fetch_koi_catalog(output_path, config)

    assert Path(output_path).exists(), "CSV not saved"
    assert len(result) > 5000, f"Expected >5000 rows, got {len(result)}"
    assert "koi_disposition" in result.columns
