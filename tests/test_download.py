"""Tests for src/data/download.py — download_lightcurve and batch_download.

TDD: All tests written BEFORE implementation.

Mock strategy:
    - lightkurve.search_lightcurve is patched for ALL tests; no real
      network calls are made.
    - A fake LightCurve-like object (FakeLightCurve) is returned from
      mocked searches to satisfy duck-typing checks.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Fake lightkurve objects
# ---------------------------------------------------------------------------

class FakeLightCurve:
    """Minimal stand-in for a lightkurve.LightCurve object."""

    def __init__(self, koi_id: str) -> None:
        self.koi_id = koi_id
        self.label = koi_id

    def to_fits(self, path: str, overwrite: bool = False) -> None:
        """Write a placeholder file so cache existence checks pass."""
        from pathlib import Path
        Path(path).write_bytes(b"FAKE_FITS")

    def __repr__(self) -> str:  # pragma: no cover
        return f"FakeLightCurve(koi_id={self.koi_id!r})"


class FakeSearchResult:
    """Minimal stand-in for lightkurve.SearchResult."""

    def __init__(self, koi_id: str, empty: bool = False) -> None:
        self._koi_id = koi_id
        self._empty = empty

    def __len__(self) -> int:
        return 0 if self._empty else 1

    def download(self) -> FakeLightCurve:
        if self._empty:
            raise ValueError("No results to download")
        return FakeLightCurve(self._koi_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDownloadLightcurveReturnsLc:
    """test_download_lightcurve_returns_lc: function returns a LightCurve-like object."""

    def test_download_lightcurve_returns_lc(self, tmp_path: Path) -> None:
        """download_lightcurve returns a LightCurve-like object for a known KOI."""
        from src.data.download import download_lightcurve

        koi_id = "K00001.01"
        fake_result = FakeSearchResult(koi_id)

        with patch("src.data.download.lk.search_lightcurve", return_value=fake_result):
            lc = download_lightcurve(koi_id=koi_id, cache_dir=str(tmp_path))

        assert lc is not None, "Expected a LightCurve object, got None"
        # Duck-type check: must have a .label attribute (like a real LightCurve)
        assert hasattr(lc, "label") or isinstance(lc, FakeLightCurve), (
            f"Return value must be LightCurve-like, got {type(lc)}"
        )


class TestDownloadLightcurveCachesLocally:
    """test_download_lightcurve_caches_locally: second call skips search."""

    def test_download_lightcurve_caches_locally(self, tmp_path: Path) -> None:
        """Second call with same koi_id must use cache — search not called again."""
        from src.data.download import download_lightcurve

        koi_id = "K00002.01"
        fake_result = FakeSearchResult(koi_id)
        search_mock = MagicMock(return_value=fake_result)

        read_mock = MagicMock(return_value=FakeLightCurve(koi_id))
        with patch("src.data.download.lk.search_lightcurve", search_mock), \
             patch("src.data.download.lk.read", read_mock):
            # First call — should search, download, and cache as FITS
            lc1 = download_lightcurve(koi_id=koi_id, cache_dir=str(tmp_path))
            # Second call — should read from FITS cache (lk.read), skip search
            lc2 = download_lightcurve(koi_id=koi_id, cache_dir=str(tmp_path))

        # search_lightcurve must have been called exactly ONCE
        assert search_mock.call_count == 1, (
            f"Expected search to be called once (cache hit on 2nd call), "
            f"but it was called {search_mock.call_count} times"
        )
        assert lc1 is not None
        assert lc2 is not None


class TestDownloadMissingKoiLogsWarning:
    """test_download_missing_koi_logs_warning: no results → WARNING + None return."""

    def test_download_missing_koi_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """KOI with no search results must log a WARNING and return None."""
        from src.data.download import download_lightcurve

        koi_id = "K99999.01"
        empty_result = FakeSearchResult(koi_id, empty=True)

        with patch(
            "src.data.download.lk.search_lightcurve", return_value=empty_result
        ):
            with caplog.at_level(logging.WARNING, logger="src.data.download"):
                result = download_lightcurve(koi_id=koi_id, cache_dir=str(tmp_path))

        # Must return None, not raise
        assert result is None, (
            f"Expected None for missing KOI, got {result!r}"
        )
        # Must log a WARNING
        warning_msgs = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_msgs) >= 1, (
            f"Expected at least 1 WARNING log for missing KOI, got: {warning_msgs}"
        )
        assert any(koi_id in str(m) for m in warning_msgs), (
            f"WARNING must mention the koi_id '{koi_id}', got: {warning_msgs}"
        )


class TestBatchDownloadSkipsFailures:
    """test_batch_download_skips_failures: one failing KOI doesn't stop others."""

    def test_batch_download_skips_failures(self, tmp_path: Path) -> None:
        """One failing KOI in a batch must not prevent others from succeeding."""
        from src.data.download import batch_download

        good_ids = ["K00010.01", "K00011.01"]
        bad_id = "K00099.01"
        all_ids = good_ids + [bad_id]

        def _fake_search(target: str, *args, **kwargs) -> FakeSearchResult:
            if bad_id in target:
                return FakeSearchResult(bad_id, empty=True)
            return FakeSearchResult(target)

        with patch("src.data.download.lk.search_lightcurve", side_effect=_fake_search):
            results, failed = batch_download(
                koi_ids=all_ids,
                cache_dir=str(tmp_path),
                n_workers=1,
            )

        # Both good KOIs must succeed
        for gid in good_ids:
            assert results.get(gid) is not None, (
                f"Expected successful download for {gid}"
            )
        # bad_id must appear in failed list
        assert bad_id in failed, (
            f"Expected {bad_id} in failed list, got failed={failed}"
        )


class TestDownloadRaisesOnDownloadFailure:
    """download_lightcurve returns None when search_result.download() raises."""

    def test_download_failure_on_download_call(self, tmp_path: Path) -> None:
        """When search_result.download() raises, function returns None and logs WARNING."""
        from src.data.download import download_lightcurve

        koi_id = "K00030.01"

        class _FailingSearchResult:
            def __len__(self) -> int:
                return 1

            def download(self) -> None:
                raise ConnectionError("Simulated download failure")

        with patch(
            "src.data.download.lk.search_lightcurve",
            return_value=_FailingSearchResult(),
        ):
            result = download_lightcurve(koi_id=koi_id, cache_dir=str(tmp_path))

        assert result is None, "Expected None when download() raises"

    def test_cache_write_failure_is_non_fatal(self, tmp_path: Path) -> None:
        """When _save_to_cache raises, the light curve is still returned."""
        from src.data.download import download_lightcurve

        koi_id = "K00031.01"
        fake_result = FakeSearchResult(koi_id)

        with patch("src.data.download.lk.search_lightcurve", return_value=fake_result):
            with patch(
                "src.data.download._save_to_cache",
                side_effect=OSError("disk full"),
            ):
                result = download_lightcurve(koi_id=koi_id, cache_dir=str(tmp_path))

        # LightCurve must still be returned even though cache write failed
        assert result is not None, (
            "Expected LightCurve even when cache write fails"
        )


class TestBatchDownloadFutureException:
    """Covers the executor future.result() exception branch in batch_download."""

    def test_batch_download_future_exception(self, tmp_path: Path) -> None:
        """When a future raises, that KOI goes to failed and others succeed."""
        from src.data.download import batch_download
        from concurrent.futures import Future

        good_id = "K00040.01"
        explode_id = "K00041.01"

        original_submit = None

        def _patched_worker(kid: str) -> tuple[str, object]:
            if kid == explode_id:
                raise RuntimeError("future exploded")
            return kid, FakeLightCurve(kid)

        # We patch download_lightcurve so the worker raises for explode_id
        call_count = {"n": 0}

        def _fake_dl(koi_id: str, cache_dir: str, timeout: int = 60) -> object:
            if koi_id == explode_id:
                raise RuntimeError("forced future error")
            return FakeLightCurve(koi_id)

        with patch("src.data.download.download_lightcurve", side_effect=_fake_dl):
            results, failed = batch_download(
                koi_ids=[good_id, explode_id],
                cache_dir=str(tmp_path),
                n_workers=1,
            )

        assert results[good_id] is not None, "good_id must succeed"
        assert results[explode_id] is None, "explode_id must be None"
        assert explode_id in failed, "explode_id must be in failed"


class TestBatchDownloadReturnsDict:
    """test_batch_download_returns_dict: returns dict mapping koi_id -> LightCurve."""

    def test_batch_download_returns_dict(self, tmp_path: Path) -> None:
        """batch_download must return (dict[koi_id -> LC|None], list[failed])."""
        from src.data.download import batch_download

        koi_ids = ["K00020.01", "K00021.01", "K00022.01"]
        good_id = "K00020.01"
        fail_id = "K00021.01"
        missing_id = "K00022.01"

        def _fake_search(target: str, *args, **kwargs) -> FakeSearchResult:
            if fail_id in target:
                # Simulate a runtime error (e.g. network timeout)
                raise RuntimeError("Simulated network error")
            if missing_id in target:
                return FakeSearchResult(missing_id, empty=True)
            return FakeSearchResult(target)

        with patch("src.data.download.lk.search_lightcurve", side_effect=_fake_search):
            results, failed = batch_download(
                koi_ids=koi_ids,
                cache_dir=str(tmp_path),
                n_workers=1,
            )

        # Return type checks
        assert isinstance(results, dict), (
            f"First return value must be dict, got {type(results)}"
        )
        assert isinstance(failed, list), (
            f"Second return value must be list, got {type(failed)}"
        )
        # All requested koi_ids must be keys in results (None for failures/missing)
        for kid in koi_ids:
            assert kid in results, f"koi_id {kid!r} missing from results dict"

        # good_id → LightCurve-like
        assert results[good_id] is not None, (
            f"Expected non-None result for successful KOI {good_id}"
        )
        # fail_id → RuntimeError was raised; must be in failed and None in results
        assert fail_id in failed, (
            f"Expected {fail_id} in failed list"
        )
        assert results[fail_id] is None, (
            f"Expected None in results dict for failed KOI {fail_id}"
        )
        # missing_id → no results from search; None in results
        assert results[missing_id] is None, (
            f"Expected None in results dict for missing KOI {missing_id}"
        )
