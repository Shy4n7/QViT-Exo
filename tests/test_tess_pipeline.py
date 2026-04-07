"""Tests for Phase 6 — TESS download and catalog utilities.

All tests that exercise network-dependent functions mock the HTTP layer so
they run offline.  Pure helper functions are tested directly without mocking.
"""

from __future__ import annotations

import io
import textwrap

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# tess_download helpers
# ---------------------------------------------------------------------------

class TestTicTarget:
    def test_integer_input(self):
        from src.data.tess_download import _tic_target
        assert _tic_target(123456789) == "TIC 123456789"

    def test_string_integer(self):
        from src.data.tess_download import _tic_target
        assert _tic_target("123456789") == "TIC 123456789"

    def test_tic_prefix_uppercase(self):
        from src.data.tess_download import _tic_target
        assert _tic_target("TIC 123456789") == "TIC 123456789"

    def test_tic_prefix_lowercase(self):
        from src.data.tess_download import _tic_target
        assert _tic_target("tic 123456789") == "TIC 123456789"

    def test_tic_prefix_no_space(self):
        from src.data.tess_download import _tic_target
        assert _tic_target("TIC123456789") == "TIC 123456789"

    def test_strips_whitespace(self):
        from src.data.tess_download import _tic_target
        assert _tic_target("  TIC 42  ") == "TIC 42"


class TestCacheKey:
    def test_no_sectors_returns_all(self):
        from src.data.tess_download import _cache_key
        assert _cache_key(12345, None) == "TIC_12345_all"

    def test_single_sector(self):
        from src.data.tess_download import _cache_key
        assert _cache_key(12345, [70]) == "TIC_12345_s70_70"

    def test_sector_range(self):
        from src.data.tess_download import _cache_key
        assert _cache_key(12345, [70, 75, 72]) == "TIC_12345_s70_75"

    def test_tic_string_with_prefix(self):
        from src.data.tess_download import _cache_key
        assert _cache_key("TIC 12345", None) == "TIC_12345_all"

    def test_different_tic_ids_different_keys(self):
        from src.data.tess_download import _cache_key
        assert _cache_key(1, None) != _cache_key(2, None)

    def test_different_sectors_different_keys(self):
        from src.data.tess_download import _cache_key
        assert _cache_key(1, [70]) != _cache_key(1, [80])


# ---------------------------------------------------------------------------
# tess_catalog._find_column
# ---------------------------------------------------------------------------

class TestFindColumn:
    def test_exact_match(self):
        from src.data.tess_catalog import _find_column
        df = pd.DataFrame({"TFOPWG Disp": [], "Period (days)": []})
        assert _find_column(df, ["TFOPWG Disp", "disp"]) == "TFOPWG Disp"

    def test_fallback_match(self):
        from src.data.tess_catalog import _find_column
        df = pd.DataFrame({"disp": []})
        assert _find_column(df, ["TFOPWG Disp", "disp"]) == "disp"

    def test_raises_when_missing(self):
        from src.data.tess_catalog import _find_column
        df = pd.DataFrame({"other": []})
        with pytest.raises(KeyError):
            _find_column(df, ["TFOPWG Disp", "disp"])


# ---------------------------------------------------------------------------
# tess_catalog.filter_unvetted
# ---------------------------------------------------------------------------

def _make_toi_df() -> pd.DataFrame:
    """Minimal TOI-like DataFrame for testing."""
    return pd.DataFrame({
        "TIC ID":        [111, 222, 333, 444, 555, 666],
        "TOI":           [1.01, 2.01, 3.01, 4.01, 5.01, 6.01],
        "TFOPWG Disp":   ["PC", "CP", "FP", "FA", "APC", "PC"],
        "Period (days)": [3.0,  5.0,  2.0,  1.0,  4.0,  np.nan],
        "Epoch (BJD)":   [2458000, 2458001, 2458002, 2458003, 2458004, 2458005],
    })


class TestFilterUnvetted:
    def test_returns_only_pc_apc(self):
        from src.data.tess_catalog import filter_unvetted
        df = _make_toi_df()
        result = filter_unvetted(df)
        # TIC 111 (PC, valid period), TIC 555 (APC, valid period) remain;
        # TIC 666 (PC) dropped due to NaN period; CP/FP/FA dropped.
        assert set(result["TIC ID"].tolist()) == {111, 555}

    def test_drops_nan_period(self):
        from src.data.tess_catalog import filter_unvetted
        df = _make_toi_df()
        result = filter_unvetted(df)
        assert 666 not in result["TIC ID"].values

    def test_drops_confirmed(self):
        from src.data.tess_catalog import filter_unvetted
        df = _make_toi_df()
        result = filter_unvetted(df)
        assert 222 not in result["TIC ID"].values

    def test_drops_false_positive(self):
        from src.data.tess_catalog import filter_unvetted
        df = _make_toi_df()
        result = filter_unvetted(df)
        assert 333 not in result["TIC ID"].values

    def test_output_is_copy(self):
        from src.data.tess_catalog import filter_unvetted
        df = _make_toi_df()
        result = filter_unvetted(df)
        result.iloc[0, 0] = 9999
        assert df.iloc[0, 0] != 9999  # original not mutated

    def test_reset_index(self):
        from src.data.tess_catalog import filter_unvetted
        df = _make_toi_df()
        result = filter_unvetted(df)
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# tess_catalog.remove_confirmed
# ---------------------------------------------------------------------------

class TestRemoveConfirmed:
    def test_removes_confirmed_ids(self):
        from src.data.tess_catalog import remove_confirmed
        df = pd.DataFrame({"TIC ID": [100, 200, 300], "x": [1, 2, 3]})
        result = remove_confirmed(df, frozenset({200}))
        assert 200 not in result["TIC ID"].values
        assert 100 in result["TIC ID"].values

    def test_empty_confirmed_returns_all(self):
        from src.data.tess_catalog import remove_confirmed
        df = pd.DataFrame({"TIC ID": [100, 200], "x": [1, 2]})
        result = remove_confirmed(df, frozenset())
        assert len(result) == 2

    def test_all_confirmed_returns_empty(self):
        from src.data.tess_catalog import remove_confirmed
        df = pd.DataFrame({"TIC ID": [100, 200], "x": [1, 2]})
        result = remove_confirmed(df, frozenset({100, 200}))
        assert len(result) == 0

    def test_output_is_copy(self):
        from src.data.tess_catalog import remove_confirmed
        df = pd.DataFrame({"TIC ID": [100, 200], "x": [1, 2]})
        result = remove_confirmed(df, frozenset({200}))
        result.iloc[0, 1] = 999
        assert df.iloc[0, 1] == 1  # original not mutated


# ---------------------------------------------------------------------------
# tess_catalog.extract_toi_params
# ---------------------------------------------------------------------------

class TestExtractToiParams:
    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "TIC ID":        123456,
            "TOI":           789.01,
            "Period (days)": 3.14,
            "Epoch (BJD)":   2458500.0,
            "Duration (hours)": 2.5,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_basic_extraction(self):
        from src.data.tess_catalog import extract_toi_params
        row = self._make_row()
        p = extract_toi_params(row)
        assert p["tic_id"] == 123456
        assert abs(p["toi_id"] - 789.01) < 1e-6
        assert abs(p["period_days"] - 3.14) < 1e-6

    def test_epoch_converted_to_btjd(self):
        """BJD > BTJD_OFFSET → subtract offset."""
        from src.data.tess_catalog import extract_toi_params, BTJD_OFFSET
        row = self._make_row(**{"Epoch (BJD)": 2458500.0})
        p = extract_toi_params(row)
        expected = 2458500.0 - BTJD_OFFSET
        assert abs(p["epoch_btjd"] - expected) < 1e-6

    def test_epoch_already_btjd_not_double_subtracted(self):
        """Epoch < BTJD_OFFSET already — should NOT subtract again."""
        from src.data.tess_catalog import extract_toi_params
        row = self._make_row(**{"Epoch (BJD)": 1500.0})
        p = extract_toi_params(row)
        assert abs(p["epoch_btjd"] - 1500.0) < 1e-6

    def test_default_duration_when_missing(self):
        from src.data.tess_catalog import extract_toi_params
        row = self._make_row()
        row = row.drop("Duration (hours)")
        p = extract_toi_params(row)
        assert p["duration_hours"] == 2.0

    def test_tic_id_is_int(self):
        from src.data.tess_catalog import extract_toi_params
        row = self._make_row()
        p = extract_toi_params(row)
        assert isinstance(p["tic_id"], int)


# ---------------------------------------------------------------------------
# run_tess_search._attention_coherence
# ---------------------------------------------------------------------------

class TestAttentionCoherence:
    def test_uniform_attention_low_coherence(self):
        """Uniform attention → coherence = indicator_fraction (small)."""
        import importlib
        mod = importlib.import_module("scripts.run_tess_search")
        profile = np.ones(14) / 14.0
        score = mod._attention_coherence(profile, duration_fraction=0.1)
        assert 0.0 <= score <= 1.0

    def test_perfect_coherence(self):
        """All attention on ingress/egress → score near 1."""
        import importlib
        from src.interpretability.attention_analysis import ingress_egress_indicator
        mod = importlib.import_module("scripts.run_tess_search")
        ind = ingress_egress_indicator(14, t_mid=0.5, duration=0.3)
        # Put all weight on the indicator
        profile = ind / (ind.sum() + 1e-12)
        score = mod._attention_coherence(profile, duration_fraction=0.3)
        assert score > 0.8

    def test_zero_profile_returns_zero(self):
        import importlib
        mod = importlib.import_module("scripts.run_tess_search")
        profile = np.zeros(14)
        score = mod._attention_coherence(profile, duration_fraction=0.1)
        assert score == 0.0

    def test_score_in_unit_interval(self):
        import importlib
        mod = importlib.import_module("scripts.run_tess_search")
        rng = np.random.default_rng(0)
        for _ in range(10):
            p = np.abs(rng.standard_normal(14))
            p /= p.sum()
            score = mod._attention_coherence(p, duration_fraction=0.15)
            assert 0.0 <= score <= 1.0
