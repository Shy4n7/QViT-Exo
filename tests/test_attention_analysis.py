"""Tests for Phase 5 — interpretability / attention analysis."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# attention_to_lightcurve_profile
# ---------------------------------------------------------------------------

class TestAttentionToLightcurve:
    def test_output_shape(self):
        from src.interpretability.attention_analysis import attention_to_lightcurve_profile
        a = np.random.rand(14, 14)
        p = attention_to_lightcurve_profile(a)
        assert p.shape == (14,)

    def test_sums_to_one(self):
        from src.interpretability.attention_analysis import attention_to_lightcurve_profile
        a = np.random.rand(14, 14)
        p = attention_to_lightcurve_profile(a)
        assert abs(p.sum() - 1.0) < 1e-6

    def test_non_negative(self):
        from src.interpretability.attention_analysis import attention_to_lightcurve_profile
        a = np.abs(np.random.rand(14, 14))
        p = attention_to_lightcurve_profile(a)
        assert (p >= 0).all()

    def test_zero_map_returns_zeros(self):
        from src.interpretability.attention_analysis import attention_to_lightcurve_profile
        p = attention_to_lightcurve_profile(np.zeros((14, 14)))
        assert (p == 0).all()


# ---------------------------------------------------------------------------
# ingress_egress_indicator
# ---------------------------------------------------------------------------

class TestIngressEgressIndicator:
    def test_output_shape(self):
        from src.interpretability.attention_analysis import ingress_egress_indicator
        ind = ingress_egress_indicator(100, t_mid=0.5, duration=0.2)
        assert ind.shape == (100,)

    def test_values_binary(self):
        from src.interpretability.attention_analysis import ingress_egress_indicator
        ind = ingress_egress_indicator(100, t_mid=0.5, duration=0.2)
        assert set(np.unique(ind)).issubset({0.0, 1.0})

    def test_nonzero_around_transit(self):
        from src.interpretability.attention_analysis import ingress_egress_indicator
        ind = ingress_egress_indicator(200, t_mid=0.5, duration=0.3)
        assert ind.sum() > 0

    def test_zero_outside_transit(self):
        from src.interpretability.attention_analysis import ingress_egress_indicator
        ind = ingress_egress_indicator(200, t_mid=0.5, duration=0.1)
        # Flat region in mid-transit should be 0
        mid_start = int(200 * 0.5) - 5
        mid_end   = int(200 * 0.5) + 5
        assert ind[mid_start:mid_end].sum() == 0


# ---------------------------------------------------------------------------
# correlate_with_transit
# ---------------------------------------------------------------------------

class TestCorrelateWithTransit:
    def test_returns_dataclass(self):
        from src.interpretability.attention_analysis import (
            correlate_with_transit, ingress_egress_indicator,
        )
        prof = np.random.rand(14)
        prof /= prof.sum()
        ind  = ingress_egress_indicator(14, t_mid=0.5, duration=0.2)
        r    = correlate_with_transit(prof, ind, n_permutations=50)
        assert hasattr(r, "spearman_rho")
        assert hasattr(r, "p_value")
        assert hasattr(r, "permutation_p")

    def test_rho_in_range(self):
        from src.interpretability.attention_analysis import (
            correlate_with_transit, ingress_egress_indicator,
        )
        prof = np.random.rand(14)
        ind  = ingress_egress_indicator(14, t_mid=0.5, duration=0.2)
        r    = correlate_with_transit(prof, ind, n_permutations=50)
        assert -1.0 <= r.spearman_rho <= 1.0

    def test_perfect_correlation_significant(self):
        from src.interpretability.attention_analysis import correlate_with_transit
        x   = np.linspace(0, 1, 50)
        r   = correlate_with_transit(x, x, n_permutations=200)
        assert r.spearman_rho > 0.99
        assert r.p_value < 0.05


# ---------------------------------------------------------------------------
# compare_quantum_classical_attention
# ---------------------------------------------------------------------------

class TestCompareQuantumClassical:
    def test_returns_dataclass(self):
        from src.interpretability.attention_analysis import compare_quantum_classical_attention
        q = np.random.rand(14, 14)
        c = np.random.rand(14, 14)
        r = compare_quantum_classical_attention(q, c)
        assert hasattr(r, "mannwhitney_u")
        assert hasattr(r, "p_value")
        assert hasattr(r, "significant")

    def test_identical_maps_not_significant(self):
        from src.interpretability.attention_analysis import compare_quantum_classical_attention
        a = np.random.rand(14, 14)
        r = compare_quantum_classical_attention(a, a)
        assert r.p_value > 0.05

    def test_very_different_maps_significant(self):
        from src.interpretability.attention_analysis import compare_quantum_classical_attention
        q = np.ones((14, 14))             # all-ones
        c = np.zeros((14, 14)) + 1e-6     # near-zeros
        r = compare_quantum_classical_attention(q, c)
        assert r.p_value < 0.05

    def test_entropy_non_negative(self):
        from src.interpretability.attention_analysis import compare_quantum_classical_attention
        q = np.abs(np.random.rand(14, 14)) + 1e-6
        c = np.abs(np.random.rand(14, 14)) + 1e-6
        r = compare_quantum_classical_attention(q, c)
        assert r.quantum_entropy >= 0
        assert r.classical_entropy >= 0


# ---------------------------------------------------------------------------
# cls_attention_last_block
# ---------------------------------------------------------------------------

class TestClsAttentionLastBlock:
    def test_output_shape(self):
        from src.interpretability.attention_analysis import cls_attention_last_block
        # Simulate attn_maps from ViT-B/16: (B, 12, 197, 197)
        attn_maps = {f"block_{i}": torch.rand(2, 12, 197, 197) for i in range(12)}
        result = cls_attention_last_block(attn_maps)
        assert result.shape == (2, 14, 14)

    def test_sums_to_one(self):
        from src.interpretability.attention_analysis import cls_attention_last_block
        attn_maps = {f"block_{i}": torch.rand(2, 12, 197, 197) for i in range(12)}
        result = cls_attention_last_block(attn_maps)
        sums = result.reshape(2, -1).sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_uses_last_block(self):
        from src.interpretability.attention_analysis import cls_attention_last_block
        # block_11 is last; make it distinct
        attn_maps = {f"block_{i}": torch.zeros(1, 12, 197, 197) for i in range(11)}
        attn_maps["block_11"] = torch.ones(1, 12, 197, 197)
        result = cls_attention_last_block(attn_maps)
        # All patches should have equal attention (uniform from block_11)
        assert result.var().item() < 1e-6
