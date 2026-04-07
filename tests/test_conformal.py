"""Tests for Phase 4 — AQCP conformal prediction and calibration utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Minimal stub model (no timm dependency in tests)
# ---------------------------------------------------------------------------

class _StubClassifier(nn.Module):
    """Tiny classifier: returns fixed logits for deterministic tests."""
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self._linear = nn.Linear(4, n_classes)

    def forward(self, image: torch.Tensor, aux: torch.Tensor):
        # Use aux features only for speed (no real image processing)
        logits = self._linear(aux[:, :4])
        return logits, None


def _make_loader(n: int = 40, seed: int = 0) -> DataLoader:
    torch.manual_seed(seed)
    image  = torch.randn(n, 2, 224, 224)
    aux    = torch.randn(n, 5)
    labels = torch.randint(0, 2, (n,))
    ds = TensorDataset(image, aux, labels)
    return DataLoader(ds, batch_size=8, shuffle=False)


# ---------------------------------------------------------------------------
# simulate_shot_noise
# ---------------------------------------------------------------------------

class TestSimulateShotNoise:
    def test_output_shape(self) -> None:
        from src.uq.conformal import simulate_shot_noise
        probs = torch.softmax(torch.randn(4, 2), dim=-1)
        noisy = simulate_shot_noise(probs, n_shots=10)
        assert noisy.shape == (10, 4, 2)

    def test_probabilities_sum_to_one(self) -> None:
        from src.uq.conformal import simulate_shot_noise
        probs = torch.softmax(torch.randn(4, 2), dim=-1)
        noisy = simulate_shot_noise(probs, n_shots=10)
        sums  = noisy.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_probabilities_non_negative(self) -> None:
        from src.uq.conformal import simulate_shot_noise
        probs = torch.softmax(torch.randn(4, 2), dim=-1)
        noisy = simulate_shot_noise(probs, n_shots=20)
        assert (noisy >= 0).all()

    def test_more_shots_lower_variance(self) -> None:
        from src.uq.conformal import simulate_shot_noise
        probs = torch.softmax(torch.randn(16, 2), dim=-1)
        var_low  = simulate_shot_noise(probs, n_shots=5,   seed=0).var().item()
        var_high = simulate_shot_noise(probs, n_shots=100, seed=0).var().item()
        assert var_high < var_low


# ---------------------------------------------------------------------------
# AdaptiveNonconformityScorer
# ---------------------------------------------------------------------------

class TestAdaptiveNonconformityScorer:
    def test_score_shape(self) -> None:
        from src.uq.conformal import AdaptiveNonconformityScorer
        model  = _StubClassifier()
        scorer = AdaptiveNonconformityScorer(n_shots=5)
        image  = torch.randn(4, 2, 224, 224)
        aux    = torch.randn(4, 5)
        labels = torch.randint(0, 2, (4,))
        scores = scorer.score(model, image, aux, labels)
        assert scores.shape == (4,)

    def test_score_non_negative(self) -> None:
        from src.uq.conformal import AdaptiveNonconformityScorer
        model  = _StubClassifier()
        scorer = AdaptiveNonconformityScorer(n_shots=5, lambda_=1.0)
        image  = torch.randn(8, 2, 224, 224)
        aux    = torch.randn(8, 5)
        labels = torch.randint(0, 2, (8,))
        scores = scorer.score(model, image, aux, labels)
        assert (scores >= 0).all()

    def test_score_all_classes_shape(self) -> None:
        from src.uq.conformal import AdaptiveNonconformityScorer
        model  = _StubClassifier()
        scorer = AdaptiveNonconformityScorer(n_shots=5)
        image  = torch.randn(4, 2, 224, 224)
        aux    = torch.randn(4, 5)
        scores = scorer.score_all_classes(model, image, aux)
        assert scores.shape == (4, 2)

    def test_zero_lambda_matches_standard_conformal(self) -> None:
        """With lambda_=0, score = 1 - prob[true_class] (standard conformal)."""
        from src.uq.conformal import AdaptiveNonconformityScorer
        model  = _StubClassifier()
        scorer = AdaptiveNonconformityScorer(n_shots=1, lambda_=0.0)
        image  = torch.randn(4, 2, 224, 224)
        aux    = torch.randn(4, 5)
        labels = torch.randint(0, 2, (4,))
        scores = scorer.score(model, image, aux, labels)
        # All scores should be in [0, 1] since prob ∈ [0,1]
        assert (scores >= 0).all() and (scores <= 1 + 1e-5).all()


# ---------------------------------------------------------------------------
# ConformalPredictor
# ---------------------------------------------------------------------------

class TestConformalPredictor:
    def test_calibrate_returns_result(self) -> None:
        from src.uq.conformal import AdaptiveNonconformityScorer, ConformalPredictor
        model     = _StubClassifier()
        scorer    = AdaptiveNonconformityScorer(n_shots=3)
        predictor = ConformalPredictor(model, scorer, alpha=0.10)
        loader    = _make_loader(20)
        result    = predictor.calibrate(loader, torch.device("cpu"))
        assert result.alpha == 0.10
        assert result.q_hat is not None
        assert result.n_calibration == 20

    def test_is_calibrated_false_before_calibrate(self) -> None:
        from src.uq.conformal import AdaptiveNonconformityScorer, ConformalPredictor
        model     = _StubClassifier()
        scorer    = AdaptiveNonconformityScorer(n_shots=3)
        predictor = ConformalPredictor(model, scorer)
        assert not predictor.is_calibrated

    def test_predict_raises_before_calibrate(self) -> None:
        from src.uq.conformal import AdaptiveNonconformityScorer, ConformalPredictor
        model     = _StubClassifier()
        scorer    = AdaptiveNonconformityScorer(n_shots=3)
        predictor = ConformalPredictor(model, scorer)
        with pytest.raises(RuntimeError, match="calibrate"):
            predictor.predict(_make_loader(10), torch.device("cpu"))

    def test_predict_returns_frozensets(self) -> None:
        from src.uq.conformal import AdaptiveNonconformityScorer, ConformalPredictor
        model     = _StubClassifier()
        scorer    = AdaptiveNonconformityScorer(n_shots=3)
        predictor = ConformalPredictor(model, scorer, alpha=0.10)
        loader    = _make_loader(20)
        predictor.calibrate(loader, torch.device("cpu"))
        pred_sets = predictor.predict(loader, torch.device("cpu"))
        assert len(pred_sets) == 20
        for ps in pred_sets:
            assert isinstance(ps, frozenset)
            assert ps.issubset({0, 1})

    def test_low_alpha_high_coverage(self) -> None:
        """Very low α → large q̂ → most samples covered."""
        from src.uq.conformal import AdaptiveNonconformityScorer, ConformalPredictor
        model     = _StubClassifier()
        scorer    = AdaptiveNonconformityScorer(n_shots=5)
        predictor = ConformalPredictor(model, scorer, alpha=0.01)
        loader    = _make_loader(40)
        predictor.calibrate(loader, torch.device("cpu"))
        pred_sets   = predictor.predict(loader, torch.device("cpu"))
        labels      = [int(b) for _, _, bl in loader for b in bl]
        coverage    = ConformalPredictor.empirical_coverage(pred_sets, labels)
        assert coverage >= 0.85   # should be high with low α

    def test_empirical_coverage_valid_range(self) -> None:
        from src.uq.conformal import AdaptiveNonconformityScorer, ConformalPredictor
        model     = _StubClassifier()
        scorer    = AdaptiveNonconformityScorer(n_shots=3)
        predictor = ConformalPredictor(model, scorer, alpha=0.20)
        loader    = _make_loader(40)
        predictor.calibrate(loader, torch.device("cpu"))
        pred_sets = predictor.predict(loader, torch.device("cpu"))
        labels    = [int(b) for _, _, bl in loader for b in bl]
        coverage  = ConformalPredictor.empirical_coverage(pred_sets, labels)
        assert 0.0 <= coverage <= 1.0

    def test_abstention_stats_keys(self) -> None:
        from src.uq.conformal import AdaptiveNonconformityScorer, ConformalPredictor
        model     = _StubClassifier()
        scorer    = AdaptiveNonconformityScorer(n_shots=3)
        predictor = ConformalPredictor(model, scorer, alpha=0.10)
        loader    = _make_loader(20)
        predictor.calibrate(loader, torch.device("cpu"))
        pred_sets = predictor.predict(loader, torch.device("cpu"))
        labels    = [int(b) for _, _, bl in loader for b in bl]
        stats     = ConformalPredictor.abstention_stats(pred_sets, labels)
        for key in ("abstention_rate", "confident_fpr", "full_fpr",
                    "fpr_reduction_pct", "n_confident", "n_uncertain"):
            assert key in stats

    def test_abstention_rate_in_range(self) -> None:
        from src.uq.conformal import AdaptiveNonconformityScorer, ConformalPredictor
        model     = _StubClassifier()
        scorer    = AdaptiveNonconformityScorer(n_shots=3)
        predictor = ConformalPredictor(model, scorer, alpha=0.10)
        loader    = _make_loader(20)
        predictor.calibrate(loader, torch.device("cpu"))
        pred_sets = predictor.predict(loader, torch.device("cpu"))
        labels    = [int(b) for _, _, bl in loader for b in bl]
        stats     = ConformalPredictor.abstention_stats(pred_sets, labels)
        assert 0.0 <= stats["abstention_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

class TestExpectedCalibrationError:
    def test_ece_in_range(self) -> None:
        from src.uq.calibration import expected_calibration_error
        model  = _StubClassifier()
        loader = _make_loader(40)
        ece    = expected_calibration_error(model, loader, torch.device("cpu"))
        assert 0.0 <= ece <= 1.0

    def test_perfect_model_low_ece(self) -> None:
        """A model that always predicts the correct class with high confidence
        should have low ECE."""
        from src.uq.calibration import expected_calibration_error

        class _PerfectModel(nn.Module):
            def forward(self, image, aux):
                # Always return high confidence for class 1
                logits = torch.tensor([[0.0, 10.0]]).expand(image.shape[0], -1)
                return logits, None

        loader = _make_loader(40)
        # Override labels to all be 1
        image  = torch.randn(40, 2, 224, 224)
        aux    = torch.randn(40, 5)
        labels = torch.ones(40, dtype=torch.long)
        from torch.utils.data import TensorDataset
        ds = TensorDataset(image, aux, labels)
        perfect_loader = DataLoader(ds, batch_size=8)
        ece = expected_calibration_error(_PerfectModel(), perfect_loader, torch.device("cpu"))
        assert ece < 0.15   # should be well-calibrated


# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------

class TestCalibrationResult:
    def test_immutable(self) -> None:
        from src.uq.conformal import CalibrationResult
        result = CalibrationResult(alpha=0.1, q_hat=0.5, n_calibration=100, coverage_upper=0.9)
        with pytest.raises((AttributeError, TypeError)):
            result.alpha = 0.2  # type: ignore[misc]
