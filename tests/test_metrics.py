"""Tests for classification metrics used in exoplanet transit vetting.

TDD: All tests written before implementation.
Key metric: planet_recall (sensitivity for label=1) — false negatives are
more costly than false positives in the exoplanet discovery domain.
"""

import dataclasses

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_binary_data():
    """y_true and y_pred that are perfectly aligned, y_prob maximally confident."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.01, 0.01, 0.01, 0.01, 0.99, 0.99, 0.99, 0.99])
    return y_true, y_pred, y_prob


@pytest.fixture
def all_wrong_data():
    """y_pred is the exact complement of y_true."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.99, 0.99, 0.99, 0.99, 0.01, 0.01, 0.01, 0.01])
    return y_true, y_pred, y_prob


@pytest.fixture
def known_confusion_data():
    """4 TP, 1 FP, 1 FN, 4 TN — gives precision=4/5, recall=4/5 for class 1."""
    # class 1 = planet
    # TP: predict 1, true 1  x4
    # FP: predict 1, true 0  x1
    # FN: predict 0, true 1  x1
    # TN: predict 0, true 0  x4
    y_true = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.85, 0.8, 0.75, 0.6, 0.3, 0.2, 0.15, 0.1, 0.05])
    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------

def _import():
    from src.training.metrics import MetricsResult, compute_metrics, format_metrics
    return MetricsResult, compute_metrics, format_metrics


# ---------------------------------------------------------------------------
# MetricsResult dataclass structure
# ---------------------------------------------------------------------------

class TestMetricsResultStructure:
    def test_metrics_result_is_dataclass(self):
        """MetricsResult must be a dataclass."""
        MetricsResult, _, _ = _import()
        assert dataclasses.is_dataclass(MetricsResult), (
            "MetricsResult must be defined with @dataclass"
        )

    def test_metrics_result_has_required_fields(self):
        """MetricsResult must have all seven required fields."""
        MetricsResult, _, _ = _import()
        field_names = {f.name for f in dataclasses.fields(MetricsResult)}
        required = {
            "precision", "recall", "f1", "auc_roc",
            "planet_recall", "planet_precision", "confusion_matrix",
        }
        missing = required - field_names
        assert not missing, f"MetricsResult is missing fields: {missing}"

    def test_planet_recall_field(self, known_confusion_data):
        """MetricsResult.planet_recall must match recall for label=1 specifically."""
        MetricsResult, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = known_confusion_data
        result = compute_metrics(y_true, y_pred, y_prob)
        # 4 TP, 1 FN → planet_recall = 4/5 = 0.8
        assert abs(result.planet_recall - 0.8) < 1e-6, (
            f"Expected planet_recall=0.8, got {result.planet_recall}"
        )


# ---------------------------------------------------------------------------
# Perfect predictions
# ---------------------------------------------------------------------------

class TestPerfectPredictions:
    def test_perfect_predictions_all_ones(self, perfect_binary_data):
        """All-correct predictions → precision=recall=f1=1.0."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = perfect_binary_data
        result = compute_metrics(y_true, y_pred, y_prob)
        assert abs(result.precision - 1.0) < 1e-6, f"precision={result.precision}"
        assert abs(result.recall - 1.0) < 1e-6, f"recall={result.recall}"
        assert abs(result.f1 - 1.0) < 1e-6, f"f1={result.f1}"

    def test_perfect_planet_recall(self, perfect_binary_data):
        """Perfect predictions → planet_recall=1.0."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = perfect_binary_data
        result = compute_metrics(y_true, y_pred, y_prob)
        assert abs(result.planet_recall - 1.0) < 1e-6, (
            f"Expected planet_recall=1.0, got {result.planet_recall}"
        )


# ---------------------------------------------------------------------------
# All-wrong predictions
# ---------------------------------------------------------------------------

class TestAllWrongPredictions:
    def test_all_wrong_planet_recall_is_zero(self, all_wrong_data):
        """All-flipped predictions → recall for planet class (label=1) = 0.0."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = all_wrong_data
        result = compute_metrics(y_true, y_pred, y_prob)
        assert abs(result.planet_recall - 0.0) < 1e-6, (
            f"Expected planet_recall=0.0, got {result.planet_recall}"
        )

    def test_all_wrong_planet_precision_is_zero(self, all_wrong_data):
        """All-flipped: no true planets in positive predictions → planet_precision=0.0."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = all_wrong_data
        result = compute_metrics(y_true, y_pred, y_prob)
        assert abs(result.planet_precision - 0.0) < 1e-6, (
            f"Expected planet_precision=0.0, got {result.planet_precision}"
        )


# ---------------------------------------------------------------------------
# Known confusion matrix
# ---------------------------------------------------------------------------

class TestKnownConfusionMatrix:
    def test_known_confusion_matrix_precision(self, known_confusion_data):
        """4 TP, 1 FP → planet_precision = 4/5 = 0.8."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = known_confusion_data
        result = compute_metrics(y_true, y_pred, y_prob)
        assert abs(result.planet_precision - 0.8) < 1e-6, (
            f"Expected planet_precision=0.8, got {result.planet_precision}"
        )

    def test_known_confusion_matrix_recall(self, known_confusion_data):
        """4 TP, 1 FN → planet_recall = 4/5 = 0.8."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = known_confusion_data
        result = compute_metrics(y_true, y_pred, y_prob)
        assert abs(result.planet_recall - 0.8) < 1e-6, (
            f"Expected planet_recall=0.8, got {result.planet_recall}"
        )

    def test_known_confusion_matrix_shape(self, known_confusion_data):
        """confusion_matrix field must be a 2x2 numpy array."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = known_confusion_data
        result = compute_metrics(y_true, y_pred, y_prob)
        assert isinstance(result.confusion_matrix, np.ndarray), (
            "confusion_matrix must be np.ndarray"
        )
        assert result.confusion_matrix.shape == (2, 2), (
            f"Expected (2,2) confusion matrix, got {result.confusion_matrix.shape}"
        )

    def test_known_confusion_matrix_values(self, known_confusion_data):
        """Confusion matrix values must match: TN=4, FP=1, FN=1, TP=4."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = known_confusion_data
        result = compute_metrics(y_true, y_pred, y_prob)
        # sklearn convention: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = result.confusion_matrix.ravel()
        assert tn == 4, f"Expected TN=4, got {tn}"
        assert fp == 1, f"Expected FP=1, got {fp}"
        assert fn == 1, f"Expected FN=1, got {fn}"
        assert tp == 4, f"Expected TP=4, got {tp}"


# ---------------------------------------------------------------------------
# AUC-ROC
# ---------------------------------------------------------------------------

class TestAucRoc:
    def test_auc_roc_in_range(self, known_confusion_data):
        """AUC-ROC must always be in [0.0, 1.0]."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = known_confusion_data
        result = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= result.auc_roc <= 1.0, (
            f"AUC-ROC {result.auc_roc} outside [0.0, 1.0]"
        )

    def test_auc_roc_perfect(self, perfect_binary_data):
        """Perfectly separable probability scores → AUC-ROC = 1.0."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = perfect_binary_data
        result = compute_metrics(y_true, y_pred, y_prob)
        assert abs(result.auc_roc - 1.0) < 1e-6, (
            f"Expected AUC-ROC=1.0, got {result.auc_roc}"
        )

    def test_auc_roc_worst(self, all_wrong_data):
        """Worst-case probability scores → AUC-ROC = 0.0."""
        _, compute_metrics, _ = _import()
        y_true, y_pred, y_prob = all_wrong_data
        result = compute_metrics(y_true, y_pred, y_prob)
        assert abs(result.auc_roc - 0.0) < 1e-6, (
            f"Expected AUC-ROC=0.0 for inverted probs, got {result.auc_roc}"
        )


# ---------------------------------------------------------------------------
# format_metrics
# ---------------------------------------------------------------------------

class TestFormatMetrics:
    def test_format_metrics_returns_string(self, perfect_binary_data):
        """format_metrics must return a non-empty string."""
        MetricsResult, compute_metrics, format_metrics = _import()
        y_true, y_pred, y_prob = perfect_binary_data
        result = compute_metrics(y_true, y_pred, y_prob)
        formatted = format_metrics(result)
        assert isinstance(formatted, str) and len(formatted) > 0

    def test_format_metrics_contains_key_fields(self, perfect_binary_data):
        """Formatted string must mention precision, recall, f1, auc, and planet_recall."""
        MetricsResult, compute_metrics, format_metrics = _import()
        y_true, y_pred, y_prob = perfect_binary_data
        result = compute_metrics(y_true, y_pred, y_prob)
        formatted = format_metrics(result).lower()
        for keyword in ("precision", "recall", "f1", "auc"):
            assert keyword in formatted, (
                f"format_metrics output missing keyword '{keyword}'"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_sample_planet(self):
        """Single-sample edge case: one planet, predicted correctly."""
        _, compute_metrics, _ = _import()
        y_true = np.array([1])
        y_pred = np.array([1])
        y_prob = np.array([0.9])
        result = compute_metrics(y_true, y_pred, y_prob)
        assert result.planet_recall == 1.0

    def test_metrics_with_large_balanced_dataset(self):
        """Metrics are stable on a large balanced dataset (1000 samples each class)."""
        _, compute_metrics, _ = _import()
        rng = np.random.default_rng(seed=99)
        n = 1000
        y_true = np.array([0] * n + [1] * n)
        # 90% accuracy classifier
        y_pred = y_true.copy()
        flip_idx = rng.choice(2 * n, size=200, replace=False)
        y_pred[flip_idx] = 1 - y_pred[flip_idx]
        y_prob = np.where(y_pred == 1, 0.8, 0.2).astype(float)

        result = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.auc_roc <= 1.0
        assert 0.0 <= result.planet_recall <= 1.0
