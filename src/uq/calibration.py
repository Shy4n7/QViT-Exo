"""Calibration evaluation utilities for AQCP.

Provides:
  - coverage_across_alphas()   : empirical vs nominal coverage curve (the key figure)
  - abstention_curve()         : FPR vs abstention rate as α varies
  - expected_calibration_error(): ECE for classical softmax miscalibration comparison
  - format_calibration_report(): human-readable summary for paper Table 2

The central scientific claim being validated:
    "Quantum AQCP is better calibrated than classical softmax probability"

Measured by:
    1. Coverage curve close to diagonal (y = x) → well-calibrated
    2. Lower ECE than classical softmax
    3. Abstaining on uncertain cases reduces FPR significantly
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.uq.conformal import (
    AdaptiveNonconformityScorer,
    ConformalPredictor,
    CalibrationResult,
)


# ---------------------------------------------------------------------------
# Coverage across alpha levels
# ---------------------------------------------------------------------------

def coverage_across_alphas(
    model:      nn.Module,
    cal_loader: DataLoader,
    test_loader: DataLoader,
    device:     torch.device,
    alphas:     list[float] | None = None,
    n_shots:    int = 20,
    lambda_:    float = 1.0,
) -> dict[str, list[float]]:
    """Compute empirical coverage at multiple nominal error levels.

    For each α in alphas:
        1. Calibrate AQCP on cal_loader
        2. Predict on test_loader
        3. Measure empirical coverage

    The result can be plotted as empirical vs nominal to show calibration.
    A perfectly calibrated model sits on the diagonal (y = x).

    Parameters
    ----------
    model        : nn.Module   Trained quantum or classical ViT.
    cal_loader   : DataLoader  Calibration set (yields image, aux, label).
    test_loader  : DataLoader  Test set (yields image, aux, label).
    device       : torch.device
    alphas       : list[float] Alpha values to evaluate (default 0.05 to 0.50).
    n_shots      : int         Quantum measurement shots per sample.
    lambda_      : float       AQCP variance adaptation weight.

    Returns
    -------
    dict with keys:
        "alphas"           : list[float]  Nominal error rates tested.
        "nominal_coverage" : list[float]  1 - alpha  (expected coverage).
        "empirical_coverage": list[float] Fraction of test samples covered.
        "q_hats"           : list[float]  Calibrated quantile per alpha.
    """
    if alphas is None:
        alphas = [round(a, 2) for a in np.arange(0.05, 0.55, 0.05).tolist()]

    scorer = AdaptiveNonconformityScorer(n_shots=n_shots, lambda_=lambda_)

    # Collect test labels once
    test_labels = _collect_labels(test_loader)

    nominal_coverages:  list[float] = []
    empirical_coverages: list[float] = []
    q_hats:             list[float] = []

    for alpha in alphas:
        predictor = ConformalPredictor(model, scorer, alpha=alpha)
        predictor.calibrate(cal_loader, device)
        pred_sets = predictor.predict(test_loader, device)
        emp_cov   = ConformalPredictor.empirical_coverage(pred_sets, test_labels)

        nominal_coverages.append(1.0 - alpha)
        empirical_coverages.append(emp_cov)
        q_hats.append(predictor._q_hat)  # type: ignore[attr-defined]

    return {
        "alphas":             alphas,
        "nominal_coverage":   nominal_coverages,
        "empirical_coverage": empirical_coverages,
        "q_hats":             q_hats,
    }


# ---------------------------------------------------------------------------
# Abstention curve
# ---------------------------------------------------------------------------

def abstention_curve(
    model:      nn.Module,
    cal_loader: DataLoader,
    test_loader: DataLoader,
    device:     torch.device,
    alphas:     list[float] | None = None,
    n_shots:    int = 20,
    lambda_:    float = 1.0,
) -> dict[str, list[float]]:
    """Compute FPR and abstention rate as α varies.

    Sweeps over α to show how abstaining on uncertain samples reduces the
    false positive rate.  Produces the data for the abstention tradeoff
    figure (abstention rate on x-axis, FPR on y-axis).

    Returns
    -------
    dict with keys:
        "alphas"           : list[float]
        "abstention_rates" : list[float]  Fraction of uncertain samples.
        "confident_fprs"   : list[float]  FPR among confident predictions only.
        "full_fprs"        : list[float]  FPR if no abstention.
    """
    if alphas is None:
        alphas = [round(a, 2) for a in np.arange(0.01, 0.50, 0.03).tolist()]

    scorer     = AdaptiveNonconformityScorer(n_shots=n_shots, lambda_=lambda_)
    test_labels = _collect_labels(test_loader)

    abstention_rates: list[float] = []
    confident_fprs:   list[float] = []
    full_fprs:        list[float] = []

    for alpha in alphas:
        predictor = ConformalPredictor(model, scorer, alpha=alpha)
        predictor.calibrate(cal_loader, device)
        pred_sets = predictor.predict(test_loader, device)
        stats     = ConformalPredictor.abstention_stats(pred_sets, test_labels)

        abstention_rates.append(stats["abstention_rate"])
        confident_fprs.append(stats["confident_fpr"])
        full_fprs.append(stats["full_fpr"])

    return {
        "alphas":            alphas,
        "abstention_rates":  abstention_rates,
        "confident_fprs":    confident_fprs,
        "full_fprs":         full_fprs,
    }


# ---------------------------------------------------------------------------
# Expected Calibration Error (classical softmax comparison)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    n_bins:     int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) of classical softmax probabilities.

    ECE measures the gap between confidence (max softmax) and actual accuracy,
    binned by confidence level.  A well-calibrated classifier has ECE ≈ 0.

    Use this to show classical softmax is miscalibrated (ECE > 0) compared to
    AQCP coverage (which has a provable guarantee).

    Parameters
    ----------
    model  : nn.Module   Trained classifier.
    loader : DataLoader  Yields (image, aux, label).
    device : torch.device
    n_bins : int         Number of confidence bins (default 10).

    Returns
    -------
    float  ECE in [0, 1].  Lower is better-calibrated.
    """
    model.eval()
    model.to(device)

    all_confs:   list[float] = []
    all_correct: list[float] = []

    with torch.no_grad():
        for image, aux, labels in loader:
            image  = image.to(device)
            aux    = aux.to(device)
            labels = labels.to(device).long()

            logits, _ = model(image, aux)
            probs      = torch.softmax(logits, dim=-1)
            confidence, preds = probs.max(dim=-1)

            for c, p, y in zip(
                confidence.cpu().tolist(),
                preds.cpu().tolist(),
                labels.cpu().tolist(),
            ):
                all_confs.append(c)
                all_correct.append(float(p == y))

    confs   = np.array(all_confs)
    correct = np.array(all_correct)
    n       = len(confs)
    ece     = 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = confs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_calibration_report(
    cal_result:   CalibrationResult,
    emp_coverage: float,
    abs_stats:    dict[str, float],
    ece:          float,
) -> str:
    """Return a human-readable calibration summary for paper Table 2.

    Parameters
    ----------
    cal_result   : CalibrationResult  From ConformalPredictor.calibrate().
    emp_coverage : float              Empirical coverage at the calibrated α.
    abs_stats    : dict               From ConformalPredictor.abstention_stats().
    ece          : float              Classical softmax ECE.
    """
    lines = [
        "=" * 58,
        "  UNCERTAINTY QUANTIFICATION REPORT — Phase 4",
        "=" * 58,
        f"  Target error rate α     : {cal_result.alpha:.2f}",
        f"  Nominal coverage (1−α)  : {cal_result.coverage_upper:.2%}",
        f"  Empirical coverage      : {emp_coverage:.2%}  "
        + ("✓ VALID" if emp_coverage >= cal_result.coverage_upper - 0.02 else "✗ SHORT"),
        f"  Calibrated quantile q̂  : {cal_result.q_hat:.4f}",
        f"  Calibration set size    : {cal_result.n_calibration}",
        "-" * 58,
        "  Abstention statistics",
        f"  Abstention rate         : {abs_stats['abstention_rate']:.1%}",
        f"  Confident predictions   : {abs_stats['n_confident']}",
        f"  Uncertain (flagged)     : {abs_stats['n_uncertain']}",
        f"  Full FPR (no abstain)   : {abs_stats['full_fpr']:.1%}",
        f"  Confident FPR (abstain) : {abs_stats['confident_fpr']:.1%}",
        f"  FPR reduction           : {abs_stats['fpr_reduction_pct']:.1f}%",
        "-" * 58,
        "  Calibration quality",
        f"  Classical softmax ECE   : {ece:.4f}  (lower = better calibrated)",
        f"  AQCP coverage gap       : {abs(emp_coverage - cal_result.coverage_upper):.4f}",
        "=" * 58,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _collect_labels(loader: DataLoader) -> list[int]:
    """Collect all labels from a DataLoader into a flat list."""
    labels: list[int] = []
    for _, _, batch_labels in loader:
        labels.extend(batch_labels.tolist())
    return labels
