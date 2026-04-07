"""Adaptive Quantum Conformal Prediction (AQCP) for exoplanet transit vetting.

Implements the conformal prediction framework from arXiv 2511.18225, adapted
for our hybrid quantum-classical ViT.

Background — Standard Conformal Prediction
-------------------------------------------
For any base classifier f, conformal prediction wraps it to produce
*prediction sets* (not point predictions) with a provable coverage guarantee:

    P(y_test ∈ C(x_test)) ≥ 1 - α

where α is the user-chosen error rate.  At α=0.10 → 90% of prediction sets
contain the true label, regardless of the distribution of (x, y).

The algorithm (split conformal prediction):
    1. Reserve a calibration set (x_i, y_i), i=1..n
    2. Compute nonconformity score  s_i = score(x_i, y_i)
       For classifiers: s_i = 1 − f(x_i)[y_i]   (probability of true class)
    3. Compute quantile:  q̂ = quantile({s_i}, ⌈(n+1)(1−α)⌉/n)
    4. For test point x:  C(x) = {y : score(x, y) ≤ q̂}

What AQCP Adds
--------------
Standard conformal prediction breaks under quantum hardware noise because the
model's behaviour is non-stationary: shot noise, gate errors, and thermal
relaxation cause inference-time variance.

AQCP *adapts* the nonconformity score to incorporate this uncertainty:

    s(x, y) = (1 − μ_y) + λ · σ_y

where:
    μ_y  = mean softmax probability for class y across n_shots
    σ_y  = std  of softmax probability for class y across n_shots
    λ    = adaptation weight (default 1.0, tunable on calibration set)

When σ_y is large (noisy quantum circuit), the score is penalised → the
prediction set grows → model abstains on uncertain samples.

For simulator (default.qubit): we inject multinomial shot noise analytically,
faithfully modelling the variance a real QPU would exhibit.

Prediction set semantics (binary classification, planet=1 / FP=0)
------------------------------------------------------------------
    {1}     → "Confirmed planet" (confident)
    {0}     → "False positive"   (confident)
    {0, 1}  → "Uncertain"        → flag for astronomer follow-up
    {}      → "Empty set"        (rare, out-of-distribution)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Shot noise simulation
# ---------------------------------------------------------------------------

def simulate_shot_noise(
    probs:   torch.Tensor,
    n_shots: int,
    seed:    int | None = None,
) -> torch.Tensor:
    """Simulate quantum measurement shot noise on softmax probabilities.

    Models the variance introduced by finite sampling from a quantum
    measurement outcome distribution (multinomial shot noise):

        Var[p̂_y] ≈ p_y(1 − p_y) / n_shots    (Central Limit Theorem)

    Parameters
    ----------
    probs   : Tensor (B, C)  Softmax probabilities from a single forward pass.
    n_shots : int            Number of simulated measurement shots.
    seed    : int | None     Optional RNG seed for reproducibility.

    Returns
    -------
    Tensor (n_shots, B, C)  Noisy probability estimates per shot.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Shot noise std: σ = sqrt(p*(1-p) / n_shots)
    shot_std = torch.sqrt(probs * (1.0 - probs) / max(n_shots, 1))  # (B, C)

    # Draw n_shots independent noisy estimates
    noise = torch.randn(n_shots, *probs.shape, device=probs.device) * shot_std
    noisy = probs.unsqueeze(0) + noise                               # (n_shots, B, C)
    noisy = noisy.clamp(min=1e-8)
    noisy = noisy / noisy.sum(dim=-1, keepdim=True)                  # renormalise
    return noisy


# ---------------------------------------------------------------------------
# Nonconformity scorer
# ---------------------------------------------------------------------------

class AdaptiveNonconformityScorer:
    """Computes AQCP nonconformity scores for quantum model outputs.

    Score for sample (x, y):
        s(x, y) = (1 − μ_y) + λ · σ_y

    where μ_y and σ_y are the mean and std of the softmax probability
    for class y across n_shots simulated measurements.

    For a deterministic classical model n_shots=1 → σ_y=0 → reduces to
    standard conformal prediction score.

    Parameters
    ----------
    n_shots : int    Number of quantum measurement shots to simulate.
    lambda_ : float  Adaptation weight for the variance penalty.
    """

    def __init__(self, n_shots: int = 20, lambda_: float = 1.0) -> None:
        self._n_shots  = n_shots
        self._lambda   = lambda_

    def score(
        self,
        model: nn.Module,
        image: torch.Tensor,
        aux:   torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample nonconformity scores for a batch.

        Parameters
        ----------
        model : nn.Module          Trained classifier (forward returns (logits, *_)).
        image : Tensor (B, 2, H, W)
        aux   : Tensor (B, aux_dim)
        label : Tensor (B,) long    True class labels.

        Returns
        -------
        Tensor (B,)  Nonconformity score per sample.
        """
        with torch.no_grad():
            logits, _ = model(image, aux)   # (B, C)

        probs       = torch.softmax(logits, dim=-1)              # (B, C)
        noisy_probs = simulate_shot_noise(probs, self._n_shots)  # (n_shots, B, C)

        mu    = noisy_probs.mean(dim=0)   # (B, C) — mean across shots
        correction = 0 if noisy_probs.shape[0] <= 1 else 1
        sigma = noisy_probs.std(dim=0, correction=correction)  # (B, C) — std across shots

        # Gather values for true class
        idx    = label.long().unsqueeze(-1)          # (B, 1)
        mu_y   = mu.gather(1, idx).squeeze(-1)       # (B,)
        sigma_y = sigma.gather(1, idx).squeeze(-1)   # (B,)

        return (1.0 - mu_y) + self._lambda * sigma_y  # (B,)

    def score_all_classes(
        self,
        model: nn.Module,
        image: torch.Tensor,
        aux:   torch.Tensor,
    ) -> torch.Tensor:
        """Compute nonconformity scores for ALL classes simultaneously.

        Used during prediction to build the prediction set.

        Returns
        -------
        Tensor (B, C)  Score s(x, y) for every class y.
        """
        with torch.no_grad():
            logits, _ = model(image, aux)

        probs       = torch.softmax(logits, dim=-1)              # (B, C)
        noisy_probs = simulate_shot_noise(probs, self._n_shots)  # (n_shots, B, C)

        mu    = noisy_probs.mean(dim=0)   # (B, C)
        sigma = noisy_probs.std(dim=0)    # (B, C)

        return (1.0 - mu) + self._lambda * sigma   # (B, C)


# ---------------------------------------------------------------------------
# Conformal predictor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationResult:
    """Output of a completed calibration run.

    Attributes
    ----------
    alpha          : float           Target error rate (e.g. 0.10 for 90% coverage).
    q_hat          : float           Estimated quantile threshold.
    n_calibration  : int             Number of calibration samples used.
    coverage_upper : float           Upper bound on guaranteed coverage: ≥ 1 − α.
    """
    alpha:          float
    q_hat:          float
    n_calibration:  int
    coverage_upper: float


class ConformalPredictor:
    """AQCP wrapper for any trained classifier.

    Usage
    -----
    predictor = ConformalPredictor(model, scorer, alpha=0.10)
    predictor.calibrate(cal_loader, device)
    pred_sets = predictor.predict(test_loader, device)
    coverage  = predictor.empirical_coverage(pred_sets, true_labels)

    Parameters
    ----------
    model   : nn.Module                   Trained ExoplanetQuantumViT (or classical ViT).
    scorer  : AdaptiveNonconformityScorer  Nonconformity score function.
    alpha   : float                        Error rate.  Coverage guarantee: ≥ 1 − alpha.
    """

    def __init__(
        self,
        model:   nn.Module,
        scorer:  AdaptiveNonconformityScorer,
        alpha:   float = 0.10,
    ) -> None:
        self._model  = model
        self._scorer = scorer
        self._alpha  = alpha
        self._q_hat: float | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._q_hat is not None

    def calibrate(
        self,
        cal_loader: DataLoader,
        device:     torch.device,
    ) -> CalibrationResult:
        """Compute the quantile threshold from a held-out calibration set.

        Parameters
        ----------
        cal_loader : DataLoader  Yields (image, aux, label) 3-tuples.
        device     : torch.device

        Returns
        -------
        CalibrationResult  Immutable summary of the calibration run.
        """
        self._model.eval()
        self._model.to(device)

        all_scores: list[torch.Tensor] = []

        for image, aux, labels in cal_loader:
            image  = image.to(device)
            aux    = aux.to(device)
            labels = labels.to(device).long()

            scores = self._scorer.score(self._model, image, aux, labels)
            all_scores.append(scores.cpu())

        scores_np = torch.cat(all_scores).numpy()
        n = len(scores_np)

        # Finite-sample corrected quantile: ⌈(n+1)(1-α)⌉/n
        level = math.ceil((n + 1) * (1.0 - self._alpha)) / n
        level = min(level, 1.0)
        self._q_hat = float(np.quantile(scores_np, level))

        return CalibrationResult(
            alpha=self._alpha,
            q_hat=self._q_hat,
            n_calibration=n,
            coverage_upper=1.0 - self._alpha,
        )

    def predict(
        self,
        loader: DataLoader,
        device: torch.device,
    ) -> list[frozenset[int]]:
        """Produce prediction sets for all samples in loader.

        A sample is included in the prediction set for class y if
        score(x, y) ≤ q̂.

        Returns
        -------
        list[frozenset[int]]
            One entry per sample.  Each frozenset contains the class labels
            in the prediction set (typically 0, 1, or both).
        """
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() before predict().")

        self._model.eval()
        self._model.to(device)
        pred_sets: list[frozenset[int]] = []

        for image, aux, _ in loader:
            image = image.to(device)
            aux   = aux.to(device)

            # Scores for every class: (B, C)
            all_scores = self._scorer.score_all_classes(self._model, image, aux)
            # Prediction set: classes whose score ≤ q̂
            in_set = all_scores <= self._q_hat   # (B, C) bool
            for row in in_set:
                pred_sets.append(frozenset(int(c) for c in row.nonzero(as_tuple=True)[0]))

        return pred_sets

    @staticmethod
    def empirical_coverage(
        pred_sets:   list[frozenset[int]],
        true_labels: list[int],
    ) -> float:
        """Fraction of samples where true label is inside its prediction set."""
        assert len(pred_sets) == len(true_labels)
        covered = sum(y in ps for ps, y in zip(pred_sets, true_labels))
        return covered / len(true_labels)

    @staticmethod
    def abstention_stats(
        pred_sets:   list[frozenset[int]],
        true_labels: list[int],
    ) -> dict[str, float]:
        """Compute abstention statistics for the astronomer-review metric.

        Classifies each sample as:
            confident   — |prediction set| == 1
            uncertain   — |prediction set| > 1 (abstain)
            empty       — |prediction set| == 0 (OOD)

        Returns
        -------
        dict with keys:
            abstention_rate      — fraction of samples flagged for review
            confident_fpr        — false positive rate among confident predictions
            full_fpr             — false positive rate if no abstention
            fpr_reduction_pct    — % reduction in FPR achieved by abstaining
            n_confident          — count of confident predictions
            n_uncertain          — count of abstentions
        """
        assert len(pred_sets) == len(true_labels)
        n = len(pred_sets)

        confident_tp = confident_fp = confident_tn = confident_fn = 0
        full_tp = full_fp = full_tn = full_fn = 0
        n_uncertain = 0

        for ps, y in zip(pred_sets, true_labels):
            # Full prediction (argmax-style: pick singleton or majority)
            if len(ps) == 1:
                pred = next(iter(ps))
            elif len(ps) > 1:
                pred = 1   # uncertain → default to "planet" (recall-favoured)
            else:
                pred = 0   # empty → "false positive"

            if y == 1:
                if pred == 1: full_tp += 1
                else:         full_fn += 1
            else:
                if pred == 0: full_tn += 1
                else:         full_fp += 1

            if len(ps) == 1:
                pred_conf = next(iter(ps))
                if y == 1:
                    if pred_conf == 1: confident_tp += 1
                    else:              confident_fn += 1
                else:
                    if pred_conf == 0: confident_tn += 1
                    else:              confident_fp += 1
            else:
                n_uncertain += 1

        n_confident = n - n_uncertain

        def _fpr(fp: int, tn: int) -> float:
            return fp / (fp + tn) if (fp + tn) > 0 else 0.0

        full_fpr      = _fpr(full_fp, full_tn)
        confident_fpr = _fpr(confident_fp, confident_tn)
        fpr_reduction = (
            100.0 * (full_fpr - confident_fpr) / full_fpr
            if full_fpr > 0 else 0.0
        )

        return {
            "abstention_rate":   n_uncertain / n,
            "confident_fpr":     confident_fpr,
            "full_fpr":          full_fpr,
            "fpr_reduction_pct": fpr_reduction,
            "n_confident":       n_confident,
            "n_uncertain":       n_uncertain,
        }
