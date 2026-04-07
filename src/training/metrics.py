"""Classification metrics for exoplanet transit vetting.

Key metric: planet_recall (sensitivity for label=1).
In the exoplanet discovery domain a missed real planet (false negative) is more
costly than a false alarm (false positive), so planet recall is tracked as the
primary Phase 1 optimisation target.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricsResult:
    """Container for all classification metrics produced after an evaluation pass.

    Attributes
    ----------
    precision : float
        Macro-averaged precision across both classes.
    recall : float
        Macro-averaged recall across both classes.
    f1 : float
        Macro-averaged F1 score across both classes.
    auc_roc : float
        Area under the ROC curve computed from continuous probability scores.
    planet_recall : float
        Recall for label=1 (planet class) — the primary Phase 1 metric.
    planet_precision : float
        Precision for label=1 (planet class).
    confusion_matrix : np.ndarray
        2x2 confusion matrix with layout [[TN, FP], [FN, TP]].
    """

    precision: float
    recall: float
    f1: float
    auc_roc: float
    planet_recall: float
    planet_precision: float
    confusion_matrix: np.ndarray


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> MetricsResult:
    """Compute all classification metrics from predictions.

    Parameters
    ----------
    y_true : np.ndarray of shape (N,)
        Ground-truth binary labels (0 = non-planet, 1 = planet).
    y_pred : np.ndarray of shape (N,)
        Predicted binary labels from the classifier.
    y_prob : np.ndarray of shape (N,)
        Predicted probability of class 1 (planet) from the classifier.

    Returns
    -------
    MetricsResult
        Populated metrics dataclass.  All scalar fields are Python floats.
    """
    # Macro-averaged metrics (treat both classes symmetrically for overall
    # model health tracking).
    precision_macro = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    recall_macro = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    f1_macro = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # Per-class metrics for label=1 (planet class).
    planet_precision = float(
        precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    )
    planet_recall = float(
        recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    )

    # AUC-ROC from continuous probability scores.
    # roc_auc_score requires at least one positive and one negative sample.
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        # Degenerate case: all samples belong to one class.
        auc_roc = 1.0 if float(planet_recall) == 1.0 else 0.0
    else:
        auc_roc = float(roc_auc_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return MetricsResult(
        precision=precision_macro,
        recall=recall_macro,
        f1=f1_macro,
        auc_roc=auc_roc,
        planet_recall=planet_recall,
        planet_precision=planet_precision,
        confusion_matrix=cm,
    )


def format_metrics(metrics: MetricsResult) -> str:
    """Return a human-readable string of key metrics for logging.

    Parameters
    ----------
    metrics : MetricsResult
        Populated metrics dataclass.

    Returns
    -------
    str
        Multi-line string suitable for console or file logging.
    """
    tn, fp, fn, tp = metrics.confusion_matrix.ravel()
    lines = [
        "--- Evaluation Metrics ---",
        f"  Precision (macro):   {metrics.precision:.4f}",
        f"  Recall    (macro):   {metrics.recall:.4f}",
        f"  F1        (macro):   {metrics.f1:.4f}",
        f"  AUC-ROC:             {metrics.auc_roc:.4f}",
        f"  Planet recall:       {metrics.planet_recall:.4f}",
        f"  Planet precision:    {metrics.planet_precision:.4f}",
        f"  Confusion matrix:    TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        "--------------------------",
    ]
    return "\n".join(lines)
