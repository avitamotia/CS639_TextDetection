"""
Validation metrics for Experiment C.

**Operational metrics (as requested)**  
  * **Score scale:** mean and std of detection scores **conditional on true class**
    (LLM vs human).  Shows calibration shift between one-class and soft-boundary.
  * **FPR @ fixed TPR:** interpolate the ROC curve at e.g. TPR = 0.95 — cost of
    catching “almost all” LLM text in terms of false alarms on humans.

**Threshold sensitivity**  
We report **best F1** and the **threshold** on the score that achieves it (grid over
quantiles).  For a full curve, use the saved JSON history or extend with
``precision_recall_curve`` plots in ``diagrams.py`` — the ROC overlay already shows
trade-offs beyond a single point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np


@lru_cache(maxsize=1)
def _sklearn_metrics():
    """Lazy import: keeps ``import metrics`` light; fails at first use with clear error."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            auc,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )
    except ImportError as exc:
        raise ImportError(
            "Install scikit-learn in your active environment: pip install scikit-learn"
        ) from exc
    return (
        accuracy_score,
        auc,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )


def check_sklearn_available() -> None:
    """Call from CLI entrypoint to fail fast before downloading data."""
    _sklearn_metrics()


@dataclass
class EvalReport:
    """One validation snapshot; JSON-serializable via ``to_dict``."""

    roc_auc: float
    pr_auc: float
    fpr_at_target_tpr: float
    target_tpr: float
    score_mean_llm: float
    score_std_llm: float
    score_mean_human: float
    score_std_human: float
    best_f1: float
    best_f1_threshold: float
    accuracy_at_best_f1: float
    precision_at_best_f1: float
    recall_at_best_f1: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fpr_at_fixed_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """
    False positive rate when true positive rate equals ``target_tpr``.

    ``y_true``: 1 = LLM (positive).  ``y_score``: higher = more LLM-like.
    """
    *_, roc_curve = _sklearn_metrics()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(tpr) < 2:
        return float("nan")
    return float(np.interp(target_tpr, tpr, fpr))


def score_scale_per_class(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Returns (mean_llm, std_llm, mean_human, std_human) for ``y_score``."""
    llm = y_true == 1
    hum = y_true == 0
    s_llm = y_score[llm]
    s_h = y_score[hum]
    return (
        float(s_llm.mean()) if s_llm.size else float("nan"),
        float(s_llm.std()) if s_llm.size else float("nan"),
        float(s_h.mean()) if s_h.size else float("nan"),
        float(s_h.std()) if s_h.size else float("nan"),
    )


def best_f1_and_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Scan thresholds at score quantiles; maximize F1 for positive = LLM.

    **Why not a single number in the write-up:** F1-optimal threshold **moves** when
    the score scale shifts (soft-boundary vs one-class); report curves / tables.
    """
    f1_score = _sklearn_metrics()[2]
    thr_grid = np.unique(np.quantile(y_score, np.linspace(0, 1, 101)))
    best_t, best_f1 = float(thr_grid[len(thr_grid) // 2]), 0.0
    for t in thr_grid:
        pred = (y_score >= t).astype(int)
        f1v = float(f1_score(y_true, pred, zero_division=0))
        if f1v > best_f1:
            best_f1, best_t = f1v, float(t)
    return best_f1, best_t


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_tpr: float = 0.95,
) -> EvalReport:
    """Bundle ROC-AUC, PR-AUC, FPR@TPR, score moments, and F1@threshold."""
    (
        accuracy_score,
        auc_fn,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        _,
    ) = _sklearn_metrics()
    roc_auc = float(roc_auc_score(y_true, y_score))
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc_fn(rec, prec))
    fpr_tgt = fpr_at_fixed_tpr(y_true, y_score, target_tpr)
    m_llm, s_llm, m_h, s_h = score_scale_per_class(y_true, y_score)
    bf1, bthr = best_f1_and_threshold(y_true, y_score)
    pred = (y_score >= bthr).astype(int)
    return EvalReport(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        fpr_at_target_tpr=fpr_tgt,
        target_tpr=target_tpr,
        score_mean_llm=m_llm,
        score_std_llm=s_llm,
        score_mean_human=m_h,
        score_std_human=s_h,
        best_f1=bf1,
        best_f1_threshold=bthr,
        accuracy_at_best_f1=float(accuracy_score(y_true, pred)),
        precision_at_best_f1=float(precision_score(y_true, pred, zero_division=0)),
        recall_at_best_f1=float(recall_score(y_true, pred, zero_division=0)),
    )
