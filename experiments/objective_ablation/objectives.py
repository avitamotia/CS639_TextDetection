"""
Training and **evaluation scores** for Experiment C.

**One-class (separation)**  
We use the same *relative* SVDD idea as in ``SimCLR_Classifier_SCL.compute_loss`` in
``ood-llm-detect``: encourage mean LLM distance to ``c`` to be **smaller** than mean
human distance via ``softplus(mean(d_m) - mean(d_h))``.

**Soft-boundary**  
On top of that separation term (matched hyperparameters), we add the classic hinge on
**ID (machine) points only**:

    L_R = R^2 + (1/nu) * mean( relu( d_sq - R^2 ) )

where ``d_sq = ||z-c||^2`` and ``R`` is in **distance** units.  After a **warm-up** in
epochs, we set ``R`` to the ``(1 - nu)`` quantile of ``sqrt(d_sq)`` over **all training
LLM rows**, analogous to ``get_radius`` in ``train_classifier_dsvdd.py``.

**Validation scores** (higher → more confident **LLM**)  
  * one-class: ``-d_sq``  
  * soft-boundary: ``-(d_sq - R^2)``  (same shift as their eval branch when objective is soft-boundary)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

ObjectiveName = Literal["one_class", "soft_boundary"]


def _zero_loss_connected(sq_dist: torch.Tensor) -> torch.Tensor:
    """
    Scalar 0 that still depends on ``sq_dist`` so ``backward()`` reaches the head.

    Plain ``tensor.new_zeros(())`` is disconnected from the graph and crashes
    ``loss.backward()`` on single-class minibatches.
    """
    return sq_dist.sum().mul(0.0)


def separation_loss(sq_dist: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    ``softplus(mean(d_m) - mean(d_h))`` with per-batch machine/human masks.

    If the batch lacks either class, returns a **graph-connected** zero (no step signal).
    """
    m = labels == 0  # LLM
    h = labels == 1  # human
    if m.sum() == 0 or h.sum() == 0:
        return _zero_loss_connected(sq_dist)
    diff = sq_dist[m].mean() - sq_dist[h].mean()
    diff = diff.clamp(min=-100.0, max=100.0)
    return F.softplus(diff)


def soft_boundary_hinge(
    sq_dist: torch.Tensor,
    labels: torch.Tensor,
    radius: float,
    nu: float,
) -> torch.Tensor:
    """
    Soft-boundary SVDD term on **machine** rows only (ID inside the ball).

    ``radius`` is linear distance; we compare ``d_sq`` against ``radius**2``.
    """
    m = labels == 0
    if m.sum() == 0:
        return _zero_loss_connected(sq_dist)
    d_m = sq_dist[m]
    dev, dtype = sq_dist.device, sq_dist.dtype
    r = torch.tensor(radius, device=dev, dtype=dtype)
    r_sq = r * r
    hinge = F.relu(d_m - r_sq).mean()
    return r_sq + (1.0 / nu) * hinge


def training_loss(
    sq_dist: torch.Tensor,
    labels: torch.Tensor,
    objective: ObjectiveName,
    radius: float,
    nu: float,
) -> torch.Tensor:
    """
    Total loss for one forward pass.

    **Matched part:** ``separation_loss`` for both objectives.
    **Extra for soft_boundary:** hinge + ``R`` penalty as above.
    """
    l_sep = separation_loss(sq_dist, labels)
    if objective == "one_class":
        return l_sep
    return l_sep + soft_boundary_hinge(sq_dist, labels, radius, nu)


@torch.no_grad()
def update_radius_from_train_machine(
    sq_dists_machine: np.ndarray,
    nu: float,
) -> float:
    """
    ``R = quantile( sqrt(d_sq), 1 - nu )`` over training LLM squared distances.

    Mirrors ``get_radius`` in the reference training script (sqrt then quantile).
    """
    if sq_dists_machine.size == 0:
        return 1.0
    d_lin = np.sqrt(np.clip(sq_dists_machine.astype(np.float64), 1e-12, None))
    return float(np.quantile(d_lin, 1.0 - nu))


def eval_scores(sq_dist: np.ndarray, objective: ObjectiveName, radius: float) -> np.ndarray:
    """
    Higher score ⇒ model more strongly predicts **machine / LLM** (positive class for ROC).

    sklearn ``roc_auc_score`` expects ``y_score`` where larger = positive class.
    """
    d = sq_dist.astype(np.float64)
    if objective == "one_class":
        return -d
    return -(d - radius * radius)
