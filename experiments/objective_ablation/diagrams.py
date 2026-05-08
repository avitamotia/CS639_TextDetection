"""
Figures for Experiment C (matplotlib only when saving).

**Curves vs single threshold:** we save an **ROC overlay** so you can discuss
trade-offs; F1 at one threshold is still printed/JSON’d in ``metrics.evaluate`` but
should not be the only reported statistic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib required for figures. pip install matplotlib"
        ) from exc
    return plt


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def draw_concept_oneclass_vs_softboundary(out_path: Path, dpi: int = 140) -> Path:
    """
    Toy 1-D style plot: squared distance vs abstract index, and shifted scores.

    Illustrates why soft-boundary rescales scores by ``R^2`` (conceptual, not real data).
    """
    plt = _plt()
    out_path = Path(out_path)
    _mkdir(out_path.parent)

    x = np.linspace(0, 3, 200)
    d_llm = 0.3 + 0.15 * np.exp(-((x - 0.8) ** 2) / 0.08)
    d_hum = 0.3 + 1.2 * np.exp(-((x - 2.0) ** 2) / 0.25)
    R = 0.65
    r_sq = R**2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    ax = axes[0]
    ax.plot(x, d_llm, label="LLM $d_{sq}$", color="#27ae60")
    ax.plot(x, d_hum, label="human $d_{sq}$", color="#c0392b")
    ax.axhline(r_sq, color="#7f8c8d", linestyle="--", label=r"$R^2$")
    ax.set_title(r"Squared distance $\|z-c\|^2$ (schematic)")
    ax.set_xlabel("abstract axis")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(x, -d_llm, label="one-class $-d_{sq}$", color="#2980b9")
    ax.plot(x, -d_hum, color="#8e44ad", alpha=0.6, label="human $-d_{sq}$")
    ax.plot(x, -(d_llm - r_sq), "--", color="#16a085", label="SB LLM $-(d_{sq}-R^2)$")
    ax.plot(x, -(d_hum - r_sq), ":", color="#d35400", label="SB human")
    ax.set_title("Scores (higher → more LLM)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.suptitle("Experiment C: one-class vs soft-boundary scoring (concept)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_val_score_histograms(
    payloads: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    dpi: int = 140,
) -> Path:
    """One panel per run name: LLM vs human score distributions on validation."""
    plt = _plt()
    out_path = Path(out_path)
    _mkdir(out_path.parent)

    names = sorted(payloads.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), dpi=dpi)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        pack = payloads[name]
        s, y = pack["scores"], pack["y_true"]
        ax.hist(s[y == 1], bins=40, alpha=0.55, density=True, color="#27ae60", label="LLM")
        ax.hist(s[y == 0], bins=40, alpha=0.55, density=True, color="#c0392b", label="human")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("score (↑ LLM)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Validation score scale by class", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_roc_overlay(
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    dpi: int = 140,
) -> Path:
    """Multiple ROC curves on one axes (positive class = LLM)."""
    from sklearn.metrics import auc, roc_curve

    plt = _plt()
    out_path = Path(out_path)
    _mkdir(out_path.parent)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    for name, (y_true, y_score) in sorted(series.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (positive = LLM) — compare objectives without one threshold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_pr_overlay(
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    dpi: int = 140,
) -> Path:
    """Precision–recall curves (positive = LLM); legend shows PR-AUC (trapezoid)."""
    from sklearn.metrics import auc, precision_recall_curve

    plt = _plt()
    out_path = Path(out_path)
    _mkdir(out_path.parent)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    for name, (y_true, y_score) in sorted(series.items()):
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=2, label=f"{name} PR-AUC={pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR (positive = LLM)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_pca_z_scatter(
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    dpi: int = 140,
) -> Path:
    """
    First two PCA components of projected ``z`` (validation).

    ``series`` maps run name → (z, y_true) with y_true: 1 = LLM, 0 = human.
    """
    from sklearn.decomposition import PCA

    plt = _plt()
    out_path = Path(out_path)
    _mkdir(out_path.parent)

    names = sorted(series.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), dpi=dpi)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        z, y = series[name]
        if z.shape[0] < 3:
            ax.set_title(f"{name} (too few points)")
            continue
        xy = PCA(n_components=2, random_state=0).fit_transform(z)
        m = y == 1
        h = y == 0
        ax.scatter(xy[h, 0], xy[h, 1], s=6, alpha=0.35, c="#c0392b", label="human")
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.35, c="#27ae60", label="LLM")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
    fig.suptitle("PCA of hypersphere head output z (validation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_nu_sweep_fpr(
    nus: Sequence[float],
    fprs: Sequence[float],
    target_tpr: float,
    out_path: Path,
    dpi: int = 140,
) -> Path:
    """Operational cost vs ``nu`` for soft-boundary."""
    plt = _plt()
    out_path = Path(out_path)
    _mkdir(out_path.parent)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.plot(nus, fprs, "o-", color="#8e44ad")
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(f"FPR @ TPR={target_tpr:.2f}")
    ax.set_title("Soft-boundary: ν sweep")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_figure_bundle(
    output_dir: Path,
    val_payloads: Dict[str, Dict[str, np.ndarray]],
    roc_series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    nu_sweep: Optional[Tuple[Sequence[float], Sequence[float], float]] = None,
    pr_series: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    pca_payloads: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> List[Path]:
    """
    Write ``figures_c/`` under ``output_dir``.  Returns paths; empty list if no matplotlib.
    """
    try:
        _plt()
    except ImportError as e:
        print(f"[figures] skipped: {e}")
        return []

    base = output_dir / "figures_c"
    pr_use = roc_series if pr_series is None else pr_series
    paths: List[Path] = [
        draw_concept_oneclass_vs_softboundary(base / "concept.png"),
        draw_val_score_histograms(val_payloads, base / "val_scores.png"),
        draw_roc_overlay(roc_series, base / "roc_overlay.png"),
        draw_pr_overlay(pr_use, base / "pr_overlay.png"),
    ]
    if pca_payloads:
        paths.append(draw_pca_z_scatter(pca_payloads, base / "pca_z_val.png"))
    if nu_sweep is not None:
        nus, fprs, tpr_tgt = nu_sweep
        paths.append(draw_nu_sweep_fpr(nus, fprs, tpr_tgt, base / "nu_sweep_fpr.png"))
    return paths
