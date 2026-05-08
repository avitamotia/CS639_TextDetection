"""
Data loading for Experiment C.

We use HuggingFace ``yaful/MAGE`` (same as the course EDA notebook):

  * ``label == 0`` → machine / LLM-generated text (**in-distribution** for the hypersphere).
  * ``label == 1`` → human-written text (**OOD** in the paper framing).

The training objective encourages LLM embeddings to sit **closer** to center ``c``
than human embeddings on average; soft-boundary adds a radius ``R`` on ID tails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np


@dataclass
class TextBatch:
    """One minibatch: raw strings and parallel MAGE labels (0=LLM, 1=human)."""

    texts: List[str]
    labels: np.ndarray  # int64, shape (B,)


def load_mage_rows(
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """
    Load MAGE ``text`` and ``label`` columns.

    Parameters
    ----------
    split:
        HuggingFace split, e.g. ``"train"``.
    max_samples:
        Random subset for quick debugging (None = full split).
    seed:
        RNG for subsampling only.
    """
    from datasets import load_dataset

    ds = load_dataset("yaful/MAGE", split=split)
    n = len(ds)
    indices = np.arange(n)
    if max_samples is not None and max_samples < n:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_samples, replace=False)

    texts: List[str] = []
    labels: List[int] = []
    for i in indices:
        row = ds[int(i)]
        texts.append(row["text"])
        labels.append(int(row["label"]))
    return texts, np.asarray(labels, dtype=np.int64)


def stratified_downsample(
    texts: List[str],
    labels: np.ndarray,
    max_total: int,
    seed: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Keep up to ``max_total`` rows with a balanced draw from classes 0 and 1.

    Used for held-out reporting so random subsampling does not drop a class.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = len(texts)
    if max_total <= 0 or n <= max_total:
        return texts, labels
    half = max_total // 2
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        take = min(half, int(idx.size))
        keep.extend(idx[:take].tolist())
    rng.shuffle(keep)
    return [texts[i] for i in keep], labels[np.asarray(keep, dtype=np.int64)]


def stratified_train_val_indices(
    labels: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split row indices into train / validation with both classes in validation.

    Without stratification, a small random val slice might miss one class and break metrics.
    """
    rng = np.random.default_rng(seed)
    train_parts: List[int] = []
    val_parts: List[int] = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_parts.extend(idx[:n_val].tolist())
        train_parts.extend(idx[n_val:].tolist())
    return np.asarray(train_parts, dtype=np.int64), np.asarray(val_parts, dtype=np.int64)


def batches(
    texts: Sequence[str],
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[TextBatch]:
    """Yield ``TextBatch`` objects; use a new ``seed`` each epoch for different shuffles."""
    n = len(texts)
    order = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for start in range(0, n, batch_size):
        sl = order[start : start + batch_size]
        yield TextBatch(texts=[texts[i] for i in sl], labels=labels[sl])
