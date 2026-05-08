"""
Projection head on **frozen** Sentence-BERT embeddings.

Deep SVDD works in a learned feature space; here the SBERT trunk is fixed and we only
train a linear map ``R^{d_sbert} → R^{out_dim}``.  The hypersphere center ``c`` is a
unit vector initialized from the **mean projected LLM** embedding (see paper-style init
in ``ood-llm-detect``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HypersphereProjection(nn.Module):
    """
    ``z = W @ emb + b``; losses use squared Euclidean distance ``||z - c||^2``.

    ``c`` is a buffer (not optimized by Adam); soft-boundary uses scalar radius ``R``
    in **distance** units, updated outside autograd (quantile schedule), matching the
    reference repo’s validation scoring ``dist - R**2``.
    """

    def __init__(self, sbert_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(sbert_dim, out_dim, bias=True)
        self.register_buffer("c", torch.zeros(out_dim))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """``emb``: (B, sbert_dim) from SentenceTransformer.encode."""
        return self.proj(emb)

    @torch.no_grad()
    def init_center_from_machine(self, z_machine: torch.Tensor, eps: float = 1e-8) -> None:
        """
        Set ``c`` to L2-normalized mean of machine (LLM) projected vectors.

        Requires at least one machine row; raises if the slice is empty.
        """
        if z_machine.numel() == 0:
            raise ValueError("No machine samples to initialize c.")
        mean = z_machine.mean(dim=0)
        nrm = mean.norm(p=2).clamp_min(eps)
        self.c.copy_((mean / nrm).to(dtype=self.c.dtype, device=self.c.device))

    def squared_distances(self, z: torch.Tensor) -> torch.Tensor:
        """Per-row ``sum((z - c)^2)``, shape (B,)."""
        return ((z - self.c) ** 2).sum(dim=-1)
