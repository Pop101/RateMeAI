"""Project-shared regression head.

Lives here (not in MLTK) on purpose: MLTK's job is generic primitives,
``MLPRegressionHead`` is a *composition* of those primitives chosen for
this project's frozen-backbone-plus-scalar-target setting. The MLTK
toolkit ships ``ModernMLPBlock`` / ``SwiGLU`` / ``RMSNorm``; specific
compositions belong with the consumer.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from mltk.building_blocks.mlp_blocks import ModernMLPBlock, RMSNorm


class MLPRegressionHead(nn.Module):
    """Project + N residual ``ModernMLPBlock``s + final RMSNorm + Linear.

    Composition rationale:
      - Pre-norm residual blocks (RMSNorm → SwiGLU → LayerScale → DropPath
        → add) — modern transformer-style FFN; trains deep heads cleanly
        on small data.
      - Leading ``Linear(in_dim → hidden_dim)`` projection lets you reduce
        width from a wide backbone (1280 / 1024 / 768) and dodges DirectML's
        "graph-head must not be a Norm" backward compile issue.
      - Final RMSNorm before the regressor stabilizes the output scale.

    Defaults are tuned for the small-data, frozen-feature regime
    (~thousands of items, scalar regression):
      ``depth=4, hidden_dim=in_dim, drop_path=0.1, dropout=0.0``
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        *,
        hidden_dim: Optional[int] = None,
        depth: int = 4,
        drop_path: float = 0.1,
        dropout: float = 0.0,
        layer_scale_init: Optional[float] = 1e-5,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if not (0.0 <= drop_path < 1.0):
            raise ValueError(f"drop_path must be in [0, 1), got {drop_path}")
        hidden_dim = int(hidden_dim) if hidden_dim is not None else int(in_dim)

        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[
            ModernMLPBlock(
                hidden_dim,
                drop_path=drop_path * i / max(1, depth - 1),
                dropout=dropout,
                layer_scale_init=layer_scale_init,
            )
            for i in range(depth)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.out(x)
