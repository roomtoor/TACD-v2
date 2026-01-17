# models/projector.py
from __future__ import annotations
import torch.nn as nn

class SemanticProjector(nn.Module):
    """
    语义投影头：
      - 默认线性层即可；可选小 MLP（带 LayerNorm）更稳。
    """
    def __init__(self, in_dim: int = 512, out_dim: int = 512, mlp: bool = False):
        super().__init__()
        if mlp:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 1024),
                nn.GELU(),
                nn.Linear(1024, out_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim)
            )

    def forward(self, x):
        return self.net(x)
