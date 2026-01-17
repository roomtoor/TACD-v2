# models/classifier.py
from __future__ import annotations
import torch
import torch.nn as nn

def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

class CosineClassifier(nn.Module):
    """
    余弦分类器（稳健版）：
      - 始终对输入特征 x 和权重 W 做 L2 归一化（带 eps）
      - 温度 tau 被夹在 [1e-2, 10] 之间，避免除以 ~0 或极端放大
      - 可学习温度：log_tau 为参数
    """
    def __init__(self, temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        if learnable:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(float(temperature))))
            self.register_buffer("tau_const", torch.tensor(0.0))
        else:
            self.log_tau = None
            self.register_buffer("tau_const", torch.tensor(float(temperature)))

    def _tau(self) -> torch.Tensor:
        if self.log_tau is not None:
            return torch.exp(self.log_tau).clamp(1e-2, 10.0)
        return self.tau_const.clamp(1e-2, 10.0)

    def forward(self, feats: torch.Tensor, class_text_embeds: torch.Tensor) -> torch.Tensor:
        # feats: [B, D], class_text_embeds: [C, D]
        f = l2n(feats, dim=1)
        T = l2n(class_text_embeds, dim=1)
        tau = self._tau()
        logits = (f @ T.transpose(0, 1)) / tau
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"[CosineClassifier] non-finite logits; "
                f"tau={float(tau.detach())}, f_nan={bool(torch.isnan(f).any())}, T_nan={bool(torch.isnan(T).any())}"
            )
        return logits
