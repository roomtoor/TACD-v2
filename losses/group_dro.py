# losses/group_dro.py
import torch
import torch.nn as nn

class GroupDRO(nn.Module):
    """
    Group Distributionally Robust Optimization (GroupDRO)

    功能：
      - 跟踪每个组（域或增强类型）的平均损失；
      - 自适应更新权重 q_g；
      - 最终损失 = ∑_g q_g * L_g，重点优化最坏组。
    """
    def __init__(self, num_groups: int, eta: float = 0.01, device: str = "cuda"):
        super().__init__()
        self.num_groups = num_groups
        self.eta = eta
        self.register_buffer("q", torch.ones(num_groups, device=device) / num_groups)

    @torch.no_grad()
    def update(self, group_losses: torch.Tensor):
        """
        根据各组的当前损失更新权重。
        group_losses: [G]
        """
        assert group_losses.numel() == self.num_groups
        self.q *= torch.exp(self.eta * group_losses.detach())
        self.q /= self.q.sum()  # 归一化到 simplex

    def forward(self, group_losses: torch.Tensor) -> torch.Tensor:
        """
        group_losses: [G]
        return: 加权后的 DRO 损失 (标量)
        """
        assert group_losses.numel() == self.num_groups
        return (self.q * group_losses).sum()
