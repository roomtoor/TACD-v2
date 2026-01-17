# models/tacd_v2.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .clip_backbone import CLIPBackbone
from .projector import SemanticProjector
from .classifier import CosineClassifier


def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class TACDv2(nn.Module):
    """
    TACD-v2（稳健版）：
      - 冻结 CLIP backbone
      - 语义投影头 P_sem
      - 文本锚定风格子空间剔除：f_clean = f − σ(α)·Proj_Q(f)，其中 Q 来自对 E_s 列正交化（QR）
      - 余弦分类头（与类文本锚 T 对齐），所有向量均 L2 归一化并带 eps 防护
    说明：
      - 外部传入的 E_s: [D, K] 会被 L2N(+QR) 成为正交基 Q: [D, r] 并缓存为 buffer
      - 外部传入的 T  : [C, D] 会被逐行 L2N 并缓存为 buffer
    """
    def __init__(
        self,
        clip_name: str = "ViT-B/16",
        device: str = "cuda",
        projector_mlp: bool = False,
        alpha: float = 0.75,
        temperature: float = 0.07,
        learnable_tau: bool = True,
        E_s: Optional[torch.Tensor] = None,  # [D, K]
        T:   Optional[torch.Tensor] = None   # [C, D]
    ):
        super().__init__()
        self.device = device
        self.backbone = CLIPBackbone(name=clip_name, device=device)
        D = self.backbone.out_dim

        # 冻结 backbone（只训头部）
        self.backbone.model.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.projector = SemanticProjector(in_dim=D, out_dim=D, mlp=projector_mlp)

        # 分类器保留，但我们会在调用前把输入/锚都 L2N 过
        self.classifier = CosineClassifier(temperature=temperature, learnable=learnable_tau)

        # alpha 设为可学习参数，并用 sigmoid 约束到 (0,1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))

        # 缓存正交基 Q 和 L2N 过的 T（作为 buffer，不反传）
        self.register_buffer("Q", torch.empty(D, 0))      # [D, r], r 可能小于 K
        self.register_buffer("T", torch.empty(0, D))      # [C, D]

        # 支持构造时注入
        if E_s is not None:
            self.set_style_subspace(E_s)
        if T is not None:
            self.set_class_texts(T)

    # --------- 公共方法：控制 alpha ----------
    @torch.no_grad()
    def set_alpha(self, a: float):
        self.alpha.fill_(float(a))

    # --------- 公共方法：注入/替换风格子空间 ----------
    @torch.no_grad()
    def set_style_subspace(self, E_s: torch.Tensor):
        """
        E_s: [D, K]（列为风格方向）。会进行列 L2 归一化，再 QR 正交化，存为 Q: [D, r]
        """
        assert E_s.dim() == 2 and E_s.size(0) == self.backbone.out_dim, \
            f"E_s shape should be [D,K] with D={self.backbone.out_dim}, got {tuple(E_s.shape)}"
        if E_s.numel() == 0 or E_s.size(1) == 0:
            self.Q = torch.empty(self.backbone.out_dim, 0, device=self.device)
            return
        E_s = l2n(E_s.to(self.device), dim=0)
        Q, _ = torch.linalg.qr(E_s, mode="reduced")  # 正交列空间
        self.Q = Q  # [D, r]

    # --------- 公共方法：注入/替换类别文本锚 ----------
    @torch.no_grad()
    def set_class_texts(self, T: torch.Tensor):
        """
        T: [C, D]。逐行 L2 归一化后缓存。
        """
        assert T.dim() == 2 and T.size(1) == self.backbone.out_dim, \
            f"T shape should be [C,D] with D={self.backbone.out_dim}, got {tuple(T.shape)}"
        self.T = l2n(T.to(self.device), dim=1)

    # ------------------- 编码 -------------------
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # 保证为 float32
        return self.backbone.encode_image(images).float()

    @torch.no_grad()
    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_text(token_ids).float()

    # ------------------- 前向（分步） -------------------
    def forward_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        返回中间特征，便于日志/可视化：
          f0:   冻结 CLIP img 特征（no grad）
          f:    projector 之后（可学习）
          f_clean: 风格剔除后
        """
        f0 = self.encode_image(images)                      # [B, D], no grad
        f  = self.projector(f0)                             # [B, D]
        if self.Q.numel() == 0:
            f_clean = f
        else:
            # 正交投影：Proj_Q(f) = (f @ Q) @ Q^T
            f_proj  = (f @ self.Q) @ self.Q.transpose(0, 1)
            a       = torch.sigmoid(self.alpha)             # (0,1)
            f_clean = f - a * f_proj
        return {"f0": f0, "f": f, "f_clean": f_clean}

    # ------------------- 标准前向（输出 logits） -------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(images)["f_clean"]    # [B, D]
        assert self.T.numel() > 0, "Class text embeddings T is not set."

        # 统一在这里 L2N，避免 0/0 或极端范数带来的数值不稳
        feats_n = l2n(feats, dim=1)                         # [B, D]
        T_n     = self.T                                     # 已经是 L2N 过的 [C, D]

        # ========== 关键断言：告诉我们究竟是谁先 NaN ==========
        if not torch.isfinite(feats).all():
            raise RuntimeError("[TACDv2] feats (f_clean) non-finite!")
        if not torch.isfinite(self.T).all():
            raise RuntimeError("[TACDv2] class anchors T non-finite!")

        # 如果外部分类器内部也做 L2N，不会有坏处；这里再传入归一化后的张量
        logits = self.classifier(feats_n, T_n)              # [B, C]
        # 额外的安全检查（训练时开销很小）
        if not torch.isfinite(logits).all():
            raise RuntimeError("[TACDv2] logits contains non-finite values.")
        return logits

    # ------------------- 调试接口：返回中间量 -------------------
    @torch.no_grad()
    def debug_forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        f0 = self.encode_image(images).float()
        f  = self.projector(f0).float()
        if self.Q.numel() == 0:
            f_proj = torch.zeros_like(f)
            f_clean = f
        else:
            f_proj  = (f @ self.Q) @ self.Q.transpose(0, 1)
            a       = torch.sigmoid(self.alpha)
            f_clean = f - a * f_proj

        feats_n = l2n(f_clean, dim=1)
        T_n     = self.T
        if T_n.numel() > 0:
            logits = feats_n @ T_n.transpose(0, 1)
        else:
            logits = torch.zeros(f.size(0), 1, device=f.device)

        return {
            "f0": f0, "f": f, "f_proj": f_proj, "f_clean": f_clean,
            "feats_n": feats_n, "Q": self.Q, "T": self.T, "logits_unit_tau": logits
        }
