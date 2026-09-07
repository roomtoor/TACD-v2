# models/tasil.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Dict

from .clip_backbone import CLIPBackbone
from .projector import SemanticProjector
from .classifier import CosineClassifier


def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class TASIL(nn.Module):
    """
    Text-Anchored Style Invariance Learning (TASIL).
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
        # 每一行对应一个具体风格描述符；仅训练时构造 appearance view。
        # Reconstructed from checkpoint metadata rather than stored in state_dict.
        self.register_buffer("style_embeddings", torch.empty(0, D), persistent=False)

        # 支持构造时注入
        if E_s is not None:
            self.set_style_subspace(E_s)
        if T is not None:
            self.set_class_texts(T)

    def train(self, mode: bool = True):
        """Keep the frozen CLIP encoder in evaluation mode during head training."""
        super().train(mode)
        self.backbone.model.eval()
        return self

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

    @torch.no_grad()
    def set_style_embeddings(self, style_embeddings: torch.Tensor):
        """缓存逐描述符的文本特征，输入形状为 [K, D]。"""
        assert style_embeddings.dim() == 2 and style_embeddings.size(1) == self.backbone.out_dim, \
            (f"style_embeddings should be [K,D] with D={self.backbone.out_dim}, "
             f"got {tuple(style_embeddings.shape)}")
        if style_embeddings.size(0) == 0:
            raise ValueError("style_embeddings must contain at least one descriptor.")
        self.style_embeddings = l2n(style_embeddings.to(self.device), dim=1)

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
    def _project_and_suppress(self, clip_features: torch.Tensor):
        """共享的 projection -> style suppression 路径。"""
        f = self.projector(clip_features)
        if self.Q.numel() == 0:
            f_proj = torch.zeros_like(f)
            f_clean = f
        else:
            f_proj = (f @ self.Q) @ self.Q.transpose(0, 1)
            a = torch.sigmoid(self.alpha)
            f_clean = f - a * f_proj
        return f, f_proj, f_clean

    def _classify_clean(self, f_clean: torch.Tensor) -> torch.Tensor:
        assert self.T.numel() > 0, "Class text embeddings T is not set."
        feats_n = l2n(f_clean, dim=1)
        if not torch.isfinite(f_clean).all():
            raise RuntimeError("[TASIL] feats (f_clean) non-finite!")
        if not torch.isfinite(self.T).all():
            raise RuntimeError("[TASIL] class anchors T non-finite!")
        logits = self.classifier(feats_n, self.T)
        if not torch.isfinite(logits).all():
            raise RuntimeError("[TASIL] logits contains non-finite values.")
        return logits

    def forward_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """标准图像视图：CLIP 编码、归一化、共享投影与风格抑制。"""
        f0 = self.encode_image(images)                      # [B, D], no grad
        v = l2n(f0, dim=1)
        f, f_proj, f_clean = self._project_and_suppress(v)
        return {"f0": f0, "v": v, "f": f, "f_proj": f_proj, "f_clean": f_clean}

    def forward_appearance_features(
        self,
        images: torch.Tensor,
        style_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Construct the appearance view:
          v_ap = Norm(Norm(CLIP_image(x)) + e_j), j ~ Uniform({1,...,K})
        随后进入与其他视图完全共享的 projection 和 style suppression。
        """
        if self.style_embeddings.numel() == 0:
            raise RuntimeError("Style embeddings are not set for appearance augmentation.")

        f0 = self.encode_image(images)                      # 原图 CLIP 特征
        v_img = l2n(f0, dim=1)
        batch_size = v_img.size(0)
        num_styles = self.style_embeddings.size(0)

        if style_indices is None:
            style_indices = torch.randint(
                low=0, high=num_styles, size=(batch_size,), device=v_img.device
            )
        else:
            style_indices = style_indices.to(v_img.device, dtype=torch.long)
            if style_indices.shape != (batch_size,):
                raise ValueError(
                    f"style_indices should have shape ({batch_size},), "
                    f"got {tuple(style_indices.shape)}"
                )
            if style_indices.min() < 0 or style_indices.max() >= num_styles:
                raise ValueError("style_indices contains an out-of-range descriptor index.")

        e_j = self.style_embeddings.index_select(0, style_indices)
        v_ap = l2n(v_img + e_j, dim=1)
        f, f_proj, f_clean = self._project_and_suppress(v_ap)
        return {
            "f0": f0,
            "v_img": v_img,
            "style_indices": style_indices,
            "e_style": e_j,
            "v_ap": v_ap,
            "f": f,
            "f_proj": f_proj,
            "f_clean": f_clean,
        }

    # ------------------- 标准前向（输出 logits） -------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self._classify_clean(self.forward_features(images)["f_clean"])

    def forward_appearance(
        self,
        images: torch.Tensor,
        style_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = self.forward_appearance_features(images, style_indices=style_indices)
        return self._classify_clean(feats["f_clean"])

    # ------------------- 调试接口：返回中间量 -------------------
    @torch.no_grad()
    def debug_forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        f0 = self.encode_image(images).float()
        v = l2n(f0, dim=1)
        f, f_proj, f_clean = self._project_and_suppress(v)

        feats_n = l2n(f_clean, dim=1)
        T_n     = self.T
        if T_n.numel() > 0:
            logits = feats_n @ T_n.transpose(0, 1)
        else:
            logits = torch.zeros(f.size(0), 1, device=f.device)

        return {
            "f0": f0, "v": v, "f": f, "f_proj": f_proj, "f_clean": f_clean,
            "feats_n": feats_n, "Q": self.Q, "T": self.T, "logits_unit_tau": logits
        }
