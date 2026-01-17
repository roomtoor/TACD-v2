# models/clip_backbone.py
from __future__ import annotations
import torch
import torch.nn as nn
import clip  # pip install git+https://github.com/openai/CLIP.git
from typing import Tuple

class CLIPBackbone(nn.Module):
    """
    轻封装的 CLIP：冻结参数，统一暴露 encode_image / encode_text。
    - 默认 float32、eval 模式，避免 AMP/半精度带来的不一致。
    - 仅暴露视觉/文本编码接口，不包含任何训练头。
    """
    def __init__(self, name: str = "ViT-B/16", device: str = "cuda"):
        super().__init__()
        self.model, self.preprocess = clip.load(name, device=device, jit=False)
        self.model.eval()
        self.model.float()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        # 视觉侧输出维度（CLIP 统一是 visual.output_dim）
        self.out_dim: int = self.model.visual.output_dim
        # 记录输入分辨率（有时不同 CLIP 变体不一样）
        # 预处理里通常会 resize 到 224；保留这个信息方便 data pipeline
        self.image_size: int = 224

        # 提供 mean/std（和 CLIP 预处理一致），便于你自定义 transform
        self.pixel_mean = (0.48145466, 0.4578275, 0.40821073)
        self.pixel_std  = (0.26862954, 0.26130258, 0.27577711)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: 已按 CLIP mean/std 归一化的张量 [B, 3, H, W]
        return: float32 特征 [B, D]
        """
        feats = self.model.encode_image(images)
        return feats.float()

    @torch.no_grad()
    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: clip.tokenize(...) 的结果 [N, L]
        return: float32 文本特征 [N, D]
        """
        t = self.model.encode_text(token_ids)
        return t.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 不建议直接 forward，用 encode_image 更清晰
        return self.encode_image(x)
