# textspace/text_embed.py
import torch
import torch.nn.functional as F
import clip
from textspace.style_bank import DEFAULT_STYLE_WORDS
from typing import List, Optional

@torch.no_grad()
def build_style_subspace(
    clip_model,
    device: str = "cuda",
    k: Optional[int] = None,
    use_qr: bool = True,
    extra_words: Optional[List[str]] = None
) -> torch.Tensor:
    """
    构建文本锚定的风格子空间 E_s ∈ R^{d×k}
    ------------------------------------------------
    Args:
        clip_model : 已加载的 CLIP 模型（可来自 TACDv2.backbone.model）
        device     : "cuda" or "cpu"
        k          : 使用前 k 个风格词
        use_qr     : 是否正交化列向量（推荐）
        extra_words: 附加的风格词列表
    Returns:
        E_s : torch.Tensor [d, k]
    """
    styles = list(DEFAULT_STYLE_WORDS)
    if extra_words:
        styles.extend(extra_words)
    if k:
        styles = styles[:k]

    # tokenize & encode
    tokens = torch.cat([clip.tokenize(s).to(device) for s in styles], dim=0)
    tfeat = clip_model.encode_text(tokens).float()   # [k, d]
    tfeat = F.normalize(tfeat, dim=-1)               # 行向量单位化

    # 转置成 [d, k]
    E = tfeat.t().contiguous()
    if use_qr:
        Q, _ = torch.linalg.qr(E)                    # 正交化
        E = Q[:, :E.shape[1]]
    return E


@torch.no_grad()
def build_class_texts(
    clip_model,
    classnames: List[str],
    device: str = "cuda",
    templates: Optional[List[str]] = None
) -> torch.Tensor:
    """
    生成类别文本锚 T ∈ R^{C×d}
    ------------------------------------------------
    Args:
        clip_model : CLIP 模型
        classnames : 类别名称列表
        templates  : 文本模板，可多样化
    Returns:
        T : torch.Tensor [C, d]
    """
    if templates is None:
        templates = [
            "a photo of a {}.",
            "a sketch of a {}.",
            "a cartoon of a {}.",
            "a drawing of a {}.",
            "an image of a {}."
        ]

    all_embeds = []
    for cname in classnames:
        tokens = torch.cat([clip.tokenize(t.format(cname)).to(device) for t in templates], dim=0)
        text_features = clip_model.encode_text(tokens).float()  # [n_templates, d]
        text_features = F.normalize(text_features, dim=-1)
        mean_feat = text_features.mean(dim=0, keepdim=True)
        mean_feat = F.normalize(mean_feat, dim=-1)
        all_embeds.append(mean_feat)

    T = torch.cat(all_embeds, dim=0)  # [C, d]
    return T

