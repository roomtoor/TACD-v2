# losses/core.py
import torch
import torch.nn.functional as F

def cosine_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    CrossEntropy over cosine logits.
    logits: [B, C]
    targets: [B]
    """
    return F.cross_entropy(logits, targets)

def symmetric_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """
    Symmetric KL divergence between two prediction distributions.
    用于一致性约束 (weak vs strong augmentations)。
    """
    p_log = F.log_softmax(p_logits, dim=-1)
    q_log = F.log_softmax(q_logits, dim=-1)
    p_prob = p_log.exp()
    q_prob = q_log.exp()

    # 双向 KL
    kl1 = F.kl_div(p_log, q_prob, reduction="batchmean")
    kl2 = F.kl_div(q_log, p_prob, reduction="batchmean")
    return kl1 + kl2
