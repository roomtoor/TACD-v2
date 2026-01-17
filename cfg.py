# cfg.py
"""
全局配置文件（Config）
--------------------
集中管理 TACD-v2 的超参数、路径、模型结构和实验选项。
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict

@dataclass
class TrainConfig:
    # ====== 基础路径 ======
    dataset_root: str = "./OfficeHomeDataset"      # Office-Home 根目录
    exp_name: str = "TACDv2_SSDG_GroupDRO"
    log_dir: str = "./logs"
    ckpt_dir: str = "./checkpoints"

    # ====== 数据相关 ======
    dataset_name: str = "terraincognita"   # ["officehome", "terraincognita"]
    img_size: int = 224
    batch_size: int = 4
    num_workers: int = 8
    source_domains: List[str] = field(default_factory=lambda: ["Art"])
    target_domains: List[str] = field(default_factory=lambda: ["Real World", "Clipart", "Product"])
    num_classes: int = 0
    shuffle: bool = True


    # ====== 模型结构 ======
    clip_backbone: str = "ViT-B/16"
    freeze_clip: bool = True
    projector_dim: int = 512                        # 语义空间维度
    use_lora: bool = False                          
    text_anchor_topk: Optional[int] = 30            # 选取前多少风格词构建 E_s

    # ====== 损失系数 ======
    lambda_cls: float = 1.0     # 主分类
    lambda_cons: float = 0.3    # KL 一致性
    lambda_group: float = 0.3   # GroupDRO 组稳健性
    alpha_style_remove: float =-30  # 风格剔除强度 f_clean = f - αE_s(E_s^T f)，带有sigmoid

    # ====== 优化器与训练 ======
    lr: float = 8e-5
    weight_decay: float = 1e-4
    epochs: int = 30
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    use_amp: bool = True   # 混合精度

    # ====== 随机性控制 ======
    seed: int = 3
    deterministic: bool = True

    # ====== 日志与保存 ======
    save_interval: int = 1
    print_interval: int = 50
    eval_interval: int = 1

    def as_dict(self) -> Dict:
        return asdict(self)

# ---------------------------------------------------------------------
#  构建器函数
# ---------------------------------------------------------------------
def get_cfg(overrides: Optional[Dict] = None) -> TrainConfig:
    """
    用于 run_train.py：
      cfg = get_cfg()
      print(cfg.batch_size)
    或者：
      cfg = get_cfg({'epochs': 50, 'lr': 1e-4})
    """
    cfg = TrainConfig()
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                print(f"[WARN] Unknown cfg key: {k}")
    return cfg

# ---------------------------------------------------------------------
#  Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    cfg = get_cfg({"epochs": 10, "source_domains": ["Art", "Product", "Real World"]})
    print(cfg)
