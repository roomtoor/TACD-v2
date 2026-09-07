# cfg.py
"""
全局配置文件（Config）
--------------------
Central configuration for TASIL experiments.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict

@dataclass
class TrainConfig:
    # ====== 基础路径 ======
    dataset_root: str = "./OfficeHomeDataset"      # Office-Home 根目录
    exp_name: str = "TASIL_SSDG_GroupDRO"
    log_dir: str = "./logs"
    ckpt_dir: str = "./checkpoints"

    # ====== 数据相关 ======
    dataset_name: str = "officehome"
    img_size: int = 224
    batch_size: int = 4
    num_workers: int = 8
    source_domains: List[str] = field(default_factory=lambda: ["Art"])


    # ====== 模型结构 ======
    clip_backbone: str = "ViT-B/16"
    projector_mlp: bool = False
    init_temperature: float = 0.07
    learnable_tau: bool = True
    text_anchor_topk: Optional[int] = 29            # complete 29-descriptor style bank

    # ====== 损失系数 ======
    lambda_cls: float = 1.0     # 主分类
    lambda_cons: float = 0.3    # KL 一致性
    lambda_group: float = 0.3   # GroupDRO 组稳健性
    # 可学习 raw 参数 a 的初值；有效抑制系数 alpha_eff = sigmoid(a)。
    # 主实验从 a=0（alpha_eff=0.5）开始学习。
    alpha_style_remove: float = 0.0

    # ====== 优化器与训练 ======
    lr: float = 8e-5
    weight_decay: float = 1e-4
    epochs: int = 30
    grad_clip: float = 1.0

    # ====== 随机性控制 ======
    seed: int = 3
    deterministic: bool = True

    # ====== 日志与保存 ======
    print_interval: int = 50

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
    cfg = get_cfg({"epochs": 10, "source_domains": ["Art"]})
    print(cfg)
