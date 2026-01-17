# cfg_baseline.py
"""
Baseline config:
只使用 CLIP 图像特征 + 线性分类头，
无风格剔除、无一致性、无 GroupDRO。
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

@dataclass
class BaselineConfig:
    # ====== 路径 ======
    dataset_root: str = "./OfficeHomeDataset"
    exp_name: str = "Baseline_OfficeHome"
    log_dir: str = "./logs"
    ckpt_dir: str = "./checkpoints"

    # ====== 数据 ======
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 8
    source_domains: List[str] = ("Product", )
    target_domains: List[str] = ("Real World","Clipart", "Art")
    num_classes: int = 65

    # ====== 模型 ======
    clip_backbone: str = "ViT-B/16"
    freeze_clip: bool = True
    projector_mlp: bool = False
    learnable_tau: bool = False
    init_temperature: float = 0.07
    alpha_style_remove: float = 0.0
    text_anchor_topk: int = 0

    # ====== 损失/优化 ======
    lambda_cls: float = 1.0
    lambda_cons: float = 0.0
    lambda_group: float = 0.0
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    grad_clip: float = 1.0

    # ====== 运行控制 ======
    seed: int = 3
    deterministic: bool = True
    print_interval: int = 50
    eval_interval: int = 1

    def as_dict(self) -> Dict:
        return asdict(self)


def get_cfg_baseline(overrides: Optional[Dict] = None) -> BaselineConfig:
    cfg = BaselineConfig()
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                print(f"[WARN] Unknown cfg key: {k}")
    return cfg
