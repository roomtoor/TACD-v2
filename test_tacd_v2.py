# test_tacd_v2.py
from __future__ import annotations

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfg import get_cfg
from utils import set_seed, accuracy, Logger
from data import get_weak_transform
from models import TACDv2
from textspace import build_style_subspace, build_class_texts


def _to_xy(batch):
    """宽松解包：支持 (x,y)、(x,y,meta) 或 dict。"""
    if isinstance(batch, dict):
        # 兼容未来你把 val 也做成 dict 的情况
        if "x" in batch:
            return batch["x"], batch["y"]
        if "x_w" in batch:
            return batch["x_w"], batch["y"]
        raise KeyError(f"Unexpected dict keys: {list(batch.keys())}")

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        return batch[0], batch[1]

    raise TypeError(f"Unexpected batch type: {type(batch)}")


def resolve_dataset(dataset_name: str):
    """
    返回:
      BaseDataset: OfficeHomeDataset / TerraIncognitaDataset
      scan_classes: 扫全域交集类表
      to_prompts: 目录类名 -> prompt name
      all_domains: 域列表
    """
    ds = dataset_name.lower()
    if ds in ["officehome", "office-home", "oh"]:
        from data import OfficeHomeDataset as BaseDataset
        from data import scan_officehome_classes as scan_classes
        from data import officehome_prompt_names as to_prompts
        all_domains = ["Art", "Clipart", "Product", "Real World"]
        return BaseDataset, scan_classes, to_prompts, all_domains

    if ds in ["terraincognita", "terra", "ti"]:
        from data import TerraIncognitaDataset as BaseDataset
        from data import scan_terraincognita_classes as scan_classes

        def to_prompts(names):
            # Terra: 类目录一般就是自然词/下划线形式
            return [c.replace("_", " ").lower() for c in names]

        all_domains = ["location_38", "location_43", "location_46", "location_100"]
        return BaseDataset, scan_classes, to_prompts, all_domains

    raise ValueError(f"Unsupported dataset={dataset_name}")


def make_loader(BaseDataset, root, domain, class_names, img_size, batch_size, workers):
    t_weak = get_weak_transform(img_size)
    ds = BaseDataset(
        root=root,
        domain=domain,
        transform=t_weak,
        class_names=class_names,   # 关键：统一 label space
        return_pil=False
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )


@torch.no_grad()
def eval_domain(model, loader, num_classes: int, per_class: bool = False):
    """
    返回:
      - acc1: 样本加权的 top-1 accuracy
      - per_class_acc (optional)
    """
    model.eval()
    dev = next(model.parameters()).device

    total = 0
    correct_total = 0

    if per_class:
        correct = np.zeros(num_classes, dtype=np.int64)
        count = np.zeros(num_classes, dtype=np.int64)

    for batch in loader:
        x, y = _to_xy(batch)
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(dim=1)

        total += y.numel()
        correct_total += (pred == y).sum().item()

        if per_class:
            y_np = y.cpu().numpy()
            p_np = pred.cpu().numpy()
            for yi, pi in zip(y_np, p_np):
                if 0 <= yi < num_classes:
                    count[yi] += 1
                    if yi == pi:
                        correct[yi] += 1

    acc1 = 100.0 * (correct_total / max(total, 1))

    res = {"acc1": float(acc1)}
    if per_class:
        with np.errstate(divide="ignore", invalid="ignore"):
            pc = np.where(count > 0, correct / count, 0.0)
        res["per_class_acc"] = pc
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="officehome",
                    choices=["officehome", "terraincognita"],
                    help="dataset name")

    ap.add_argument("--root", type=str, default=None, help="Dataset root (override cfg)")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pth). If omitted, auto-pick latest")

    ap.add_argument("--source", type=str, default=None,
                    help='单源域名称（若提供，则默认只评测除该域外的所有域）')
    ap.add_argument("--domains", type=str, nargs="*", default=None,
                    help='显式指定评测域（优先级高于 --source）')

    ap.add_argument("--per_class", action="store_true", help="输出每类准确率并保存为 .npy")
    args = ap.parse_args()

    # ---- cfg & env ----
    cfg_over = {}
    if args.root:
        cfg_over["dataset_root"] = args.root
    cfg_over["dataset_name"] = args.dataset.lower()
    cfg = get_cfg(cfg_over)

    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = Logger(cfg.log_dir, cfg.exp_name + f"_TEST_{args.dataset.lower()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed, cfg.deterministic)

    # ---- dataset resolver ----
    BaseDataset, scan_classes, to_prompts, ALL_DOMAINS = resolve_dataset(args.dataset)

    # ---- classes & text anchors ----
    class_names = scan_classes(cfg.dataset_root)      # 全域交集类表
    class_prompt_names = to_prompts(class_names)
    num_classes = len(class_names)

    # ---- build model (same as train) ----
    model = TACDv2(
        clip_name=cfg.clip_backbone,
        device=device,
        projector_mlp=getattr(cfg, "projector_mlp", False),
        alpha=getattr(cfg, "alpha_style_remove", 0.7),
        temperature=getattr(cfg, "init_temperature", 0.07),
        learnable_tau=getattr(cfg, "learnable_tau", True),
    ).to(device)

    E_s = build_style_subspace(
        model.backbone.model,
        device=device,
        k=getattr(cfg, "text_anchor_topk", 8),
        use_qr=True
    )
    T = build_class_texts(model.backbone.model, class_prompt_names, device=device)
    model.set_style_subspace(E_s)
    model.set_class_texts(T)

    # ---- load checkpoint safely ----
    ckpt_path = args.ckpt
    if ckpt_path is None:
        pdir = cfg.ckpt_dir
        if not os.path.isdir(pdir):
            raise FileNotFoundError(f"Checkpoint dir not found: {pdir}. Provide --ckpt.")
        cand = [os.path.join(pdir, f) for f in os.listdir(pdir) if f.endswith(".pth")]
        if not cand:
            raise FileNotFoundError(f"No checkpoint found in {pdir}. Provide --ckpt.")
        ckpt_path = max(cand, key=os.path.getmtime)

    print(f"[Load] checkpoint: {ckpt_path}")
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # PyTorch>=2.4
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=True)

    # ---- decide eval domains ----
    if args.domains:
        domains = args.domains
    elif args.source:
        domains = [d for d in ALL_DOMAINS if d != args.source]
    else:
        domains = ALL_DOMAINS

    # ---- evaluate ----
    results = {}
    worst = 1e9

    for dom in domains:
        loader = make_loader(
            BaseDataset,
            cfg.dataset_root,
            dom,
            class_names,
            cfg.img_size,
            cfg.batch_size,
            cfg.num_workers
        )
        res = eval_domain(model, loader, num_classes, per_class=args.per_class)
        results[dom] = res["acc1"]

        logger.write(f"[Test] {dom}: acc@1={res['acc1']:.2f}")
        if args.per_class:
            out_npy = os.path.join(cfg.log_dir, f"per_class_{args.dataset.lower()}_{dom.replace(' ','_')}.npy")
            np.save(out_npy, res["per_class_acc"])

        worst = min(worst, res["acc1"])

    mean_acc = sum(results.values()) / max(1, len(results))
    logger.write(f"[Test] Mean acc@1={mean_acc:.2f}")
    logger.write(f"[Test] Worst-domain acc@1={worst:.2f}")

    # ---- summary ----
    print("== Summary ==")
    for k, v in results.items():
        print(f"{k:>14}: {v:.2f}")
    print(f"{'Mean':>14}: {mean_acc:.2f}")
    print(f"{'Worst':>14}: {worst:.2f}")


if __name__ == "__main__":
    main()
