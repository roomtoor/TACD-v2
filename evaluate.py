# evaluate.py
from __future__ import annotations

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from cfg import get_cfg
from utils import set_seed, Logger
from data import get_base_transform
from models import TASIL
from textspace import build_style_subspace, build_class_texts


def _to_xy(batch):
    """宽松解包：支持 (x,y)、(x,y,meta) 或 dict。"""
    if isinstance(batch, dict):
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
      scan_source_classes: scan the class list from the specified source
      to_prompts: 目录类名 -> prompt name
      all_domains: 域列表
      normalize_domain: 命令行域名称 -> 规范域名称
    """
    ds = dataset_name.lower()
    if ds in ["officehome", "office-home", "oh"]:
        from data import OfficeHomeDataset as BaseDataset
        from data import scan_officehome_source_classes as scan_source_classes
        from data import officehome_prompt_names as to_prompts
        all_domains = ["Art", "Clipart", "Product", "Real World"]
        return BaseDataset, scan_source_classes, to_prompts, all_domains, lambda d: d

    if ds in ["terraincognita", "terra", "ti"]:
        from data import TerraIncognitaDataset as BaseDataset
        from data import scan_terraincognita_source_classes as scan_source_classes

        def to_prompts(names):
            # Terra: 类目录一般就是自然词/下划线形式
            return [c.replace("_", " ").lower() for c in names]

        all_domains = ["location_38", "location_43", "location_46", "location_100"]
        return BaseDataset, scan_source_classes, to_prompts, all_domains, lambda d: d

    if ds in ["domainnet", "domain-net", "dn"]:
        from data import DomainNetDataset as BaseDataset
        from data import scan_domainnet_source_classes as scan_source_classes
        from data import domainnet_prompt_names as to_prompts
        from data import normalize_domainnet_domain as normalize_domain

        all_domains = ["clip", "info", "paint", "quick", "real", "sketch"]
        return BaseDataset, scan_source_classes, to_prompts, all_domains, normalize_domain

    if ds in ["vlcs"]:
        from data import VLCSDataset as BaseDataset
        from data import scan_vlcs_source_classes as scan_source_classes
        from data import vlcs_prompt_names as to_prompts
        from data import normalize_vlcs_domain as normalize_domain

        all_domains = ["C", "L", "S", "V"]
        return BaseDataset, scan_source_classes, to_prompts, all_domains, normalize_domain

    raise ValueError(f"Unsupported dataset={dataset_name}")


def make_loader(BaseDataset, root, domain, class_names, img_size, batch_size, workers):
    # Evaluation must be deterministic. Keep RandomHorizontalFlip and
    # ColorJitter in the training-only weak transform, but do not use them here.
    t_test = get_base_transform(img_size)
    ds = BaseDataset(
        root=root,
        domain=domain,
        transform=t_test,
        class_names=class_names,
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
                    choices=["officehome", "terraincognita", "domainnet", "vlcs"],
                    help="dataset name")

    ap.add_argument("--root", type=str, default=None, help="Dataset root (override cfg)")
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="explicit path to the fixed final-epoch checkpoint (.pth)",
    )

    ap.add_argument("--source", type=str, default=None,
                    help="source domain; restored from checkpoint metadata when available")
    ap.add_argument("--domains", type=str, nargs="*", default=None,
                    help="explicit held-out evaluation domains")

    ap.add_argument("--per_class", action="store_true",
                    help="save per-class accuracy as .npy")
    ap.add_argument("--seed", type=int, default=None, help="training/evaluation seed")
    args = ap.parse_args()

    # ---- load one explicit checkpoint before reconstructing its label/text space ----
    ckpt_path = args.ckpt
    print(f"[Load] checkpoint: {ckpt_path}")
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
    model_state = (
        state["model"]
        if isinstance(state, dict) and "model" in state
        else state
    )

    checkpoint_dataset = metadata.get("dataset_name")
    if checkpoint_dataset and checkpoint_dataset.lower() != args.dataset.lower():
        raise ValueError(
            f"Checkpoint dataset={checkpoint_dataset} does not match --dataset={args.dataset}."
        )

    # ---- cfg & env ----
    cfg_over = {"dataset_name": args.dataset.lower()}
    if args.root:
        cfg_over["dataset_root"] = args.root
    metadata_seed = metadata.get("seed")
    if args.seed is not None:
        cfg_over["seed"] = args.seed
    elif metadata_seed is not None:
        cfg_over["seed"] = int(metadata_seed)
    if metadata.get("clip_backbone"):
        cfg_over["clip_backbone"] = metadata["clip_backbone"]
    cfg = get_cfg(cfg_over)

    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = Logger(
        cfg.log_dir,
        cfg.exp_name + f"_TEST_{args.dataset.lower()}_seed{cfg.seed}",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed, cfg.deterministic)

    # ---- dataset resolver and source identity ----
    (
        BaseDataset,
        scan_source_classes,
        to_prompts,
        ALL_DOMAINS,
        normalize_domain,
    ) = resolve_dataset(args.dataset)

    metadata_source = metadata.get("source_domain")
    source_value = args.source or metadata_source
    if source_value is None:
        raise ValueError(
            "The checkpoint has no source-domain metadata; provide --source explicitly."
        )
    source_domain = normalize_domain(source_value)
    if source_domain not in ALL_DOMAINS:
        raise ValueError(f"Unknown source domain {source_value}; expected one of {ALL_DOMAINS}")
    if args.source and metadata_source:
        checkpoint_source = normalize_domain(metadata_source)
        if checkpoint_source != source_domain:
            raise ValueError(
                f"Checkpoint source={checkpoint_source} does not match --source={source_domain}."
            )

    # Reuse the exact source-derived label mapping stored during training.
    # Checkpoints without metadata reconstruct it from the specified source.
    class_names = metadata.get("class_names")
    if class_names:
        class_names = list(class_names)
        class_origin = "checkpoint metadata"
    else:
        class_names = scan_source_classes(cfg.dataset_root, source_domain)
        class_origin = "source-domain directory (checkpoint without metadata)"
    class_prompt_names = to_prompts(class_names)
    num_classes = len(class_names)
    logger.write(
        f"[Classes] origin={class_origin} | source={source_domain} | "
        f"count={num_classes} | names={class_names}"
    )

    # ---- build model using the checkpoint's frozen text geometry ----
    model = TASIL(
        clip_name=cfg.clip_backbone,
        device=device,
        projector_mlp=getattr(cfg, "projector_mlp", False),
        alpha=getattr(cfg, "alpha_style_remove", 0.7),
        temperature=getattr(cfg, "init_temperature", 0.07),
        learnable_tau=getattr(cfg, "learnable_tau", True),
    ).to(device)

    suppression_words = metadata.get("suppression_descriptors")
    E_s = build_style_subspace(
        model.backbone.model,
        device=device,
        k=None if suppression_words else getattr(cfg, "text_anchor_topk", 8),
        use_qr=True,
        style_words=suppression_words,
    )
    T = build_class_texts(model.backbone.model, class_prompt_names, device=device)
    model.set_style_subspace(E_s)
    model.set_class_texts(T)
    model.load_state_dict(model_state, strict=True)

    # ---- decide held-out evaluation domains ----
    if args.domains:
        domains = [normalize_domain(d) for d in args.domains]
    else:
        domains = [d for d in ALL_DOMAINS if d != source_domain]
    invalid_domains = [d for d in domains if d not in ALL_DOMAINS]
    if invalid_domains:
        raise ValueError(
            f"Unknown evaluation domains {invalid_domains}; expected a subset of {ALL_DOMAINS}."
        )
    if not domains:
        raise ValueError("At least one evaluation domain is required.")

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
