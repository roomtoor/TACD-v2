# test_baseline.py
from __future__ import annotations
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cfg_baseline import get_cfg_baseline
from utils import set_seed, accuracy, Logger, save_checkpoint
from data import (
    OfficeHomeDataset,
    scan_officehome_classes,
    get_weak_transform,
)
from models.clip_backbone import CLIPBackbone

ALL_DOMAINS = ["Art", "Clipart", "Product", "Real World"]


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class LinearProbe(nn.Module):
    """冻结 CLIP，只训练线性分类头"""
    def __init__(self, clip_name: str, num_classes: int, device: str):
        super().__init__()
        self.backbone = CLIPBackbone(name=clip_name, device=device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        d = self.backbone.out_dim
        self.head = nn.Linear(d, num_classes)

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone.encode_image(x).float()
        logits = self.head(f)
        return logits


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
def make_loaders(cfg):
    class_names = scan_officehome_classes(cfg.dataset_root)
    t_weak = get_weak_transform(cfg.img_size)

    def make(domain):
        ds = OfficeHomeDataset(
            root=cfg.dataset_root,
            domain=domain,
            transform=t_weak,
            class_names=class_names
        )
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

    return {d: make(d) for d in cfg.source_domains}, class_names


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
def train_baseline(model, loaders, cfg, device, logger):
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for loader in loaders.values():
            for x, y, _ in loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                n += y.size(0)

        avg_loss = total_loss / max(1, n)
        logger.write(f"[Train][{epoch}] loss={avg_loss:.4f}")

        # ----------------------------------------------------
        # ✅ 保存 Linear baseline checkpoint
        # ----------------------------------------------------
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            ckpt_dir=cfg.ckpt_dir,
            exp_name=cfg.exp_name + "_linear"
        )

    return model


# ------------------------------------------------------------
# Eval
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, cfg, device, class_names):
    model.eval()
    t_weak = get_weak_transform(cfg.img_size)
    results = {}

    for domain in cfg.target_domains:
        ds = OfficeHomeDataset(
            root=cfg.dataset_root,
            domain=domain,
            transform=t_weak,
            class_names=class_names
        )
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

        accs = []
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            acc1 = accuracy(logits, y, topk=(1,))[0].item()
            accs.append(acc1)

        results[domain] = sum(accs) / len(accs)
        print(f"[Val] {domain}: acc@1={results[domain]:.2f}")

    mean_acc = sum(results.values()) / len(results)
    print(f"[Val] Mean acc@1={mean_acc:.2f}")
    return results


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=ALL_DOMAINS,
        help="单源域；若提供则自动设置 target_domains 为其他三个域"
    )
    args = parser.parse_args()

    # -------- cfg override --------
    cfg_over = {}
    if args.root:
        cfg_over["dataset_root"] = args.root
    if args.epochs:
        cfg_over["epochs"] = args.epochs

    if args.source:
        cfg_over["source_domains"] = [args.source]
        cfg_over["target_domains"] = [d for d in ALL_DOMAINS if d != args.source]

    cfg = get_cfg_baseline(cfg_over)

    # -------- env --------
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    logger = Logger(cfg.log_dir, cfg.exp_name + "_linear")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(cfg.seed, cfg.deterministic)

    # -------- data --------
    loaders, class_names = make_loaders(cfg)

    # -------- model --------
    model = LinearProbe(
        cfg.clip_backbone,
        cfg.num_classes,
        device
    ).to(device)

    # -------- train & eval --------
    model = train_baseline(model, loaders, cfg, device, logger)
    evaluate(model, cfg, device, class_names)


if __name__ == "__main__":
    main()
