# train_utils.py
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from pathlib import Path
from utils import AverageMeter, accuracy

__all__ = [
    "collate_fn_dict",
    "group_mean_losses",
    "make_train_loader",
    "make_val_loaders",
    "train_one_epoch",
    "evaluate",
]

# ------------------------- helpers -------------------------
def collate_fn_dict(batch_list: List[Dict]):
    """把 dict 样本列表堆叠成 batch（张量堆叠，其它转列表）。"""
    out = {}
    keys = batch_list[0].keys()
    for k in keys:
        v0 = batch_list[0][k]
        if torch.is_tensor(v0):
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)
        elif isinstance(v0, int):
            out[k] = torch.tensor([b[k] for b in batch_list], dtype=torch.long)
        else:
            out[k] = [b[k] for b in batch_list]
    return out


def group_mean_losses(losses: torch.Tensor, group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    losses: [B], group_ids: [B] in [0, G-1]
    return: [G] 每组样本的均值损失（无样本则为0）
    """
    device = losses.device
    out = torch.zeros(num_groups, device=device)
    for g in range(num_groups):
        m = (group_ids == g)
        if m.any():
            out[g] = losses[m].mean()
    return out


def make_train_loader(cfg) -> Tuple[DataLoader, List[str]]:
    """
    拼接多个源域，提供弱/强/反事实多视图。支持 OfficeHome / TerraIncognita。
    返回:
      - train_loader: dict batch (x_w/x_s[/x_cf]/y/domain/path)
      - class_names: 全域统一类名列表（用于 val / 文本模板 / num_classes 对齐）
    """
    train_sets = []
    dataset_name = getattr(cfg, "dataset_name", "officehome").lower()

    if dataset_name in ["officehome", "office-home", "oh"]:
        # 延迟导入避免循环依赖
        from data import OfficeHomeMultiView, scan_officehome_classes

        #  OfficeHome：四域交集，确保 label 对齐
        class_names = scan_officehome_classes(cfg.dataset_root)

        for d in cfg.source_domains:
            train_sets.append(
                OfficeHomeMultiView(
                    root=cfg.dataset_root,
                    domain=d,
                    img_size=cfg.img_size,
                    return_counterfactual=True,
                    class_names=class_names,
                )
            )

    elif dataset_name in ["terraincognita", "terra", "ti"]:
        from data import TerraIncognitaMultiView, scan_terraincognita_classes

        #  Terra：四域交集（source+target 都一致），防止 label space 错位
        class_names = scan_terraincognita_classes(cfg.dataset_root)

        for d in cfg.source_domains:
            train_sets.append(
                TerraIncognitaMultiView(
                    root=cfg.dataset_root,
                    domain=d,
                    img_size=cfg.img_size,
                    return_counterfactual=True,
                    class_names=class_names,
                )
            )

    else:
        raise ValueError(f"Unknown cfg.dataset_name={dataset_name}")

    if len(train_sets) == 0:
        raise RuntimeError("[make_train_loader] train_sets 为空：检查 cfg.source_domains / dataset_root")

    concat = ConcatDataset(train_sets)

    loader = DataLoader(
        concat,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=getattr(cfg, "drop_last", True),
        collate_fn=collate_fn_dict,  # 训练期 multi-view dict 必须用 dict collate
    )
    return loader, class_names


def make_val_loaders(cfg, class_names: List[str]) -> Dict[str, DataLoader]:
    """
    每个目标域一个验证 loader（弱增广/单视图）。
    返回 dict: domain -> loader
    batch 形式为 (x, y)（如果你的 evaluate() 期望 dict，可把下面的 ds 也改成返回 dict）。
    """
    loaders: Dict[str, DataLoader] = {}
    dataset_name = getattr(cfg, "dataset_name", "officehome").lower()

    from data import get_weak_transform
    t_weak = get_weak_transform(cfg.img_size)

    if dataset_name in ["officehome", "office-home", "oh"]:
        from data import OfficeHomeDataset

        for d in cfg.target_domains:
            ds = OfficeHomeDataset(
                root=cfg.dataset_root,
                domain=d,
                transform=t_weak,          #  直接用弱增广（输出 tensor）
                class_names=class_names,   #  统一 label space
                return_pil=False,
            )
            loaders[d] = DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
        return loaders

    elif dataset_name in ["terraincognita", "terra", "ti"]:
        from data import TerraIncognitaDataset

        for d in cfg.target_domains:
            ds = TerraIncognitaDataset(
                root=cfg.dataset_root,
                domain=d,
                transform=t_weak,          #  直接用弱增广（输出 tensor）
                class_names=class_names,   #  统一 label space
                return_pil=False,
            )
            loaders[d] = DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
        return loaders

    else:
        raise ValueError(f"Unknown cfg.dataset_name={dataset_name}")



# ------------------------- train & eval -------------------------
def _check_finite(t: torch.Tensor, name: str):
    if not torch.isfinite(t).all():
        bad = t[~torch.isfinite(t)]
        mn = float(t.min()) if t.numel() else float("nan")
        mx = float(t.max()) if t.numel() else float("nan")
        eg = float(bad[0]) if bad.numel() else float("nan")
        raise RuntimeError(f"[NaN-DETECT] {name} non-finite! min={mn} max={mx} example={eg}")


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    dro,
    cfg,
    device: str,
    epoch: int,
    logger,
    *,
    nan_guard: bool = False,
):
    """
    只做前向/反向/日志；DRO 的权重更新在这里完成。
    需要的外部组件：cosine_ce, symmetric_kl（从 losses import）
    """
    from losses import cosine_ce, symmetric_kl  # 延迟导入

    model.train()
    loss_meter = AverageMeter("loss")
    cls_meter  = AverageMeter("cls")
    cons_meter = AverageMeter("cons")
    dro_meter  = AverageMeter("dro")

    pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")

    for it, batch in enumerate(pbar):
        x_w = batch["x_w"].to(device, non_blocking=True)
        x_s = batch["x_s"].to(device, non_blocking=True)
        y   = batch["y"].to(device, non_blocking=True)
        dom = batch["domain"].to(device, non_blocking=True)

        logits_w = model(x_w)           # f_clean -> logits
        logits_s = model(x_s)

        if nan_guard:
            _check_finite(logits_w, "logits_w")
            _check_finite(logits_s, "logits_s")

        loss_cls  = cosine_ce(logits_w, y)
        loss_cons = symmetric_kl(logits_w, logits_s)

        # 组损失（各域）
        with torch.no_grad():
            per_ex_loss = F.cross_entropy(logits_w, y, reduction="none").clamp_min(0.0)  # [B]
        dom_group_losses = group_mean_losses(per_ex_loss, dom, len(cfg.source_domains))  # [G_dom]

        # 反事实组
        group_losses_list = [dom_group_losses]
        if "x_cf" in batch:
            x_cf = batch["x_cf"].to(device, non_blocking=True)
            logits_cf = model(x_cf)
            loss_cf = cosine_ce(logits_cf, y)
            group_losses_list.append(loss_cf.detach().unsqueeze(0))  # [1]

        group_losses = torch.cat(group_losses_list, dim=0)  # [G_dom + 1]
        dro.update(group_losses.detach())
        loss_dro = dro(group_losses)

        loss = cfg.lambda_cls * loss_cls + cfg.lambda_cons * loss_cons + cfg.lambda_group * loss_dro

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        loss_meter.update(loss.item(), y.size(0))
        cls_meter.update(loss_cls.item(), y.size(0))
        cons_meter.update(loss_cons.item(), y.size(0))
        dro_meter.update(loss_dro.item(), 1)

        if (it + 1) % cfg.print_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "cls": f"{cls_meter.avg:.4f}",
                "cons": f"{cons_meter.avg:.4f}",
                "dro": f"{dro_meter.avg:.4f}",
            })

    logger.write(
        f"[Train][{epoch}] loss={loss_meter.avg:.4f} | "
        f"cls={cls_meter.avg:.4f} | cons={cons_meter.avg:.4f} | dro={dro_meter.avg:.4f}"
    )


@torch.no_grad()
def evaluate(model, val_loaders: Dict[str, DataLoader], device, epoch, logger):
    model.eval()
    results = {}

    for domain, loader in val_loaders.items():
        top1_meter = AverageMeter(f"acc@1_{domain}")

        for batch in loader:
            # 兼容 (x, y) 或 (x, y, path)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x, y = batch
                elif len(batch) >= 3:
                    x, y = batch[0], batch[1]
                else:
                    raise RuntimeError(f"[evaluate] Unexpected batch tuple length={len(batch)}")
            else:
                # 如果你未来把 val 也做成 dict，这里也能兼容
                # 约定 key: 'x' 或 'x_w'
                if "x" in batch:
                    x, y = batch["x"], batch["y"]
                elif "x_w" in batch:
                    x, y = batch["x_w"], batch["y"]
                else:
                    raise RuntimeError(f"[evaluate] Unexpected batch type/keys: {type(batch)} {getattr(batch,'keys',lambda:[])()}")

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            acc1 = accuracy(logits, y, topk=(1,))[0].item()
            top1_meter.update(acc1, y.size(0))

        results[domain] = top1_meter.avg
        logger.write(f"[Val][{epoch}] {domain}: acc@1={top1_meter.avg:.2f}")

    mean_acc = sum(results.values()) / max(len(results), 1)
    logger.write(f"[Val][{epoch}] Mean acc@1={mean_acc:.2f}")
    return results, mean_acc
