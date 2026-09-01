# run_train.py  (SSDG: Single-Source -> Multi-Target)  — OfficeHome & TerraIncognita
from __future__ import annotations
import os
import argparse
import torch

from cfg import get_cfg
from utils import set_seed, save_checkpoint, cosine_lr_schedule, Logger
from data import (
    officehome_prompt_names,
    normalize_domainnet_domain,
    normalize_vlcs_domain,
)
from models import TASIL
from textspace import build_style_embeddings, build_class_texts
from losses import GroupDRO

from train_utils import (
    make_train_loader,
    train_one_epoch,
)

# -------------------------
# Supported domain lists
# -------------------------
OFFICEHOME_DOMAINS = ["Art", "Clipart", "Product", "Real World"]
TERRA_DOMAINS = ["location_38", "location_43", "location_46", "location_100"]
DOMAINNET_DOMAINS = ["clip", "info", "paint", "quick", "real", "sketch"]
VLCS_DOMAINS = ["C", "L", "S", "V"]

def to_prompt_names(dataset_name: str, class_names):
    ds = dataset_name.lower()
    if ds in ["officehome", "office-home", "oh"]:
        return officehome_prompt_names(class_names)
    # TerraIncognita: class names are already natural words; normalize lightly.
    return [c.replace("_", " ").lower() for c in class_names]

def get_all_domains(dataset_name: str):
    ds = dataset_name.lower()
    if ds in ["officehome", "office-home", "oh"]:
        return OFFICEHOME_DOMAINS
    elif ds in ["terraincognita", "terra", "ti"]:
        return TERRA_DOMAINS
    elif ds in ["domainnet", "domain-net", "dn"]:
        return DOMAINNET_DOMAINS
    elif ds in ["vlcs"]:
        return VLCS_DOMAINS
    else:
        raise ValueError(f"Unsupported dataset_name={dataset_name}")


def normalize_domain_arg(dataset_name: str, domain: str) -> str:
    ds = dataset_name.lower()
    if ds in ["domainnet", "domain-net", "dn"]:
        return normalize_domainnet_domain(domain)
    if ds in ["vlcs"]:
        return normalize_vlcs_domain(domain)
    return domain

def main():
    parser = argparse.ArgumentParser()

    # Dataset switch
    parser.add_argument("--dataset", type=str, default="officehome",
                        choices=["officehome", "terraincognita", "domainnet", "vlcs"],
                        help="dataset name")

    parser.add_argument("--root", type=str, default=None, help="dataset root (override cfg)")
    parser.add_argument("--epochs", type=int, default=None)

    # Split controls
    parser.add_argument("--source", type=str, required=True,
                        help="single source domain (OfficeHome: Art/Clipart/Product/Real World; "
                             "TerraIncognita: location_38/43/46/100; "
                             "DomainNet: clip/info/paint/quick/real/sketch; VLCS: C/L/S/V)")
    parser.add_argument("--targets", type=str, nargs="*", default=None,
                        help="optional custom target domains (exclude source). if None, use the other domains")

    # Debug / overrides
    parser.add_argument("--nan_guard", action="store_true", help="Enable NaN detection guards")
    parser.add_argument("--alpha", type=float, default=None, help="override cfg.alpha_style_remove")
    parser.add_argument("--seed", type=int, default=None, help="override cfg.seed")

    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    ALL_DOMAINS = get_all_domains(dataset_name)
    source_domain = normalize_domain_arg(dataset_name, args.source)

    # Validate source
    if source_domain not in ALL_DOMAINS:
        raise ValueError(f"--source={args.source} not in domains of {dataset_name}: {ALL_DOMAINS}")

    # Build targets
    if args.targets is None:
        target_domains = [d for d in ALL_DOMAINS if d != source_domain]
    else:
        normalized_targets = [normalize_domain_arg(dataset_name, d) for d in args.targets]
        target_domains = [d for d in normalized_targets if d in ALL_DOMAINS and d != source_domain]
        if len(target_domains) == 0:
            raise ValueError("`--targets` 不能为空且不能包含源域。")

    # ---- override cfg from CLI ----
    cfg_over = {
        "dataset_name": dataset_name,       # train_utils uses this to switch datasets
        "source_domains": [source_domain],  # SSDG: single source
        "target_domains": target_domains,   # multi-target
    }
    if args.root:
        cfg_over["dataset_root"] = args.root
    if args.epochs:
        cfg_over["epochs"] = args.epochs
    if args.alpha is not None:
        cfg_over["alpha_style_remove"] = args.alpha
    if args.seed is not None:
        cfg_over["seed"] = args.seed

    cfg = get_cfg(cfg_over)

    # dataset-specific num_classes safety (avoid OfficeHome defaults leaking into Terra)
    if getattr(cfg, "dataset_name", "officehome") == "terraincognita":
        cfg.num_classes = int(getattr(cfg, "num_classes", 10) or 10)

    # Unique experiment tag
    src_tag = source_domain.replace(" ", "_")
    exp_name_tagged = f"{cfg.exp_name}_SSDG_{dataset_name}_{src_tag}_seed{cfg.seed}"

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # ---- seed & logger ----
    set_seed(cfg.seed, cfg.deterministic)
    logger = Logger(cfg.log_dir, exp_name_tagged)
    logger.write(f"[Dataset] {dataset_name} | root={cfg.dataset_root}")
    logger.write(
        f"[Split] source={source_domain} | held-out targets={target_domains} "
        f"(not loaded during training)"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------- Data (single-source training only) ---------
    train_loader, class_dirnames = make_train_loader(cfg)
    cfg.num_classes = len(class_dirnames)
    class_prompt_names = to_prompt_names(dataset_name, class_dirnames)

    # --------- Model ---------
    model = TASIL(
        clip_name=cfg.clip_backbone, device=device,
        projector_mlp=getattr(cfg, "projector_mlp", False),
        alpha=getattr(cfg, "alpha_style_remove", 0.7),
        temperature=getattr(cfg, "init_temperature", 0.07),
        learnable_tau=getattr(cfg, "learnable_tau", True)
    ).to(device)

    # --------- Text spaces (built once) ---------
    style_embeddings = build_style_embeddings(
        model.backbone.model, device=device,
        k=getattr(cfg, "text_anchor_topk", 8),
    )
    E_s = style_embeddings.transpose(0, 1).contiguous()
    T = build_class_texts(model.backbone.model, class_prompt_names, device=device)
    model.set_style_embeddings(style_embeddings)
    model.set_style_subspace(E_s)
    model.set_class_texts(T)
    logger.write(
        f"[Appearance] feature-space augmentation enabled | "
        f"num_style_descriptors={style_embeddings.size(0)}"
    )

    a_raw = float(getattr(cfg, "alpha_style_remove", 0.0))
    logger.write(f"[Alpha] cfg.alpha_style_remove(raw) = {a_raw}")
    logger.write(f"[Alpha] model.alpha (tensor) = {float(model.alpha.detach().cpu())}")
    logger.write(f"[Alpha] learnable = {model.alpha.requires_grad}")
    try:
        logger.write(f"[Alpha] sigmoid(model.alpha) = {float(torch.sigmoid(model.alpha).detach().cpu())}")
    except Exception:
        pass

    # --------- Optimizer / DRO ---------
    opt_groups = [
        {"params": model.projector.parameters(), "lr": cfg.lr},
        {"params": [model.alpha], "lr": cfg.lr},
    ]
    cls_params = [p for p in model.classifier.parameters() if p.requires_grad]
    if len(cls_params) > 0:
        opt_groups.append({"params": cls_params, "lr": cfg.lr})

    optimizer = torch.optim.AdamW(opt_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 两个可微视图组：weak source view 与 feature-space appearance view。
    dro = GroupDRO(num_groups=2, eta=0.02, device=device)

    # --------- Preflight NaN check ---------
    if args.nan_guard or getattr(cfg, "nan_guard", False):
        model.eval()
        with torch.no_grad():
            batch0 = next(iter(train_loader))
            x0 = batch0["x_w"].to(device, non_blocking=True)[:8]
            x0_base = batch0["x_base"].to(device, non_blocking=True)[:8]
            dbg = model.debug_forward(x0)
            dbg_ap = model.forward_appearance_features(x0_base)

        def _flag(t): return bool(torch.isnan(t).any() or torch.isinf(t).any())
        report = {
            "Q_nan": _flag(dbg["Q"]) if dbg["Q"].numel() else False,
            "T_nan": _flag(dbg["T"]) if dbg["T"].numel() else False,
            "f0_nan": _flag(dbg["f0"]),
            "f_nan": _flag(dbg["f"]),
            "f_proj_nan": _flag(dbg["f_proj"]),
            "f_clean_nan": _flag(dbg["f_clean"]),
            "logits_unit_tau_nan": _flag(dbg["logits_unit_tau"]),
            "style_embeddings_nan": _flag(model.style_embeddings),
            "appearance_feature_nan": _flag(dbg_ap["v_ap"]),
            "appearance_clean_nan": _flag(dbg_ap["f_clean"]),
        }
        print("[Preflight]", report)
        if any(report.values()):
            raise SystemExit("Preflight failed: some tensors are non-finite. Check E_s/T/alpha/tau.")
        model.train()

    # --------- alpha warmup (optional) ---------
    alpha_warmup_epochs = int(getattr(cfg, "alpha_warmup_epochs", 0))
    alpha_target = float(getattr(cfg, "alpha_style_remove", 0.7))

    for epoch in range(1, cfg.epochs + 1):
        lr_now = cosine_lr_schedule(optimizer, cfg.lr, epoch - 1, cfg.epochs)
        logger.write(f"[LR] epoch {epoch} lr={lr_now:.6f}")

        if alpha_warmup_epochs > 0:
            if epoch <= alpha_warmup_epochs:
                model.set_alpha(0.0)
            else:
                t = (epoch - alpha_warmup_epochs) / max(1, (cfg.epochs - alpha_warmup_epochs))
                model.set_alpha(alpha_target * max(0.0, min(1.0, t)))

        train_one_epoch(
            model, train_loader, optimizer, dro, cfg, device, epoch, logger,
            nan_guard=(args.nan_guard or getattr(cfg, "nan_guard", False))
        )

    # The checkpoint rule is fixed in advance: use the final training epoch.
    # Held-out target domains are evaluated separately by evaluate.py.
    save_checkpoint(model, optimizer, cfg.epochs, cfg.ckpt_dir, exp_name_tagged)
    logger.write(f"[CKPT] saved fixed final-epoch checkpoint: epoch={cfg.epochs}")

    logger.write(
        f"=== SSDG Training finished. dataset={dataset_name} source={source_domain} | "
        f"fixed checkpoint epoch={cfg.epochs}; no target-domain evaluation during training ==="
    )

if __name__ == "__main__":
    main()
