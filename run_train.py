# Training entry point for strict single-source domain generalization.
from __future__ import annotations
import os
import argparse
import math
import torch

from cfg import get_cfg
from utils import set_seed, save_checkpoint, cosine_lr_schedule, Logger
from data import (
    officehome_prompt_names,
    normalize_domainnet_domain,
    normalize_vlcs_domain,
)
from models import TASIL
from textspace import (
    STYLE_BANKS,
    build_style_embeddings,
    build_class_texts,
    get_style_bank,
)
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
    # Debug / overrides
    parser.add_argument("--nan_guard", action="store_true", help="Enable NaN detection guards")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="override the raw alpha logit (effective coefficient = sigmoid(alpha))",
    )
    parser.add_argument(
        "--fixed-alpha-eff",
        type=float,
        default=None,
        help="freeze the effective suppression coefficient in (0,1); e.g. 0.5",
    )
    parser.add_argument(
        "--appearance-bank",
        choices=sorted(STYLE_BANKS),
        default="default-29",
        help="descriptor bank used to construct the appearance view",
    )
    parser.add_argument(
        "--suppression-bank",
        choices=sorted(STYLE_BANKS),
        default=None,
        help="descriptor bank whose QR span is suppressed; defaults to appearance bank",
    )
    parser.add_argument("--seed", type=int, default=None, help="override cfg.seed")

    args = parser.parse_args()
    if args.alpha is not None and args.fixed_alpha_eff is not None:
        raise ValueError("Use either --alpha (raw) or --fixed-alpha-eff, not both.")

    dataset_name = args.dataset.lower()
    ALL_DOMAINS = get_all_domains(dataset_name)
    source_domain = normalize_domain_arg(dataset_name, args.source)

    # Validate source
    if source_domain not in ALL_DOMAINS:
        raise ValueError(f"--source={args.source} not in domains of {dataset_name}: {ALL_DOMAINS}")

    # ---- override cfg from CLI ----
    cfg_over = {
        "dataset_name": dataset_name,       # train_utils uses this to switch datasets
        "source_domains": [source_domain],  # SSDG: single source
    }
    if args.root:
        cfg_over["dataset_root"] = args.root
    if args.epochs:
        cfg_over["epochs"] = args.epochs
    if args.alpha is not None:
        cfg_over["alpha_style_remove"] = args.alpha
    if args.fixed_alpha_eff is not None:
        if not 0.0 < args.fixed_alpha_eff < 1.0:
            raise ValueError("--fixed-alpha-eff must be strictly between 0 and 1.")
        cfg_over["alpha_style_remove"] = math.log(
            args.fixed_alpha_eff / (1.0 - args.fixed_alpha_eff)
        )
    if args.seed is not None:
        cfg_over["seed"] = args.seed

    cfg = get_cfg(cfg_over)

    # Unique experiment tag
    src_tag = source_domain.replace(" ", "_")
    suppression_bank_id = args.suppression_bank or args.appearance_bank
    exp_name_tagged = f"{cfg.exp_name}_SSDG_{dataset_name}_{src_tag}_seed{cfg.seed}"
    if (
        args.appearance_bank != "default-29"
        or suppression_bank_id != "default-29"
        or args.fixed_alpha_eff is not None
    ):
        exp_name_tagged += (
            f"_app-{args.appearance_bank}_sup-{suppression_bank_id}"
            + (
                f"_alphaeff-{args.fixed_alpha_eff:g}"
                if args.fixed_alpha_eff is not None
                else ""
            )
        )

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # ---- seed & logger ----
    set_seed(cfg.seed, cfg.deterministic)
    logger = Logger(cfg.log_dir, exp_name_tagged)
    logger.write(f"[Dataset] {dataset_name} | root={cfg.dataset_root}")
    logger.write(f"[Split] source={source_domain} | strict source-only training")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------- Data (single-source training only) ---------
    train_loader, class_dirnames = make_train_loader(cfg)
    class_prompt_names = to_prompt_names(dataset_name, class_dirnames)
    logger.write(
        f"[Classes] discovered from source only | count={len(class_dirnames)} | "
        f"names={class_dirnames}"
    )

    # --------- Model ---------
    model = TASIL(
        clip_name=cfg.clip_backbone, device=device,
        projector_mlp=cfg.projector_mlp,
        alpha=getattr(cfg, "alpha_style_remove", 0.7),
        temperature=cfg.init_temperature,
        learnable_tau=cfg.learnable_tau,
    ).to(device)

    # --------- Text spaces (built once) ---------
    appearance_words = get_style_bank(args.appearance_bank)
    suppression_words = get_style_bank(suppression_bank_id)
    appearance_embeddings = build_style_embeddings(
        model.backbone.model,
        device=device,
        style_words=appearance_words,
    )
    suppression_embeddings = build_style_embeddings(
        model.backbone.model,
        device=device,
        style_words=suppression_words,
    )
    E_s = suppression_embeddings.transpose(0, 1).contiguous()
    T = build_class_texts(model.backbone.model, class_prompt_names, device=device)
    model.set_style_embeddings(appearance_embeddings)
    model.set_style_subspace(E_s)
    model.set_class_texts(T)
    logger.write(
        f"[Banks] appearance={args.appearance_bank} ({len(appearance_words)}) | "
        f"suppression={suppression_bank_id} ({len(suppression_words)})"
    )
    logger.write(f"[Banks] appearance descriptors={appearance_words}")
    logger.write(f"[Banks] suppression descriptors={suppression_words}")

    if args.fixed_alpha_eff is not None:
        model.alpha.requires_grad_(False)
        logger.write(f"[Alpha] frozen effective coefficient={args.fixed_alpha_eff}")

    a_raw = float(getattr(cfg, "alpha_style_remove", 0.0))
    logger.write(f"[Alpha] cfg.alpha_style_remove(raw) = {a_raw}")
    logger.write(f"[Alpha] model.alpha (tensor) = {float(model.alpha.detach().cpu())}")
    logger.write(f"[Alpha] learnable = {model.alpha.requires_grad}")
    try:
        logger.write(f"[Alpha] sigmoid(model.alpha) = {float(torch.sigmoid(model.alpha).detach().cpu())}")
    except Exception:
        pass

    # --------- Optimizer / DRO ---------
    opt_groups = [{"params": model.projector.parameters(), "lr": cfg.lr}]
    if model.alpha.requires_grad:
        opt_groups.append({"params": [model.alpha], "lr": cfg.lr})
    cls_params = [p for p in model.classifier.parameters() if p.requires_grad]
    if len(cls_params) > 0:
        opt_groups.append({"params": cls_params, "lr": cfg.lr})

    optimizer = torch.optim.AdamW(opt_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 两个可微视图组：weak source view 与 feature-space appearance view。
    dro = GroupDRO(num_groups=2, eta=0.02, device=device)

    # --------- Preflight NaN check ---------
    if args.nan_guard:
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

    for epoch in range(1, cfg.epochs + 1):
        lr_now = cosine_lr_schedule(optimizer, cfg.lr, epoch - 1, cfg.epochs)
        logger.write(f"[LR] epoch {epoch} lr={lr_now:.6f}")

        train_one_epoch(
            model, train_loader, optimizer, dro, cfg, device, epoch, logger,
            nan_guard=args.nan_guard,
        )

    # The checkpoint rule is fixed in advance: use the final training epoch.
    # Held-out target domains are evaluated separately by evaluate.py.
    checkpoint_metadata = {
        "dataset_name": dataset_name,
        "source_domain": source_domain,
        "class_names": list(class_dirnames),
        "seed": int(cfg.seed),
        "clip_backbone": cfg.clip_backbone,
        "appearance_bank": args.appearance_bank,
        "suppression_bank": suppression_bank_id,
        "appearance_descriptors": appearance_words,
        "suppression_descriptors": suppression_words,
        "fixed_alpha_eff": args.fixed_alpha_eff,
        "checkpoint_rule": "fixed_final_epoch",
    }
    save_checkpoint(
        model,
        optimizer,
        cfg.epochs,
        cfg.ckpt_dir,
        exp_name_tagged,
        metadata=checkpoint_metadata,
    )
    logger.write(f"[CKPT] saved fixed final-epoch checkpoint: epoch={cfg.epochs}")

    logger.write(
        f"=== SSDG Training finished. dataset={dataset_name} source={source_domain} | "
        f"fixed checkpoint epoch={cfg.epochs}; no target-domain access during training ==="
    )

if __name__ == "__main__":
    main()
