# test_clip_zeroshot.py
from __future__ import annotations
import os
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    # OfficeHome
    OfficeHomeDataset, scan_officehome_classes, officehome_prompt_names,
    # Terra
    TerraIncognitaDataset, scan_terraincognita_classes,
    # transform
    get_weak_transform,
)

# 你项目里用的 CLIP backbone 在 TACDv2 里已经封装了，这里直接复用最省事
from models import TACDv2


def build_class_prompts(dataset: str, class_names: List[str]) -> List[str]:
    ds = dataset.lower()
    if ds in ["officehome", "office-home", "oh"]:
        return officehome_prompt_names(class_names)
    # Terra: 类名通常就是自然词/下划线
    return [c.replace("_", " ").lower() for c in class_names]


@torch.no_grad()
def clip_zeroshot_eval(
    dataset: str,
    root: str,
    domains: List[str],
    clip_name: str = "ViT-B/16",
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
    device: str = "cuda",
):
    # ---------- load classes ----------
    ds = dataset.lower()
    if ds in ["officehome", "office-home", "oh"]:
        class_names = scan_officehome_classes(root)
    elif ds in ["terraincognita", "terra", "ti"]:
        class_names = scan_terraincognita_classes(root)
    else:
        raise ValueError(f"Unsupported dataset={dataset}")

    prompts = build_class_prompts(ds, class_names)

    # ---------- build CLIP model (no training) ----------
    # 这里复用 TACDv2 的 backbone：里面已经 load 好 CLIP
    model = TACDv2(clip_name=clip_name, device=device).to(device)
    model.eval()

    clip_model = model.backbone.model  # openai clip model
    # NOTE: openai clip 通常是 model.encode_image / model.encode_text

    # ---------- encode texts once ----------
    # 你 textspace.build_class_texts 也可以用，但这里不依赖它，直接走 CLIP tokenizer 更通用
    try:
        import clip  # openai/clip
        text_tokens = clip.tokenize(prompts).to(device)
    except Exception as e:
        raise RuntimeError(
            "Cannot import openai clip tokenizer. "
            "If your project loads CLIP via openai/clip, please ensure `import clip` works."
        ) from e

    text_feat = clip_model.encode_text(text_tokens)
    text_feat = F.normalize(text_feat.float(), dim=1)

    # ---------- dataloaders ----------
    t = get_weak_transform(img_size)

    def make_loader(dom: str):
        if ds in ["officehome", "office-home", "oh"]:
            base = OfficeHomeDataset(root=root, domain=dom, transform=t, class_names=class_names, return_pil=False)
        else:
            base = TerraIncognitaDataset(root=root, domain=dom, transform=t, class_names=class_names, return_pil=False)
        return DataLoader(base, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    results = {}
    for dom in domains:
        loader = make_loader(dom)
        correct = 0
        total = 0

        for batch in loader:
            # base dataset 返回 (img, y, path)
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

            img_feat = clip_model.encode_image(x)
            img_feat = F.normalize(img_feat.float(), dim=1)

            # cosine similarity = dot product after normalize
            logits = img_feat @ text_feat.t()
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.numel()

        acc = 100.0 * correct / max(total, 1)
        results[dom] = acc
        print(f"[CLIP-ZS] {dom}: acc@1={acc:.2f}")

    mean_acc = sum(results.values()) / max(len(results), 1)
    worst_acc = min(results.values()) if results else 0.0
    print(f"[CLIP-ZS] Mean acc@1={mean_acc:.2f}")
    print(f"[CLIP-ZS] Worst-domain acc@1={worst_acc:.2f}")
    return results, mean_acc, worst_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["officehome", "terraincognita"])
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--clip", type=str, default="ViT-B/16")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--source", type=str, default=None, help="if set, eval all other domains")
    ap.add_argument("--domains", type=str, nargs="*", default=None, help="explicit domains to eval")
    args = ap.parse_args()

    if args.dataset == "officehome":
        ALL = ["Art", "Clipart", "Product", "Real World"]
    else:
        ALL = ["location_38", "location_43", "location_46", "location_100"]

    if args.domains:
        domains = args.domains
    elif args.source:
        domains = [d for d in ALL if d != args.source]
    else:
        domains = ALL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_zeroshot_eval(
        dataset=args.dataset,
        root=args.root,
        domains=domains,
        clip_name=args.clip,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )


if __name__ == "__main__":
    main()
