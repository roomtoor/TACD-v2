# data/augment.py
from __future__ import annotations
from torchvision import transforms

# 与 CLIP 一致的归一化
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def _norm():
    return transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD)

def get_weak_transform(img_size: int = 224) -> transforms.Compose:
    """弱增广：保持语义，轻扰动。"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(), _norm()
    ])

def get_strong_transform(img_size: int = 224) -> transforms.Compose:
    """强增广：更像风格扰动。"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(), _norm()
    ])

def get_base_transform(img_size: int = 224) -> transforms.Compose:
    """无随机扰动的 CLIP 输入，用于构造特征空间 appearance view。"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), _norm()
    ])
