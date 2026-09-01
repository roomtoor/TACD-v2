# data/datasets.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union

from PIL import Image
from torch.utils.data import Dataset

from .augment import (
    get_weak_transform,
    get_strong_transform,
    get_base_transform,
)

# ============================================================
# Common
# ============================================================

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# Office-Home
# ============================================================

_OFFICEHOME_DOMAIN_ALIASES: Dict[str, str] = {
    "art": "Art", "ar": "Art",
    "clipart": "Clipart", "cl": "Clipart",
    "product": "Product", "pr": "Product",
    "real_world": "Real World", "real-world": "Real World",
    "realworld": "Real World", "rw": "Real World", "real world": "Real World",
    "Art": "Art", "Clipart": "Clipart", "Product": "Product", "Real World": "Real World",
}

def _norm_domain(domain: str) -> str:
    d = domain.strip()
    d_key = d.lower().replace("-", "_")
    return _OFFICEHOME_DOMAIN_ALIASES.get(d_key, _OFFICEHOME_DOMAIN_ALIASES.get(d, d))


def scan_officehome_classes(root: Union[str, Path]) -> List[str]:
    """扫描四域的类名交集；按字母序返回，保证全域同一 label 映射。"""
    root = Path(root).expanduser().resolve()
    domains = ["Art", "Clipart", "Product", "Real World"]
    sets = []
    for d in domains:
        dom_dir = root / d
        if not dom_dir.is_dir():
            raise FileNotFoundError(f"[OfficeHome] 目录不存在: {dom_dir}")
        classes = {p.name for p in dom_dir.iterdir() if p.is_dir()}
        if not classes:
            raise RuntimeError(f"[OfficeHome] {dom_dir} 下未找到类别子目录")
        sets.append(classes)
    common = set.intersection(*sets)
    return sorted(common)


def officehome_prompt_names(class_names: List[str]) -> List[str]:
    """'Alarm_Clock' -> 'alarm clock'（用于 CLIP 文本模板）"""
    return [c.replace("_", " ").lower() for c in class_names]


_DOMAIN_ORDER = ["Art", "Clipart", "Product", "Real World"]
_DOMAIN_TO_ID = {d: i for i, d in enumerate(_DOMAIN_ORDER)}

def officehome_domain_id(domain: str) -> int:
    d = _norm_domain(domain)
    if d not in _DOMAIN_TO_ID:
        raise ValueError(f"Unknown Office-Home domain: {domain}")
    return _DOMAIN_TO_ID[d]


class OfficeHomeDataset(Dataset):
    """
    root/
      Art/        <class>/*.jpg
      Clipart/    <class>/*.jpg
      Product/    <class>/*.jpg
      Real World/ <class>/*.jpg
    """
    def __init__(
        self,
        root: Union[str, Path],
        domain: str,
        transform=None,
        class_names: Optional[List[str]] = None,
        return_pil: bool = False,
        recursive: bool = True,
    ):
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.domain = _norm_domain(domain)
        self.transform = transform
        self.return_pil = return_pil
        self.recursive = recursive

        dom_dir = self.root / self.domain
        if not dom_dir.is_dir():
            raise FileNotFoundError(f"[OfficeHome] 域目录不存在: {dom_dir}")

        if class_names is None:
            self.class_names = scan_officehome_classes(self.root)
        else:
            self.class_names = sorted(class_names)

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = dom_dir / cls
            if not cls_dir.is_dir():
                print(f"[WARN] {self.domain} 缺少类别目录: {cls_dir}")
                continue
            it = cls_dir.rglob("*") if recursive else cls_dir.iterdir()
            for p in it:
                if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                    self.samples.append((p, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(f"[OfficeHome] 在 {dom_dir} 未找到任何图像")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            pil = im.convert("RGB")
        img = self.transform(pil) if self.transform else pil
        if self.return_pil:
            return img, label, str(path), pil
        return img, label, str(path)


class OfficeHomeMultiView(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        domain: str,
        img_size: int = 224,
        return_appearance: bool = True,
        class_names: Optional[List[str]] = None,
        recursive: bool = True,
    ):
        super().__init__()
        self.base = OfficeHomeDataset(
            root=root,
            domain=domain,
            transform=None,
            class_names=class_names,
            return_pil=True,
            recursive=recursive,
        )
        self.domain_id = officehome_domain_id(domain)
        self.t_weak = get_weak_transform(img_size)
        self.t_strong = get_strong_transform(img_size)
        self.t_base = get_base_transform(img_size)
        self.return_appearance = return_appearance

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        _, y, path, pil_raw = self.base[idx]
        x_w = self.t_weak(pil_raw)
        x_s = self.t_strong(pil_raw)
        out = {
            "x_w": x_w,
            "x_s": x_s,
            "y": y,
            "domain": self.domain_id,
            "path": path,
        }
        if self.return_appearance:
            # 这里只提供无随机扰动的原图输入；appearance view 在 CLIP
            # 特征空间中由模型使用随机风格文本向量构造，不生成新像素。
            out["x_base"] = self.t_base(pil_raw)
        return out


# ============================================================
# TerraIncognita
# ============================================================

_TERRA_DOMAINS = ["location_38", "location_43", "location_46", "location_100"]
_TERRA_DOMAIN_TO_ID = {d: i for i, d in enumerate(_TERRA_DOMAINS)}

def terraincognita_domain_id(domain: str) -> int:
    if domain not in _TERRA_DOMAIN_TO_ID:
        raise ValueError(f"Unknown TerraIncognita domain: {domain}")
    return _TERRA_DOMAIN_TO_ID[domain]


def scan_terraincognita_classes(root: Union[str, Path]) -> List[str]:
    """
    扫描四个 location 的类名交集；按字母序返回，保证全域同一 label 映射。
    这是 SSDG / 多域评测的关键：避免不同域 label 对不上。
    """
    root = Path(root).expanduser().resolve()
    sets = []
    for d in _TERRA_DOMAINS:
        dom_dir = root / d
        if not dom_dir.is_dir():
            raise FileNotFoundError(f"[TerraIncognita] 目录不存在: {dom_dir}")
        classes = {p.name for p in dom_dir.iterdir() if p.is_dir()}
        if not classes:
            raise RuntimeError(f"[TerraIncognita] {dom_dir} 下未找到类别子目录")
        sets.append(classes)

    common = set.intersection(*sets)
    if not common:
        raise RuntimeError("[TerraIncognita] 四域类别交集为空：检查数据组织/是否某些域缺类")
    return sorted(common)


class TerraIncognitaDataset(Dataset):
    """
    root/
      location_38/<class>/*.jpg
      location_43/<class>/*.jpg
      location_46/<class>/*.jpg
      location_100/<class>/*.jpg

    重要：默认使用四域“类名交集”做统一 label 映射。
    """
    def __init__(
        self,
        root: Union[str, Path],
        domain: str,
        transform=None,
        return_pil: bool = False,
        recursive: bool = True,
        class_names: Optional[List[str]] = None,  # NEW: 允许外部传入统一类表
    ):
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.domain = domain
        self.transform = transform
        self.return_pil = return_pil
        self.recursive = recursive

        dom_dir = self.root / self.domain
        if not dom_dir.is_dir():
            raise FileNotFoundError(f"[TerraIncognita] 域目录不存在: {dom_dir}")

        # 统一类表：默认取四域交集，避免跨域 label 错位
        if class_names is None:
            self.class_names = scan_terraincognita_classes(self.root)
        else:
            self.class_names = sorted(class_names)

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = dom_dir / cls
            if not cls_dir.is_dir():
                # 理论上用交集时不会缺，但这里防御一下
                # print(f"[WARN] {self.domain} 缺少类别目录: {cls_dir}")
                continue
            it = cls_dir.rglob("*") if recursive else cls_dir.iterdir()
            for p in it:
                if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                    self.samples.append((p, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(f"[TerraIncognita] 在 {dom_dir} 未找到任何图像（或交集类表过滤后为空）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            pil = im.convert("RGB")
        img = self.transform(pil) if self.transform else pil
        if self.return_pil:
            return img, label, str(path), pil
        return img, label, str(path)


class TerraIncognitaMultiView(Dataset):
    """
    输出多视图 dict：
      x_w: weak augment tensor
      x_s: strong augment tensor
      x_base: 无随机扰动的原图输入；用于构造特征空间 appearance view
      y: label (统一映射)
      domain: domain_id
      path: image path
    """
    def __init__(
        self,
        root: Union[str, Path],
        domain: str,
        img_size: int = 224,
        return_appearance: bool = True,
        recursive: bool = True,
        class_names: Optional[List[str]] = None,  # NEW
    ):
        super().__init__()

        self.base = TerraIncognitaDataset(
            root=root,
            domain=domain,
            transform=None,
            return_pil=True,
            recursive=recursive,
            class_names=class_names,  # NEW
        )
        self.domain_id = terraincognita_domain_id(domain)

        # 复用你 data/augment.py 的 transform
        self.t_weak = get_weak_transform(img_size)
        self.t_strong = get_strong_transform(img_size)
        self.t_base = get_base_transform(img_size)
        self.return_appearance = return_appearance

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        _, y, path, pil_raw = self.base[idx]
        x_w = self.t_weak(pil_raw)
        x_s = self.t_strong(pil_raw)
        out = {
            "x_w": x_w,
            "x_s": x_s,
            "y": y,
            "domain": self.domain_id,
            "path": path,
        }
        if self.return_appearance:
            # 风格文本向量在模型中加入 CLIP 图像特征；此处不做像素级伪风格化。
            out["x_base"] = self.t_base(pil_raw)
        return out


# ============================================================
# DomainNet / VLCS (DomainBed-style folder datasets)
# ============================================================

_DOMAINNET_DOMAINS = ["clip", "info", "paint", "quick", "real", "sketch"]
_DOMAINNET_ALIASES = {
    "clip": "clip", "clipart": "clip",
    "info": "info", "infograph": "info",
    "paint": "paint", "painting": "paint",
    "quick": "quick", "quickdraw": "quick", "quick_draw": "quick",
    "real": "real", "real_world": "real",
    "sketch": "sketch",
}
_DOMAINNET_FOLDERS = {
    "clip": ("clip", "clipart"),
    "info": ("info", "infograph"),
    "paint": ("paint", "painting"),
    "quick": ("quick", "quickdraw"),
    "real": ("real",),
    "sketch": ("sketch",),
}

_VLCS_DOMAINS = ["C", "L", "S", "V"]
_VLCS_ALIASES = {
    "c": "C", "caltech": "C", "caltech101": "C",
    "l": "L", "labelme": "L",
    "s": "S", "sun": "S", "sun09": "S",
    "v": "V", "voc": "V", "voc2007": "V",
    "pascal": "V", "pascal_voc": "V", "pascalvoc": "V",
}
_VLCS_FOLDERS = {
    "C": ("C", "Caltech101", "Caltech"),
    "L": ("L", "LabelMe"),
    "S": ("S", "SUN09", "SUN"),
    "V": ("V", "VOC2007", "PASCAL", "PASCAL_VOC"),
}


def _alias_key(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_named_domain(domain: str, aliases: Dict[str, str], dataset_name: str) -> str:
    key = _alias_key(domain)
    if key not in aliases:
        raise ValueError(f"Unknown {dataset_name} domain: {domain}")
    return aliases[key]


def normalize_domainnet_domain(domain: str) -> str:
    return _normalize_named_domain(domain, _DOMAINNET_ALIASES, "DomainNet")


def normalize_vlcs_domain(domain: str) -> str:
    return _normalize_named_domain(domain, _VLCS_ALIASES, "VLCS")


def _select_nested_root(root: Union[str, Path], nested_names: Tuple[str, ...]) -> Path:
    root_path = Path(root).expanduser().resolve()
    for name in nested_names:
        nested = root_path / name
        if nested.is_dir():
            return nested
    return root_path


def _resolve_domain_dir(
    root: Path,
    canonical_domain: str,
    folder_candidates: Dict[str, Tuple[str, ...]],
    dataset_name: str,
) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"[{dataset_name}] 根目录不存在: {root}")

    children = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for candidate in folder_candidates[canonical_domain]:
        matched = children.get(candidate.lower())
        if matched is not None:
            return matched

    expected = ", ".join(folder_candidates[canonical_domain])
    raise FileNotFoundError(
        f"[{dataset_name}] 找不到域 {canonical_domain}；在 {root} 下期望目录之一: {expected}"
    )


def _scan_named_domain_classes(
    root: Path,
    domains: List[str],
    folder_candidates: Dict[str, Tuple[str, ...]],
    dataset_name: str,
) -> List[str]:
    """构建跨域统一标签空间；缺失类别可在个别域中没有样本。"""
    class_union = set()
    for domain in domains:
        dom_dir = _resolve_domain_dir(root, domain, folder_candidates, dataset_name)
        classes = {p.name for p in dom_dir.iterdir() if p.is_dir()}
        if not classes:
            raise RuntimeError(f"[{dataset_name}] {dom_dir} 下未找到类别子目录")
        class_union.update(classes)
    if not class_union:
        raise RuntimeError(f"[{dataset_name}] 未找到任何类别")
    return sorted(class_union)


def scan_domainnet_classes(root: Union[str, Path]) -> List[str]:
    root_path = _select_nested_root(root, ("domain_net", "DomainNet"))
    return _scan_named_domain_classes(
        root_path, _DOMAINNET_DOMAINS, _DOMAINNET_FOLDERS, "DomainNet"
    )


def scan_vlcs_classes(root: Union[str, Path]) -> List[str]:
    root_path = _select_nested_root(root, ("VLCS", "vlcs"))
    return _scan_named_domain_classes(root_path, _VLCS_DOMAINS, _VLCS_FOLDERS, "VLCS")


def domainnet_prompt_names(class_names: List[str]) -> List[str]:
    return [c.replace("_", " ").lower() for c in class_names]


def vlcs_prompt_names(class_names: List[str]) -> List[str]:
    return [c.replace("_", " ").lower() for c in class_names]


def domainnet_domain_id(domain: str) -> int:
    return _DOMAINNET_DOMAINS.index(normalize_domainnet_domain(domain))


def vlcs_domain_id(domain: str) -> int:
    return _VLCS_DOMAINS.index(normalize_vlcs_domain(domain))


class _NamedFolderDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        domain: str,
        *,
        domains: List[str],
        normalize_domain,
        folder_candidates: Dict[str, Tuple[str, ...]],
        dataset_name: str,
        nested_roots: Tuple[str, ...],
        transform=None,
        class_names: Optional[List[str]] = None,
        return_pil: bool = False,
        recursive: bool = True,
    ):
        super().__init__()
        self.root = _select_nested_root(root, nested_roots)
        self.domain = normalize_domain(domain)
        self.transform = transform
        self.return_pil = return_pil
        self.recursive = recursive
        self.dataset_name = dataset_name
        self.dom_dir = _resolve_domain_dir(
            self.root, self.domain, folder_candidates, dataset_name
        )

        if class_names is None:
            self.class_names = _scan_named_domain_classes(
                self.root, domains, folder_candidates, dataset_name
            )
        else:
            self.class_names = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.dom_dir / cls
            if not cls_dir.is_dir():
                continue
            iterator = cls_dir.rglob("*") if recursive else cls_dir.iterdir()
            paths = sorted(
                (p for p in iterator if p.is_file() and p.suffix.lower() in _IMG_EXTS),
                key=lambda p: str(p).lower(),
            )
            self.samples.extend((p, self.class_to_idx[cls]) for p in paths)

        if not self.samples:
            raise RuntimeError(f"[{dataset_name}] 在 {self.dom_dir} 未找到任何图像")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            pil = im.convert("RGB")
        img = self.transform(pil) if self.transform else pil
        if self.return_pil:
            return img, label, str(path), pil
        return img, label, str(path)


class _NamedFolderMultiView(Dataset):
    def __init__(self, base: Dataset, domain_id: int, img_size: int, return_appearance: bool):
        super().__init__()
        self.base = base
        self.domain_id = domain_id
        self.t_weak = get_weak_transform(img_size)
        self.t_strong = get_strong_transform(img_size)
        self.t_base = get_base_transform(img_size)
        self.return_appearance = return_appearance

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        _, y, path, pil_raw = self.base[idx]
        out = {
            "x_w": self.t_weak(pil_raw),
            "x_s": self.t_strong(pil_raw),
            "y": y,
            "domain": self.domain_id,
            "path": path,
        }
        if self.return_appearance:
            out["x_base"] = self.t_base(pil_raw)
        return out


class DomainNetDataset(_NamedFolderDataset):
    def __init__(
        self,
        root: Union[str, Path],
        domain: str,
        transform=None,
        class_names: Optional[List[str]] = None,
        return_pil: bool = False,
        recursive: bool = True,
    ):
        super().__init__(
            root, domain,
            domains=_DOMAINNET_DOMAINS,
            normalize_domain=normalize_domainnet_domain,
            folder_candidates=_DOMAINNET_FOLDERS,
            dataset_name="DomainNet",
            nested_roots=("domain_net", "DomainNet"),
            transform=transform,
            class_names=class_names,
            return_pil=return_pil,
            recursive=recursive,
        )


class DomainNetMultiView(_NamedFolderMultiView):
    def __init__(
        self,
        root: Union[str, Path],
        domain: str,
        img_size: int = 224,
        return_appearance: bool = True,
        class_names: Optional[List[str]] = None,
        recursive: bool = True,
    ):
        base = DomainNetDataset(
            root=root, domain=domain, transform=None, class_names=class_names,
            return_pil=True, recursive=recursive,
        )
        super().__init__(base, domainnet_domain_id(domain), img_size, return_appearance)


class VLCSDataset(_NamedFolderDataset):
    def __init__(
        self,
        root: Union[str, Path],
        domain: str,
        transform=None,
        class_names: Optional[List[str]] = None,
        return_pil: bool = False,
        recursive: bool = True,
    ):
        super().__init__(
            root, domain,
            domains=_VLCS_DOMAINS,
            normalize_domain=normalize_vlcs_domain,
            folder_candidates=_VLCS_FOLDERS,
            dataset_name="VLCS",
            nested_roots=("VLCS", "vlcs"),
            transform=transform,
            class_names=class_names,
            return_pil=return_pil,
            recursive=recursive,
        )


class VLCSMultiView(_NamedFolderMultiView):
    def __init__(
        self,
        root: Union[str, Path],
        domain: str,
        img_size: int = 224,
        return_appearance: bool = True,
        class_names: Optional[List[str]] = None,
        recursive: bool = True,
    ):
        base = VLCSDataset(
            root=root, domain=domain, transform=None, class_names=class_names,
            return_pil=True, recursive=recursive,
        )
        super().__init__(base, vlcs_domain_id(domain), img_size, return_appearance)
