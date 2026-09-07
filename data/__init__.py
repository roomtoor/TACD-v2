# data/__init__.py

from .augment import (
    get_weak_transform,
    get_strong_transform,
    get_base_transform,
)

from .datasets import (
    # -------- Office-Home --------
    OfficeHomeDataset,
    OfficeHomeMultiView,
    scan_officehome_source_classes,
    officehome_prompt_names,
    officehome_domain_id,

    # -------- TerraIncognita --------
    TerraIncognitaDataset,
    TerraIncognitaMultiView,
    scan_terraincognita_source_classes,
    terraincognita_domain_id,

    # -------- DomainNet --------
    DomainNetDataset,
    DomainNetMultiView,
    scan_domainnet_source_classes,
    domainnet_prompt_names,
    domainnet_domain_id,
    normalize_domainnet_domain,

    # -------- VLCS --------
    VLCSDataset,
    VLCSMultiView,
    scan_vlcs_source_classes,
    vlcs_prompt_names,
    vlcs_domain_id,
    normalize_vlcs_domain,
)

__all__ = [
    # augment
    "get_weak_transform",
    "get_strong_transform",
    "get_base_transform",

    # Office-Home
    "OfficeHomeDataset",
    "OfficeHomeMultiView",
    "scan_officehome_source_classes",
    "officehome_prompt_names",
    "officehome_domain_id",

    # TerraIncognita
    "TerraIncognitaDataset",
    "TerraIncognitaMultiView",
    "scan_terraincognita_source_classes",
    "terraincognita_domain_id",

    # DomainNet
    "DomainNetDataset",
    "DomainNetMultiView",
    "scan_domainnet_source_classes",
    "domainnet_prompt_names",
    "domainnet_domain_id",
    "normalize_domainnet_domain",

    # VLCS
    "VLCSDataset",
    "VLCSMultiView",
    "scan_vlcs_source_classes",
    "vlcs_prompt_names",
    "vlcs_domain_id",
    "normalize_vlcs_domain",
]
