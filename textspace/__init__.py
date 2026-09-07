# textspace/__init__.py
from .style_bank import (
    ALTERNATIVE_MIXED_12,
    DEFAULT_STYLE_WORDS,
    MIXED_STYLE_WORDS,
    STYLE_BANKS,
    get_style_bank,
)
from .text_embed import build_style_embeddings, build_style_subspace, build_class_texts

__all__ = [
    "DEFAULT_STYLE_WORDS",
    "MIXED_STYLE_WORDS",
    "ALTERNATIVE_MIXED_12",
    "STYLE_BANKS",
    "get_style_bank",
    "build_style_embeddings",
    "build_style_subspace",
    "build_class_texts",
]
