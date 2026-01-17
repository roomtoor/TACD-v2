# textspace/__init__.py
from .style_bank import DEFAULT_STYLE_WORDS
from .text_embed import build_style_subspace, build_class_texts

__all__ = [
    "DEFAULT_STYLE_WORDS",
    "build_style_subspace",
    "build_class_texts",
]