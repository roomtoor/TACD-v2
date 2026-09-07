# textspace/style_bank.py
"""Fixed descriptor banks used for appearance views and style suppression."""

from typing import Dict, List

DEFAULT_STYLE_WORDS = [
    # ────── 画风相关 ──────
    "sketch style", "line art", "pencil drawing", "cartoon style",
    "comic style", "manga style", "watercolor style", "oil painting",
    "posterized image", "flat color style", "crosshatch shading",
    # ────── 成像/光照相关 ──────
    "monochrome image", "high contrast photo", "low contrast photo",
    "noisy image", "grainy image", "low light photo", "backlit scene",
    "overexposed photo", "underexposed photo",
    "desaturated photo", "vivid color image",
    "blurred image", "motion blur photo", "gaussian blur image",
    # ────── 材质/渲染相关 ──────
    "digital painting", "3D render", "illustration style", "crayon drawing"
]


# The stratified order used by the descriptor-number study.  It alternates
# artistic/rendering and photometric/imaging concepts so that a prefix changes
# bank size without abruptly changing semantic coverage.
MIXED_STYLE_WORDS = [
    "sketch style", "monochrome image", "cartoon style", "blurred image",
    "oil painting", "low light photo", "watercolor style", "high contrast photo",
    "line art", "noisy image", "3D render", "desaturated photo",
    "pencil drawing", "motion blur photo", "comic style", "backlit scene",
    "digital painting", "overexposed photo", "manga style", "low contrast photo",
    "illustration style", "underexposed photo", "posterized image",
    "vivid color image", "flat color style", "grainy image",
    "crosshatch shading", "gaussian blur image", "crayon drawing",
]


# Disjoint, matched control bank for the shared-vs-unshared experiment.
ALTERNATIVE_MIXED_12 = [
    "pencil drawing", "comic style", "manga style", "digital painting",
    "illustration style", "crayon drawing", "motion blur photo",
    "backlit scene", "overexposed photo", "low contrast photo",
    "underexposed photo", "vivid color image",
]


STYLE_BANKS: Dict[str, List[str]] = {
    # Keep the historical ordering as the default so existing main experiments
    # remain behaviorally unchanged under the same random seed.
    "default-29": DEFAULT_STYLE_WORDS,
    "mixed-12": MIXED_STYLE_WORDS[:12],
    "alt-mixed-12": ALTERNATIVE_MIXED_12,
}


def get_style_bank(name: str) -> List[str]:
    """Return a copy of a named fixed descriptor bank."""
    key = name.strip().lower()
    if key not in STYLE_BANKS:
        raise ValueError(
            f"Unknown style bank {name!r}; expected one of {sorted(STYLE_BANKS)}"
        )
    return list(STYLE_BANKS[key])
