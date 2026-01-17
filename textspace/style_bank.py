# textspace/style_bank.py
"""
定义风格词表，用于构建风格子空间 E_s。
你也可以自行扩充，例如加入 DomainNet 的 domain-specific 风格词。
"""

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
