# TASIL

Official research implementation of **Text-Anchored Style Invariance Learning for Single-Source Domain Generalization**.

TASIL constructs a style subspace from textual style descriptors encoded by a frozen CLIP model, suppresses style-aligned components in visual representations, and trains with weak, strong, and text-guided appearance feature views. The implementation follows a strict single-source domain generalization (SDG) protocol: only one labeled source domain is used for training, and held-out target domains are used only for final evaluation.

## Release scope

This repository contains the main TASIL method, dataset loaders, training code, and evaluation code used for the paper's principal experiments. It intentionally excludes local datasets, checkpoints, logs, virtual environments, and auxiliary scripts for baselines or figure generation.

The reported paper experiments use Office-Home and TerraIncognita. Loaders for DomainNet and VLCS are also included, but results for those datasets are not claimed in the paper.

## Environment

The reference environment uses Python 3.9, PyTorch 2.4.1, and torchvision 0.19.1. Create a clean environment instead of reusing a copied virtual environment:

```bash
conda create -n tasil python=3.9 -y
conda activate tasil
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA driver if the default wheel is unsuitable for your machine. The first run may download the frozen CLIP ViT-B/16 weights.

## Datasets

Datasets are not redistributed. Download each dataset from its official source and arrange it as follows.

### Office-Home

Download: [Office-Home dataset](https://www.hemanthdv.org/OfficeHome-Dataset/)

```text
OfficeHomeDataset/
├── Art/<class_name>/*
├── Clipart/<class_name>/*
├── Product/<class_name>/*
└── Real World/<class_name>/*
```

### TerraIncognita

Download and preprocessing reference: [DomainBed](https://github.com/facebookresearch/DomainBed)

```text
TerraIncognitaDataset/
└── terra_incognita/
    ├── location_38/<class_name>/*
    ├── location_43/<class_name>/*
    ├── location_46/<class_name>/*
    └── location_100/<class_name>/*
```

The loaders form one consistent label mapping across all domains. Directory names, spaces, and capitalization must match the structures above.

## Training

The paper reports three runs with seeds `3`, `5201314`, and `30319`. Training uses one source domain for 30 epochs and saves the fixed final-epoch checkpoint. Target-domain samples are not loaded during training or model selection.

Office-Home example:

```bash
python run_train.py \
  --dataset officehome \
  --root ./OfficeHomeDataset \
  --source "Real World" \
  --seed 3 \
  --epochs 30 \
  --nan_guard
```

TerraIncognita example:

```bash
python run_train.py \
  --dataset terraincognita \
  --root ./TerraIncognitaDataset/terra_incognita \
  --source location_46 \
  --seed 3 \
  --epochs 30 \
  --nan_guard
```

Repeat each experiment for every source domain and each reported seed. Checkpoints are written to `checkpoints/`; logs are written to `logs/`.

## Evaluation

Evaluate one checkpoint on every held-out domain:

```bash
python evaluate.py \
  --dataset officehome \
  --root ./OfficeHomeDataset \
  --source Art \
  --seed 3 \
  --ckpt ./checkpoints/TASIL_SSDG_GroupDRO_SSDG_officehome_Art_seed3_ep30.pth
```

```bash
python evaluate.py \
  --dataset terraincognita \
  --root ./TerraIncognitaDataset/terra_incognita \
  --source location_46 \
  --seed 3 \
  --ckpt ./checkpoints/TASIL_SSDG_GroupDRO_SSDG_terraincognita_location_46_seed3_ep30.pth
```

The evaluation script prints per-target accuracy, mean accuracy, and worst-domain accuracy. Use `--per_class` to save per-class accuracy arrays.

## Main results

Each source column reports the mean accuracy over the other three unseen domains. Values are percentages averaged over three seeds.

| Dataset | Source 1 | Source 2 | Source 3 | Source 4 | Average |
|---|---:|---:|---:|---:|---:|
| Office-Home (Art / Clipart / Product / Real World) | 82.31 | 86.30 | 81.88 | 83.76 | 83.56 |
| TerraIncognita (L38 / L43 / L46 / L100) | 33.17 | 41.42 | 44.08 | 33.57 | 38.06 |

## Repository structure

```text
.
├── data/             # Dataset readers and augmentations
├── losses/           # Consistency and GroupDRO objectives
├── models/           # CLIP backbone, projection head, and TASIL model
├── textspace/        # Class prompts and textual style subspace
├── utils/            # Seeding, logging, schedules, and checkpoint helpers
├── cfg.py            # Default experiment configuration
├── run_train.py      # Single-source training entry point
├── train_utils.py    # Training and loader utilities
├── evaluate.py       # Held-out-domain evaluation entry point
└── requirements.txt
```

## Reproducibility notes

- The CLIP image and text encoders remain frozen.
- The default backbone is CLIP ViT-B/16.
- The effective style-suppression coefficient is `sigmoid(alpha)`; the learnable raw parameter starts at `alpha = 0`, corresponding to an initial effective value of `0.5`.
- The final training epoch is selected in advance; target-domain accuracy is not used for checkpoint selection.
- Exact reproducibility can still depend on GPU hardware, CUDA, cuDNN, and third-party library behavior.

## License

This project is released under the [MIT License](LICENSE).
