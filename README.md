# TACD
> **说明**  
> 本项目的完整代码已开源并托管于 GitHub：  
> https://github.com/roomtoor/TACD-v2

## 1. Datasets

由于项目体积限制，本项目仓库中不包含数据集文件。
请根据下方提供的 official link 自行下载数据集，并按如下方式放置。

```text
TACD_v2/
├── OfficeHomeDataset/
│   ├── Art/
│   ├── Clipart/
│   ├── Product/
│   └── Real World/
└── TerraIncognitaDataset/
    └── terra_incognita/
        ├── location_38/
        ├── location_43/
        ├── location_46/
        ├── location_100/
        └── ...
```

### Office-Home

- **Dataset name**: Office-Home

- **Official link**: https://www.hemanthdv.org/OfficeHome-Dataset/

- **Expected directory structure**:

    ```text
    OfficeHomeDataset/
    ├── Art/
    │   ├── class_1/
    │   │   └── *.jpg
    │   ├── class_2/
    │   │   └── *.jpg
    │   └── ...
    ├── Clipart/
    ├── Product/
    └── Real World/
    ```

### TerraIncognita (DomainBed setting)

- **Dataset name**: TerraIncognita

- **Official link**: https://github.com/facebookresearch/DomainBed

- **Expected directory structure**:

    ```text
    TerraIncognitaDataset/
    └── terra_incognita/
        ├── location_38/
        ├── location_43/
        ├── location_46/
        ├── location_100/
        └── ...
    ```

路径大小写与空格需与上述结构保持一致。

## 2. Environment

- 本项目的 Python 依赖已整理在 `requirements.txt` 中。
- 已在全新环境中通过 `pip install -r requirements.txt` 验证可正常运行。

> PyTorch 请根据本地 CUDA / GPU 驱动版本自行安装兼容版本，其余依赖可直接通过 `requirements.txt` 安装。

## 3. Training

- **随机种子与确定性设置**：训练过程中固定随机种子（seed=3），并启用 deterministic 选项以保证结果可复现。

- **Office-Home 主实验训练命令（示例）**：

      python run_train.py \
        --dataset officehome \
        --root ./OfficeHomeDataset \
        --source "Real World" \
        --epochs 30 \
        --nan_guard

- **TerraIncognita（DomainBed setting）主实验训练命令**：

      python run_train.py \
        --dataset terraincognita \
        --root ./TerraIncognitaDataset/terra_incognita \
        --source location_46 \
        --epochs 30

## 4. Evaluation

- **Office-Home 测试命令（示例）**：

      python test_tacd_v2.py \
        --dataset officehome \
        --root ./OfficeHomeDataset \
        --source Art \
        --ckpt ./checkpoints/TACDv2_OfficeHome_3_SSDG_Art_ep25.pth

- **TerraIncognita 测试命令（示例）**：

      python test_tacd_v2.py \
        --dataset terraincognita \
        --root ./TerraIncognitaDataset/terra_incognita \
        --source location_46 \
        --ckpt ./checkpoints/TACDv2_SSDG_SSDG_terraincognita_location_46_ep10.pth

- **评测指标**：分类准确率（Accuracy），并报告各目标域结果、平均准确率（Mean）以及最差域准确率（Worst-domain）。

## 5. Checkpoints and Logs

- **模型权重（HuggingFace）**  
  
  训练完成的模型权重已公开上传至 HuggingFace： https://huggingface.co/javaccd/TACD-v2

- **可复现性说明**  
  
  所有主实验均固定随机种子并启用 deterministic 设置。  
  
  通过本文档中提供的训练与测试命令，可复现实验结果。
