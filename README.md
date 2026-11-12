# Solar Filament Instance Segmentation
### IEEE Big Data Cup 2026 — Kaggle Competition

**Public LB: 0.41 | OOF PQ: 0.4665**

## Team

| Member | Institution |
|---|---|
| Daichi Sasaki | Hiroshima University, Japan |
| Saif Nasr | Chonnam National University, South Korea |
| Yujeong Lee | Kyungpook National University, South Korea |
| Seungyoon Im | Kyungpook National University, South Korea |
| Seo-hoo Han | Pusan National University, South Korea |

## Pipeline

```
H-alpha solar image (2048×2048)
 ├─ U-Net ResNet-34 × 5 folds  ┐  D4 TTA · overlapping tiles · Hann blending
 ├─ U-Net EfficientNet-B4 × 5  ┘  → 10 probability maps averaged
 │      └─ component tree (19 threshold levels) → candidate instances
 │            └─ 36 node features (shape, probability, intensity, tree structure)
 └─ RF-DETR-Seg-Large × 5 folds → instance masks
        └─ 4 detector features per node
 BlendSelector: hurdle (LightGBM) + LambdaRank · α=0.02 · τ=0.24
 DP antichain → RLE masks → submission.csv
```

## Setup

```bash
git clone https://github.com/seifByte/solar-filament-segmentation-2026.git
cd solar-filament-segmentation-2026

# Install torch first (match your CUDA version)
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

Download weights and cache from HuggingFace:
https://huggingface.co/datasets/Saifds10/solar-filament-segmentation-2026

Download the competition data from Kaggle:
https://www.kaggle.com/competitions/filament-segmentation-2026/data

Place them as:
```
weights/r34/fold{0-4}/last.pth
weights/b4/fold{0-4}/last.pth
weights/rfdetr/fold{0-4}/checkpoint_best_ema.pth.part{0,1}
models/selector/
cache/tree_ens_r34_b4.pkl
```

Place competition data at:
```
data/MAGFiLO_1.0_Kaggle_2026/
```

## Run

Open `notebooks/pipeline.ipynb` and run all cells.
Sections 5, 6, 8, 10 and 12.3 are skippable using the provided pretrained weights and cache.

Output: `submission.csv`

## Full Reproducibility (optional)

The following files are needed to rebuild the OOF cache and retrain the selector from scratch.
They are not required to reproduce the submission CSV.

```
cache/rfdetr_feats_fold{0-4}.pkl  ← RF-DETR OOF features (needed for Section 12.3)
maps_ens/*.npz                     ← OOF ensemble prob maps (needed for Section 10)
```

These will be added to HuggingFace when available.

## Requirements

Python >= 3.11 · CUDA GPU (tested on RTX A6000 48GB) · see `requirements.txt`