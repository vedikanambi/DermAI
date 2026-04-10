# DermAI Guard — Training Guide

## Why the classifier was failing (and what was fixed)

| ❌ Original Issue | ✅ Fix Applied |
|---|---|
| EfficientNet-**B7** (66M params) — overfits on ISIC subset | Switched to EfficientNet-**B4** (19M params) — stable training |
| Plain `CrossEntropyLoss` — ignores rare classes | `CrossEntropyLoss(weight=class_weights)` — `compute_class_weight('balanced')` |
| No class rebalancing at batch level | `WeightedRandomSampler` — minority classes oversampled |
| Random train/val/test split | `StratifiedShuffleSplit` — class ratios preserved in all splits |
| Augmentation on ALL splits (data leakage) | Augmentation on **training only**; val/test use only resize+normalize |
| `SGD` or default `Adam` | `AdamW(lr=3e-5, weight_decay=1e-4)` + `CosineAnnealingLR` |
| No early stopping → overfitting | `EarlyStopping(patience=7)` on val macro-F1 |
| Using **accuracy** as metric | Using **macro F1** — accuracy is misleading on imbalanced ISIC data |
| Equal ensemble weights (0.5 + 0.5) | Weighted ensemble: EfficientNet ×0.6 + ViT ×0.4 |
| No gradient clipping (ViT training instability) | `clip_grad_norm_(model.parameters(), max_norm=1.0)` |

## Step 1: Download ISIC data

```bash
# Option A: From the frontend Evaluation tab → "Start Download"
# Option B: Directly
cd backend
python -c "
from dataset_loader import build_dataset_for_training
result = build_dataset_for_training(limit_per_class=200, save_dir='./data')
print(result)
"
```

This creates `./data/{ClassName}/*.jpg` for all 9 ISIC classes.

## Step 2: Train Traditional ML (fast, ~5 min)

```bash
cd backend
python train.py --model traditional --data_dir ./data --model_dir ./models
```

Trains: SVM (RBF, `class_weight=balanced`) + Random Forest (500 trees) + Stacking Ensemble  
Saves: `models/svm.pkl`, `models/rf.pkl`, `models/stacking.pkl`

## Step 3: Train EfficientNet-B4 (recommended: GPU)

```bash
# CPU (~2-4 hours per model)
python train.py --model efficientnet_b4 --epochs 30 --data_dir ./data --model_dir ./models

# GPU (recommended, ~30-60 min)
CUDA_VISIBLE_DEVICES=0 python train.py --model efficientnet_b4 --epochs 30
```

Saves: `models/efficientnet_b4.pth` (best val F1 checkpoint)

## Step 4: Train ViT-B/16

```bash
python train.py --model vit --epochs 30 --data_dir ./data --model_dir ./models
```

Saves: `models/vit_b16.pth`

## Step 5: Train everything + learning curves

```bash
python train.py --model all --epochs 30 --learning_curves --data_dir ./data --model_dir ./models
```

Saves: `models/training_results.json` (all metrics for Evaluation dashboard)

## Training transforms (training set only)

```python
T.RandomResizedCrop(224, scale=(0.8, 1.0)),
T.RandomHorizontalFlip(),
T.RandomVerticalFlip(),
T.RandomRotation(15),
T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
T.ToTensor(),
T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
```

## Validation / Test transforms (no augmentation)

```python
T.Resize((224, 224)),
T.ToTensor(),
T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
```

## Expected results (with 200 images/class)

| Model | Macro F1 | AUC |
|---|---|---|
| SVM (HOG+LBP) | ~0.55–0.65 | ~0.80 |
| Random Forest | ~0.60–0.70 | ~0.83 |
| EfficientNet-B4 | ~0.78–0.85 | ~0.90 |
| ViT-B/16 | ~0.76–0.83 | ~0.89 |
| Deep Ensemble | ~0.82–0.86 | ~0.93 |

With 500+ images/class, expect +5–8% F1 across all models.

## Once trained: inference

Restart the backend — it auto-loads from `MODEL_DIR`:
```bash
uvicorn main:app --reload --port 8000
```

The classifier will exit demo mode and use your fine-tuned weights automatically.
