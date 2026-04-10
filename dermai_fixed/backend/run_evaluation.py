"""
run_evaluation.py  —  DermAI Guard Real Evaluation Pipeline
=============================================================
This script computes ALL metrics from ACTUAL model predictions on held-out test data.
Run AFTER training: python run_evaluation.py --data_dir ./data --model_dir ./models

Outputs:
  - models/evaluation_results.json    (all metrics, confusion matrices, learning curves)
  - models/evaluation_results.txt     (human-readable report)
  - Console: full classification reports per model

This addresses the academic requirement for a real, reproducible evaluation pipeline.
The results in evaluation.py are pre-computed benchmarks from this script.
"""

import os
import json
import time
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("evaluation.log")],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Melanoma", "Melanocytic Nevi", "Basal Cell Carcinoma",
    "Actinic Keratosis", "Benign Keratosis", "Dermatofibroma",
    "Vascular Lesion", "Squamous Cell Carcinoma", "Unknown",
]
NUM_CLASSES   = len(CLASS_NAMES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ───────────────────────────────────────────────────────────────

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class SkinDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_split(data_dir: str, split: str = "test") -> Tuple[List[str], List[int]]:
    """Load images from data_dir/{split}/{ClassName}/*.jpg"""
    paths, labels = [], []
    split_path = Path(data_dir) / split
    if not split_path.exists():
        # fallback: assume flat structure
        split_path = Path(data_dir)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_folder = split_path / cls_name.replace(" ", "_")
        if not cls_folder.exists():
            logger.warning(f"Folder not found: {cls_folder}")
            continue
        imgs = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png"))
        paths.extend([str(p) for p in imgs])
        labels.extend([cls_idx] * len(imgs))
        logger.info(f"  {cls_name}: {len(imgs)} images")
    return paths, labels


# ── Model builders ─────────────────────────────────────────────────────────────

def build_efficientnet_b4() -> nn.Module:
    model = tv_models.efficientnet_b4(
        weights=tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model.to(DEVICE)


def build_vit() -> nn.Module:
    model = tv_models.vit_b_16(
        weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1
    )
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    return model.to(DEVICE)


def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    if not os.path.exists(path):
        logger.warning(f"Checkpoint not found: {path}. Using ImageNet weights only.")
        return model
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    logger.info(f"✓ Loaded checkpoint: {path}")
    return model


# ── Core evaluation functions ──────────────────────────────────────────────────

def evaluate_deep_model(
    model: nn.Module,
    loader: DataLoader,
    model_name: str,
) -> dict:
    """
    Run model on test set. Computes:
      - Accuracy, Macro F1, Weighted F1
      - Per-class precision, recall, F1, support
      - AUC-ROC (one-vs-rest)
      - Confusion matrix (NUM_CLASSES × NUM_CLASSES)
      - Inference latency (p50, p95, p99 in ms)
      - Throughput (images/sec)
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    latencies = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            t0 = time.perf_counter()
            logits = model(imgs)
            latencies.append((time.perf_counter() - t0) * 1000 / len(imgs))  # ms/image
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Classification metrics
    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report      = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    # AUC-ROC (one-vs-rest)
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    try:
        auc_scores = roc_auc_score(y_bin, y_probs, average=None, multi_class="ovr")
        mean_auc   = float(np.mean(auc_scores))
        auc_per_class = {CLASS_NAMES[i]: round(float(auc_scores[i]), 4) for i in range(NUM_CLASSES)}
    except Exception as e:
        logger.warning(f"AUC computation failed: {e}")
        auc_scores = [0.0] * NUM_CLASSES
        mean_auc   = 0.0
        auc_per_class = {n: 0.0 for n in CLASS_NAMES}

    # Latency stats
    lats = np.array(latencies)
    p50 = float(np.percentile(lats, 50))
    p95 = float(np.percentile(lats, 95))
    p99 = float(np.percentile(lats, 99))
    throughput = round(1000.0 / p50, 1)  # images/sec

    result = {
        "model":         model_name,
        "accuracy":      round(float(acc), 4),
        "macro_f1":      round(float(macro_f1), 4),
        "weighted_f1":   round(float(weighted_f1), 4),
        "mean_auc":      round(mean_auc, 4),
        "auc_per_class": auc_per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "latency": {
            "p50_ms":            round(p50, 2),
            "p95_ms":            round(p95, 2),
            "p99_ms":            round(p99, 2),
            "throughput_img_s":  throughput,
        },
    }

    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS — {model_name}")
    logger.info(f"  Accuracy:    {acc:.4f}")
    logger.info(f"  Macro F1:    {macro_f1:.4f}   ← PRIMARY METRIC")
    logger.info(f"  Weighted F1: {weighted_f1:.4f}")
    logger.info(f"  Mean AUC:    {mean_auc:.4f}")
    logger.info(f"  Latency p50: {p50:.1f}ms  p95: {p95:.1f}ms  p99: {p99:.1f}ms")
    logger.info(f"  Throughput:  {throughput} img/s")
    logger.info(f"\nConfusion matrix:\n{cm}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)}")
    logger.info(f"{'='*60}")

    return result


def evaluate_ensemble(
    eff_model: nn.Module,
    vit_model: nn.Module,
    loader: DataLoader,
    eff_weight: float = 0.6,
    vit_weight: float = 0.4,
) -> dict:
    """Weighted ensemble: combine EfficientNet + ViT softmax outputs."""
    eff_model.eval()
    vit_model.eval()
    all_preds, all_labels, all_probs = [], [], []
    latencies = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            t0 = time.perf_counter()
            eff_probs = torch.softmax(eff_model(imgs), dim=1)
            vit_probs = torch.softmax(vit_model(imgs), dim=1)
            ensemble_probs = (eff_weight * eff_probs + vit_weight * vit_probs).cpu().numpy()
            latencies.append((time.perf_counter() - t0) * 1000 / len(imgs))

            preds = np.argmax(ensemble_probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(ensemble_probs)

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report      = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm          = confusion_matrix(y_true, y_pred)

    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    try:
        auc_scores    = roc_auc_score(y_bin, y_probs, average=None, multi_class="ovr")
        mean_auc      = float(np.mean(auc_scores))
        auc_per_class = {CLASS_NAMES[i]: round(float(auc_scores[i]), 4) for i in range(NUM_CLASSES)}
    except Exception:
        mean_auc      = 0.0
        auc_per_class = {n: 0.0 for n in CLASS_NAMES}

    lats = np.array(latencies)
    p50  = float(np.percentile(lats, 50))

    result = {
        "model":         "Deep Ensemble (EfficientNet-B4 + ViT-B/16)",
        "ensemble_weights": {"efficientnet": eff_weight, "vit": vit_weight},
        "accuracy":      round(float(acc), 4),
        "macro_f1":      round(float(macro_f1), 4),
        "weighted_f1":   round(float(weighted_f1), 4),
        "mean_auc":      round(mean_auc, 4),
        "auc_per_class": auc_per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "latency": {
            "p50_ms": round(p50, 2),
            "p95_ms": round(float(np.percentile(lats, 95)), 2),
            "p99_ms": round(float(np.percentile(lats, 99)), 2),
            "throughput_img_s": round(1000.0 / p50, 1),
        },
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"ENSEMBLE RESULTS")
    logger.info(f"  Accuracy:    {acc:.4f}")
    logger.info(f"  Macro F1:    {macro_f1:.4f}")
    logger.info(f"  Weighted F1: {weighted_f1:.4f}")
    logger.info(f"  Mean AUC:    {mean_auc:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)}")
    logger.info(f"{'='*60}")

    return result


def evaluate_traditional_ml(
    test_paths: List[str],
    test_labels: List[int],
    model_dir: str,
) -> dict:
    """Evaluate SVM and RF on HOG+LBP features."""
    from skimage.feature import hog, local_binary_pattern
    from skimage.color import rgb2gray

    logger.info("Extracting HOG+LBP features for test set…")
    feats = []
    for path in test_paths:
        try:
            img  = Image.open(path).convert("RGB")
            arr  = np.array(img.resize((128, 128)))
            gray = rgb2gray(arr)
            hog_feat = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                           cells_per_block=(1, 1), feature_vector=True)
            lbp      = local_binary_pattern(gray, P=8, R=1, method="uniform")
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
            feats.append(np.concatenate([hog_feat, lbp_hist]))
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            feats.append(np.zeros(138))  # 128 HOG + 10 LBP

    X_test = np.array(feats)
    y_test = np.array(test_labels)

    results = {}
    for model_name, fname in [("SVM", "svm.pkl"), ("Random Forest", "rf.pkl")]:
        pkl_path = os.path.join(model_dir, fname)
        if not os.path.exists(pkl_path):
            logger.warning(f"{model_name} checkpoint not found: {pkl_path}")
            continue

        with open(pkl_path, "rb") as f:
            clf = pickle.load(f)

        t0    = time.perf_counter()
        preds = clf.predict(X_test)
        latency_ms = (time.perf_counter() - t0) * 1000 / len(y_test)

        # Probabilities for AUC
        try:
            probs = clf.predict_proba(X_test)
            y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
            auc_scores    = roc_auc_score(y_bin, probs, average=None, multi_class="ovr")
            mean_auc      = float(np.mean(auc_scores))
            auc_per_class = {CLASS_NAMES[i]: round(float(auc_scores[i]), 4) for i in range(NUM_CLASSES)}
        except Exception:
            mean_auc      = 0.0
            auc_per_class = {n: 0.0 for n in CLASS_NAMES}

        macro_f1    = f1_score(y_test, preds, average="macro",    zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        report      = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        cm          = confusion_matrix(y_test, preds)

        results[model_name] = {
            "model":         model_name,
            "accuracy":      round(float(accuracy_score(y_test, preds)), 4),
            "macro_f1":      round(float(macro_f1), 4),
            "weighted_f1":   round(float(weighted_f1), 4),
            "mean_auc":      round(mean_auc, 4),
            "auc_per_class": auc_per_class,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "latency": {
                "p50_ms":           round(latency_ms, 2),
                "throughput_img_s": round(1000.0 / latency_ms, 1),
            },
        }

        logger.info(f"\n{model_name}: macro_F1={macro_f1:.4f}, weighted_F1={weighted_f1:.4f}, AUC={mean_auc:.4f}")
        logger.info(f"\n{classification_report(y_test, preds, target_names=CLASS_NAMES, zero_division=0)}")

    return results


def majority_baseline(test_labels: List[int]) -> dict:
    """
    Naive majority-class baseline: always predicts the most frequent class.
    Required by the brief as a baseline comparison.
    """
    labels_arr   = np.array(test_labels)
    majority_cls = int(np.bincount(labels_arr).argmax())
    preds        = np.full_like(labels_arr, majority_cls)

    macro_f1    = f1_score(labels_arr, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(labels_arr, preds, average="weighted", zero_division=0)
    acc         = accuracy_score(labels_arr, preds)

    result = {
        "model":            "Majority Class Baseline",
        "majority_class":   CLASS_NAMES[majority_cls],
        "accuracy":         round(float(acc), 4),
        "macro_f1":         round(float(macro_f1), 4),
        "weighted_f1":      round(float(weighted_f1), 4),
        "note": "Always predicts most frequent class. Macro F1 near 0 on imbalanced data.",
    }

    logger.info(f"\nMajority Baseline: acc={acc:.4f}, macro_F1={macro_f1:.4f}")
    logger.info(f"  Always predicts: '{CLASS_NAMES[majority_cls]}'")
    return result


# ── Learning curve evaluation ──────────────────────────────────────────────────

def compute_learning_curves(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    model_dir: str,
    fractions=(0.10, 0.25, 0.50, 0.75, 1.00),
    batch_size: int = 32,
    epochs: int = 10,
) -> dict:
    """
    For each data fraction, train EfficientNet-B4 from scratch (10 epochs)
    and record train/val F1. Demonstrates model scalability.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from train import (
        ISICDataset, train_transform, make_weighted_sampler,
        compute_loss_weights, train_epoch, validate, EarlyStopping,
    )

    val_ds     = ISICDataset(val_paths, val_labels, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {"sizes_pct": [], "train_f1": [], "val_f1": []}
    labels_arr = np.array(train_labels)

    for frac in fractions:
        n_use = max(int(len(train_paths) * frac), NUM_CLASSES)
        sss   = StratifiedShuffleSplit(n_splits=1, train_size=n_use, random_state=42)
        try:
            sub_idx, _ = next(sss.split(train_paths, labels_arr))
        except Exception:
            sub_idx = np.random.choice(len(train_paths), n_use, replace=False)

        sub_paths  = [train_paths[i] for i in sub_idx]
        sub_labels = [train_labels[i] for i in sub_idx]

        class_weights = compute_loss_weights(sub_labels)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        model     = build_efficientnet_b4()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_ds     = ISICDataset(sub_paths, sub_labels, transform=train_transform)
        sampler      = make_weighted_sampler(sub_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)

        best_train_f1, best_val_f1 = 0.0, 0.0
        for ep in range(1, epochs + 1):
            tr_loss, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, ep)
            _, val_f1, _, _ = validate(model, val_loader, criterion)
            scheduler.step()
            if val_f1 > best_val_f1:
                best_val_f1   = val_f1
                best_train_f1 = tr_f1

        pct = int(frac * 100)
        results["sizes_pct"].append(pct)
        results["train_f1"].append(round(best_train_f1, 4))
        results["val_f1"].append(round(best_val_f1, 4))
        logger.info(f"  LC {pct}%: train_F1={best_train_f1:.4f}, val_F1={best_val_f1:.4f}")

    return results


# ── Statistical data analysis ──────────────────────────────────────────────────

def statistical_analysis(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: List[str],
    test_labels: List[int],
) -> dict:
    """
    Statistical analysis of dataset characteristics:
      - Class distribution (train/val/test)
      - Imbalance ratio
      - Image statistics (mean pixel values, std)
      - Class-level image size statistics
    """
    splits = {
        "train": (train_paths, train_labels),
        "val":   (val_paths,   val_labels),
        "test":  (test_paths,  test_labels),
    }

    analysis = {}

    # Class distributions
    for split_name, (paths, labels) in splits.items():
        counts = np.bincount(np.array(labels), minlength=NUM_CLASSES)
        total  = max(sum(counts), 1)
        analysis[f"{split_name}_class_distribution"] = {
            CLASS_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)
        }
        analysis[f"{split_name}_class_percentages"] = {
            CLASS_NAMES[i]: round(100.0 * counts[i] / total, 2) for i in range(NUM_CLASSES)
        }
        analysis[f"{split_name}_total"] = int(total)

    # Imbalance ratio (max class / min class in training set)
    train_counts = np.bincount(np.array(train_labels), minlength=NUM_CLASSES)
    min_count = max(train_counts.min(), 1)
    max_count = train_counts.max()
    analysis["imbalance_ratio"] = round(float(max_count / min_count), 2)
    analysis["most_frequent_class"]  = CLASS_NAMES[train_counts.argmax()]
    analysis["least_frequent_class"] = CLASS_NAMES[train_counts.argmin()]

    # Pixel statistics on a random 200-image sample
    sample_paths  = train_paths[:200]
    pixel_means_r, pixel_means_g, pixel_means_b = [], [], []
    image_widths, image_heights = [], []

    for path in sample_paths:
        try:
            img = Image.open(path).convert("RGB")
            image_widths.append(img.width)
            image_heights.append(img.height)
            arr = np.array(img).astype(np.float32) / 255.0
            pixel_means_r.append(float(arr[:, :, 0].mean()))
            pixel_means_g.append(float(arr[:, :, 1].mean()))
            pixel_means_b.append(float(arr[:, :, 2].mean()))
        except Exception:
            pass

    if pixel_means_r:
        analysis["pixel_statistics"] = {
            "mean_R": round(float(np.mean(pixel_means_r)), 4),
            "mean_G": round(float(np.mean(pixel_means_g)), 4),
            "mean_B": round(float(np.mean(pixel_means_b)), 4),
            "std_R":  round(float(np.std(pixel_means_r)),  4),
            "std_G":  round(float(np.std(pixel_means_g)),  4),
            "std_B":  round(float(np.std(pixel_means_b)),  4),
        }
        analysis["image_size_statistics"] = {
            "mean_width":  round(float(np.mean(image_widths)),  1),
            "mean_height": round(float(np.mean(image_heights)), 1),
            "std_width":   round(float(np.std(image_widths)),   1),
            "std_height":  round(float(np.std(image_heights)),  1),
            "min_width":   int(min(image_widths)),
            "min_height":  int(min(image_heights)),
        }

    logger.info("\nStatistical Analysis:")
    logger.info(f"  Training images: {analysis['train_total']}")
    logger.info(f"  Imbalance ratio: {analysis['imbalance_ratio']:.1f}x "
                f"({analysis['most_frequent_class']} vs {analysis['least_frequent_class']})")

    return analysis


# ── Error analysis ─────────────────────────────────────────────────────────────

def error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    top_k: int = 5,
) -> dict:
    """
    Detailed error analysis:
      - Most confused class pairs
      - Low-confidence correct predictions
      - High-confidence wrong predictions
    """
    cm = confusion_matrix(y_true, y_pred)

    # Most confused pairs (off-diagonal elements)
    confused_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true":            CLASS_NAMES[i],
                    "predicted":       CLASS_NAMES[j],
                    "count":           int(cm[i, j]),
                    "pct_of_true_cls": round(100.0 * cm[i, j] / max(cm[i].sum(), 1), 1),
                })

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Confidence analysis
    max_conf = y_probs.max(axis=1)
    correct  = (y_true == y_pred)

    # Wrong predictions with high confidence (model overconfidence)
    wrong_high_conf = []
    wrong_mask = ~correct & (max_conf > 0.8)
    for idx in np.where(wrong_mask)[0][:20]:
        wrong_high_conf.append({
            "true":       CLASS_NAMES[y_true[idx]],
            "predicted":  CLASS_NAMES[y_pred[idx]],
            "confidence": round(float(max_conf[idx]), 3),
        })

    # Per-class accuracy
    per_class_acc = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            per_class_acc[cls_name] = round(float(correct[mask].mean()), 4)

    return {
        "most_confused_pairs":      confused_pairs[:top_k],
        "high_conf_wrong_preds":    wrong_high_conf[:10],
        "per_class_accuracy":       per_class_acc,
        "overall_error_rate":       round(float(1 - correct.mean()), 4),
        "mean_correct_confidence":  round(float(max_conf[correct].mean()), 4) if correct.any() else 0.0,
        "mean_wrong_confidence":    round(float(max_conf[~correct].mean()), 4) if (~correct).any() else 0.0,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DermAI Guard Evaluation Pipeline")
    parser.add_argument("--data_dir",   type=str, default="./data",   help="Data directory")
    parser.add_argument("--model_dir",  type=str, default="./models", help="Model checkpoints dir")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split",      type=str, default="test",
                        help="Which split to evaluate on (test/val)")
    parser.add_argument("--learning_curves", action="store_true",
                        help="Also run learning curve experiments (slow, requires data)")
    parser.add_argument("--traditional_only", action="store_true",
                        help="Only evaluate traditional ML models")
    args = parser.parse_args()

    logger.info(f"\n{'='*60}\nDermAI Guard Evaluation Pipeline\n{'='*60}")
    logger.info(f"Device: {DEVICE}")

    # Load test data
    logger.info(f"\nLoading {args.split} data from {args.data_dir}…")
    test_paths, test_labels = load_split(args.data_dir, args.split)

    if len(test_paths) == 0:
        logger.error(f"No images found in {args.data_dir}/{args.split}/")
        logger.error("Expected structure: data/{split}/{ClassName}/*.jpg")
        return

    logger.info(f"Total {args.split} images: {len(test_paths)}")

    all_results = {}

    # Majority baseline (always required)
    logger.info("\n=== MAJORITY CLASS BASELINE ===")
    all_results["majority_baseline"] = majority_baseline(test_labels)

    # Traditional ML
    logger.info("\n=== TRADITIONAL ML EVALUATION ===")
    ml_results = evaluate_traditional_ml(test_paths, test_labels, args.model_dir)
    all_results["traditional_ml"] = ml_results

    if not args.traditional_only:
        # Deep models
        test_ds     = SkinDataset(test_paths, test_labels, transform=val_transform)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        logger.info("\n=== EFFICIENTNET-B4 EVALUATION ===")
        eff_model = build_efficientnet_b4()
        eff_model = load_checkpoint(eff_model, os.path.join(args.model_dir, "efficientnet_b4.pth"))
        eff_results = evaluate_deep_model(eff_model, test_loader, "EfficientNet-B4")
        all_results["efficientnet_b4"] = eff_results

        logger.info("\n=== ViT-B/16 EVALUATION ===")
        vit_model = build_vit()
        vit_model = load_checkpoint(vit_model, os.path.join(args.model_dir, "vit_b16.pth"))
        vit_results = evaluate_deep_model(vit_model, test_loader, "ViT-B/16")
        all_results["vit_b16"] = vit_results

        logger.info("\n=== DEEP ENSEMBLE EVALUATION ===")
        ensemble_results = evaluate_ensemble(eff_model, vit_model, test_loader)
        all_results["deep_ensemble"] = ensemble_results

        # Error analysis on ensemble
        logger.info("\n=== ERROR ANALYSIS ===")
        eff_model.eval()
        vit_model.eval()
        all_preds, all_labels_arr, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                ep = torch.softmax(eff_model(imgs), dim=1)
                vp = torch.softmax(vit_model(imgs), dim=1)
                probs = (0.6 * ep + 0.4 * vp).cpu().numpy()
                all_preds.extend(np.argmax(probs, axis=1))
                all_labels_arr.extend(labels.numpy())
                all_probs.extend(probs)
        error_analysis_results = error_analysis(
            np.array(all_labels_arr),
            np.array(all_preds),
            np.array(all_probs),
        )
        all_results["error_analysis"] = error_analysis_results

        logger.info("\nMost confused pairs:")
        for pair in error_analysis_results["most_confused_pairs"]:
            logger.info(f"  {pair['true']} → {pair['predicted']}: {pair['count']} ({pair['pct_of_true_cls']}%)")

    # Learning curves (optional, slow)
    if args.learning_curves:
        logger.info("\n=== LEARNING CURVES ===")
        train_paths, train_labels = load_split(args.data_dir, "train")
        val_paths, val_labels     = load_split(args.data_dir, "val")
        lc = compute_learning_curves(train_paths, train_labels, val_paths, val_labels, args.model_dir)
        all_results["learning_curves"] = lc

    # Statistical analysis
    logger.info("\n=== STATISTICAL DATA ANALYSIS ===")
    train_paths, train_labels = load_split(args.data_dir, "train")
    val_paths, val_labels     = load_split(args.data_dir, "val")
    stats = statistical_analysis(
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
    )
    all_results["statistical_analysis"] = stats

    # Model comparison summary
    summary = []
    for key, model_name, mtype in [
        ("majority_baseline",  "Majority Baseline",        "Baseline"),
        ("traditional_ml",     "SVM (HOG+LBP)",            "Traditional ML"),
        ("traditional_ml",     "Random Forest (HOG+LBP)",  "Traditional ML"),
        ("efficientnet_b4",    "EfficientNet-B4",           "Deep Learning"),
        ("vit_b16",            "ViT-B/16",                  "Deep Learning"),
        ("deep_ensemble",      "Deep Ensemble",             "Deep Ensemble"),
    ]:
        res = all_results.get(key, {})
        if key == "traditional_ml":
            res = res.get(model_name.split(" ")[0], {})
        if res:
            summary.append({
                "model":       model_name,
                "type":        mtype,
                "macro_f1":    res.get("macro_f1",    0.0),
                "weighted_f1": res.get("weighted_f1", 0.0),
                "mean_auc":    res.get("mean_auc",    0.0),
                "latency_p50": res.get("latency", {}).get("p50_ms", 0),
            })

    all_results["model_comparison"] = summary

    # Save JSON
    out_path = os.path.join(args.model_dir, "evaluation_results.json")
    os.makedirs(args.model_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✓ Results saved to {out_path}")

    # Print final summary table
    logger.info(f"\n{'='*70}")
    logger.info("FINAL MODEL COMPARISON")
    logger.info(f"{'Model':<35} {'Type':<18} {'MacroF1':>8} {'WeightedF1':>11} {'AUC':>7}")
    logger.info("-" * 70)
    for row in summary:
        logger.info(
            f"{row['model']:<35} {row['type']:<18} "
            f"{row['macro_f1']:>8.4f} {row['weighted_f1']:>11.4f} {row['mean_auc']:>7.4f}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
