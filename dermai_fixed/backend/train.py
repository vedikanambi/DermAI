"""
train.py  —  DermAI Guard Training Pipeline
============================================
Run: python train.py [--data_dir ./data] [--epochs 30] [--model efficientnet_b4|vit|both]

Fixes applied vs. original failing classifier:
  ✔ EfficientNet-B4 instead of B7 (more stable, less overfit on ISIC scale)
  ✔ Class-weighted CrossEntropyLoss  (balances severe ISIC class imbalance)
  ✔ WeightedRandomSampler            (oversample minority at batch level)
  ✔ StratifiedShuffleSplit           (preserves class ratios in all splits)
  ✔ Training-only augmentation       (NO augmentation on val/test → no leakage)
  ✔ AdamW + CosineAnnealingLR        (better convergence than SGD)
  ✔ Early stopping (patience=7)      (stops overfitting automatically)
  ✔ F1-macro as primary metric       (not accuracy — accuracy misleads on imbalanced data)
  ✔ Best checkpoint saved on val F1  (not val loss)
  ✔ Traditional ML training          (HOG+LBP → SVM/RF, saved as .pkl)
  ✔ Learning curves at 5 sizes       (10/25/50/75/100% of training set)
  ✔ Full evaluation: F1, AUC, confusion matrix, per-class report
"""

import os
import json
import time
import pickle
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv_models

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training device: {DEVICE}")


# ── Transforms ─────────────────────────────────────────────────────────────────

# TRAINING: augmentation applied ONLY to training set
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# VALIDATION / TEST: NO augmentation (prevents data leakage)
val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Dataset ────────────────────────────────────────────────────────────────────

class ISICDataset(Dataset):
    """
    Loads ISIC images from data_dir/{ClassName}/*.jpg
    Supports separate transforms for train vs. val/test.
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx: int):
        path  = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to open {path}: {e}. Using blank image.")
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


def load_image_paths(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Scan data_dir for class folders matching CLASS_NAMES.
    Returns (paths, labels) with integer class indices.
    """
    paths, labels = [], []
    data_path = Path(data_dir)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_folder = data_path / cls_name.replace(" ", "_")
        if not cls_folder.exists():
            logger.warning(f"Class folder not found: {cls_folder}")
            continue
        imgs = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png"))
        logger.info(f"  {cls_name}: {len(imgs)} images")
        paths.extend([str(p) for p in imgs])
        labels.extend([cls_idx] * len(imgs))
    return paths, labels


def stratified_split(
    paths: List[str], labels: List[int],
    val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> Tuple[List, List, List, List, List, List]:
    """
    Stratified 70/15/15 split using StratifiedShuffleSplit.
    Preserves class ratios in train, val, and test sets.
    """
    labels_arr = np.array(labels)

    # First: split off test set
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(paths, labels_arr))

    trainval_paths  = [paths[i]  for i in trainval_idx]
    trainval_labels = [labels[i] for i in trainval_idx]

    # Second: split val from remaining train+val
    val_fraction = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss2.split(trainval_paths, trainval_labels))

    train_paths  = [trainval_paths[i]  for i in train_idx]
    train_labels = [trainval_labels[i] for i in train_idx]
    val_paths    = [trainval_paths[i]  for i in val_idx]
    val_labels   = [trainval_labels[i] for i in val_idx]
    test_paths   = [paths[i]  for i in test_idx]
    test_labels  = [labels[i] for i in test_idx]

    logger.info(f"Split: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def make_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    WeightedRandomSampler: each class gets equal expected sampling frequency.
    Effectively oversamples rare classes at the batch level.
    """
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(labels),
        replacement=True,
    )


# ── Model builders ────────────────────────────────────────────────────────────

def build_efficientnet_b4() -> nn.Module:
    """
    EfficientNet-B4 with Dropout(0.4) head.
    Reason for B4 over B7:
      - B7 (66M params) overfits aggressively on ISIC subset (~25k images)
      - B4 (19M params) balances capacity and regularisation
      - Training time: B4 ~3× faster than B7 on CPU
    """
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


# ── Class-weighted loss ───────────────────────────────────────────────────────

def compute_loss_weights(train_labels: List[int]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    This is the MOST IMPORTANT fix for the failing classifier.
    Without this, the model predicts the majority class (Melanocytic Nevi ~67%)
    and achieves "high accuracy" but near-zero F1 on rare classes.
    """
    labels_arr = np.array(train_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=labels_arr,
    )
    logger.info("Class weights (higher = rarer class):")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
        logger.info(f"  [{i}] {name}: {w:.3f}")
    return torch.FloatTensor(weights).to(DEVICE)


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when val F1-macro does not improve for `patience` epochs.
    Saves best model checkpoint automatically.
    """
    def __init__(self, patience: int = 7, min_delta: float = 0.001,
                 checkpoint_path: str = "best_model.pth"):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = checkpoint_path
        self.best_score       = -np.inf
        self.counter          = 0
        self.stopped          = False

    def __call__(self, val_f1: float, model: nn.Module) -> bool:
        if val_f1 > self.best_score + self.min_delta:
            self.best_score = val_f1
            self.counter    = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_f1": val_f1}, self.checkpoint_path)
            logger.info(f"  ✓ Checkpoint saved (val F1: {val_f1:.4f})")
            return False
        else:
            self.counter += 1
            logger.info(f"  EarlyStopping: {self.counter}/{self.patience} (best={self.best_score:.4f})")
            if self.counter >= self.patience:
                logger.info("  ✗ Early stopping triggered.")
                self.stopped = True
                return True
        return False


# ── Training epoch ────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, epoch: int) -> Tuple[float, float]:
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        # Gradient clipping prevents exploding gradients (especially ViT)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, loader, criterion) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1, np.array(all_labels), np.array(all_probs)


# ── Full evaluation ────────────────────────────────────────────────────────────

def full_evaluation(model, loader, split_name: str = "test") -> dict:
    """
    Compute comprehensive metrics: accuracy, macro F1, weighted F1,
    per-class F1, AUC-ROC (one-vs-rest), confusion matrix.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc          = accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None,       zero_division=0)
    cm           = confusion_matrix(y_true, y_pred)
    report       = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

    # AUC-ROC (one-vs-rest, requires at least 2 classes present)
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    try:
        auc_per_class = roc_auc_score(y_bin, y_probs, average=None, multi_class="ovr")
        mean_auc = float(np.mean(auc_per_class))
    except Exception:
        auc_per_class = [0.0] * NUM_CLASSES
        mean_auc = 0.0

    results = {
        "split":         split_name,
        "accuracy":      round(acc, 4),
        "macro_f1":      round(macro_f1, 4),
        "weighted_f1":   round(weighted_f1, 4),
        "mean_auc":      round(mean_auc, 4),
        "per_class_f1":  {CLASS_NAMES[i]: round(float(per_class_f1[i]), 4) for i in range(NUM_CLASSES)},
        "auc_per_class": {CLASS_NAMES[i]: round(float(auc_per_class[i]), 4) for i in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION — {split_name.upper()}")
    logger.info(f"  Accuracy:    {acc:.4f}")
    logger.info(f"  Macro F1:    {macro_f1:.4f}   ← PRIMARY METRIC")
    logger.info(f"  Weighted F1: {weighted_f1:.4f}")
    logger.info(f"  Mean AUC:    {mean_auc:.4f}")
    logger.info(f"\nPer-class F1:")
    for name, f1 in results["per_class_f1"].items():
        logger.info(f"  {name:<30} {f1:.4f}")
    logger.info(f"{'='*60}\n")

    return results


# ── Learning curves ────────────────────────────────────────────────────────────

def learning_curve_experiment(
    train_paths, train_labels, val_paths, val_labels,
    model_fn, model_name: str,
    criterion, epochs: int = 15, batch_size: int = 32,
    sizes=(0.10, 0.25, 0.50, 0.75, 1.00),
) -> dict:
    """
    Train model on increasing fractions of the training set.
    Records train F1 and val F1 at each size fraction.
    Uses stratified sampling to maintain class balance at each size.
    """
    results = {"sizes": [], "train_f1": [], "val_f1": []}
    val_ds = ISICDataset(val_paths, val_labels, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    labels_arr = np.array(train_labels)

    for frac in sizes:
        n_total = len(train_paths)
        n_use = max(int(n_total * frac), NUM_CLASSES)  # at least 1 per class

        # Stratified subsample
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_use, random_state=42)
        try:
            sub_idx, _ = next(sss.split(train_paths, labels_arr))
        except Exception:
            sub_idx = np.random.choice(n_total, n_use, replace=False)

        sub_paths  = [train_paths[i]  for i in sub_idx]
        sub_labels = [train_labels[i] for i in sub_idx]

        train_ds = ISICDataset(sub_paths, sub_labels, transform=train_transform)
        sampler  = make_weighted_sampler(sub_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)

        # Fresh model instance
        model = model_fn().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_f1, best_train_f1 = 0.0, 0.0
        for ep in range(1, epochs + 1):
            tr_loss, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, ep)
            _, val_f1, _, _ = validate(model, val_loader, criterion)
            scheduler.step()
            if val_f1 > best_val_f1:
                best_val_f1   = val_f1
                best_train_f1 = tr_f1

        pct = int(frac * 100)
        results["sizes"].append(pct)
        results["train_f1"].append(round(best_train_f1, 4))
        results["val_f1"].append(round(best_val_f1, 4))
        logger.info(f"  LC [{model_name}] {pct}%: train_F1={best_train_f1:.3f}, val_F1={best_val_f1:.3f}")

    return results


# ── Traditional ML training ───────────────────────────────────────────────────

def train_traditional_ml(
    train_paths: List[str], train_labels: List[int],
    val_paths: List[str], val_labels: List[int],
    save_dir: str = "./models",
) -> dict:
    """
    Train SVM (RBF) and Random Forest on HOG + LBP features.
    Both use class_weight='balanced' to handle imbalance.
    Saves svm.pkl and rf.pkl to save_dir.
    Returns evaluation metrics.
    """
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from skimage.feature import hog, local_binary_pattern
    from skimage.color import rgb2gray

    def extract_features(paths: List[str]) -> np.ndarray:
        feats = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                arr = np.array(img.resize((128, 128)))
                gray = rgb2gray(arr)
                hog_feat = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                               cells_per_block=(1, 1), feature_vector=True)
                lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
                feats.append(np.concatenate([hog_feat, lbp_hist]))
            except Exception as e:
                logger.warning(f"Feature extraction failed for {path}: {e}")
                feats.append(np.zeros(hog_feat.shape[0] + 10))
        return np.array(feats)

    logger.info("Extracting HOG+LBP features for training set…")
    t0 = time.time()
    X_train = extract_features(train_paths)
    y_train = np.array(train_labels)
    logger.info(f"  Train features: {X_train.shape} ({time.time()-t0:.1f}s)")

    logger.info("Extracting HOG+LBP features for validation set…")
    X_val = extract_features(val_paths)
    y_val = np.array(val_labels)

    # SVM — RBF kernel, class_weight balanced
    logger.info("Training SVM (RBF, class_weight=balanced)…")
    svm = SVC(
        kernel="rbf", C=10.0, gamma="scale",
        class_weight="balanced", probability=True, random_state=42,
    )
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_val)
    svm_f1   = f1_score(y_val, svm_pred, average="macro", zero_division=0)
    logger.info(f"  SVM val macro-F1: {svm_f1:.4f}")

    # Random Forest — class_weight balanced_subsample
    logger.info("Training Random Forest (500 trees, class_weight=balanced_subsample)…")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_split=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_f1   = f1_score(y_val, rf_pred, average="macro", zero_division=0)
    logger.info(f"  RF val macro-F1:  {rf_f1:.4f}")

    # Stacking Ensemble: SVM + RF → Logistic Regression meta-learner
    logger.info("Training Stacking Ensemble (SVM + RF → LogisticRegression)…")
    stacking = StackingClassifier(
        estimators=[("svm", svm), ("rf", rf)],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        cv=3, passthrough=False,
    )
    stacking.fit(X_train, y_train)
    stack_pred = stacking.predict(X_val)
    stack_f1   = f1_score(y_val, stack_pred, average="macro", zero_division=0)
    logger.info(f"  Stacking val macro-F1: {stack_f1:.4f}")

    # Save models
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "svm.pkl"),      "wb") as f: pickle.dump(svm, f)
    with open(os.path.join(save_dir, "rf.pkl"),       "wb") as f: pickle.dump(rf, f)
    with open(os.path.join(save_dir, "stacking.pkl"), "wb") as f: pickle.dump(stacking, f)
    logger.info(f"✓ Traditional ML models saved to {save_dir}/")

    return {
        "svm_macro_f1":      round(svm_f1, 4),
        "rf_macro_f1":       round(rf_f1, 4),
        "stacking_macro_f1": round(stack_f1, 4),
        "classification_report_svm": classification_report(
            y_val, svm_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "classification_report_rf": classification_report(
            y_val, rf_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train_deep_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int = 30,
    save_dir: str = "./models",
) -> dict:
    """
    Full training loop with:
      - AdamW optimiser (lr=3e-5, weight_decay=1e-4)
      - CosineAnnealingLR scheduler
      - Gradient clipping (max_norm=1.0)
      - Early stopping (patience=7, monitors val macro-F1)
      - Saves best checkpoint on val F1 improvement
    """
    ckpt_path = os.path.join(save_dir, f"{model_name}.pth")
    early_stop = EarlyStopping(patience=7, checkpoint_path=ckpt_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )

    history = {"epoch": [], "train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}

    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name} for up to {epochs} epochs…")
    logger.info(f"Device: {DEVICE} | Checkpoint: {ckpt_path}")
    logger.info(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_f1, _, _ = validate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_f1"].append(round(tr_f1, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_f1"].append(round(val_f1, 4))

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train — loss={tr_loss:.4f} F1={tr_f1:.4f} | "
            f"Val — loss={val_loss:.4f} F1={val_f1:.4f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        if early_stop(val_f1, model):
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    # Reload best checkpoint
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"✓ Loaded best checkpoint (val F1={state['val_f1']:.4f})")

    return history


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DermAI Guard Training Pipeline")
    parser.add_argument("--data_dir",   type=str, default="./data",   help="Path to data directory")
    parser.add_argument("--model_dir",  type=str, default="./models", help="Path to save models")
    parser.add_argument("--epochs",     type=int, default=30,          help="Max epochs per model")
    parser.add_argument("--batch_size", type=int, default=32,          help="Batch size")
    parser.add_argument("--model",      type=str, default="both",
                        choices=["efficientnet_b4", "vit", "both", "traditional", "all"],
                        help="Which model(s) to train")
    parser.add_argument("--learning_curves", action="store_true",
                        help="Run learning curve experiments (slow)")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("Loading image paths…")
    all_paths, all_labels = load_image_paths(args.data_dir)
    if len(all_paths) == 0:
        logger.error(
            f"No images found in {args.data_dir}. "
            "Run dataset download first: POST /dataset/download from the frontend Evaluation tab."
        )
        return

    logger.info(f"Total images: {len(all_paths)}")

    # Stratified 70/15/15 split
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        stratified_split(all_paths, all_labels)

    # Class-weighted loss (CRITICAL for imbalanced ISIC data)
    class_weights = compute_loss_weights(train_labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Data loaders
    train_ds = ISICDataset(train_paths, train_labels, transform=train_transform)
    val_ds   = ISICDataset(val_paths,   val_labels,   transform=val_transform)
    test_ds  = ISICDataset(test_paths,  test_labels,  transform=val_transform)

    sampler = make_weighted_sampler(train_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,   num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,   num_workers=0)

    all_results = {}

    # ── Traditional ML ──
    if args.model in ("traditional", "all"):
        logger.info("\n=== TRADITIONAL ML TRAINING ===")
        ml_results = train_traditional_ml(
            train_paths, train_labels, val_paths, val_labels, args.model_dir
        )
        all_results["traditional_ml"] = ml_results

    # ── EfficientNet-B4 ──
    if args.model in ("efficientnet_b4", "both", "all"):
        logger.info("\n=== EFFICIENTNET-B4 TRAINING ===")
        eff_model = build_efficientnet_b4()
        history = train_deep_model(
            eff_model, "efficientnet_b4", train_loader, val_loader,
            criterion, args.epochs, args.model_dir
        )
        logger.info("\nEfficientNet-B4 TEST EVALUATION:")
        eff_results = full_evaluation(eff_model, test_loader, "test_efficientnet_b4")
        eff_results["training_history"] = history
        all_results["efficientnet_b4"] = eff_results

        if args.learning_curves:
            logger.info("Running EfficientNet-B4 learning curves…")
            lc = learning_curve_experiment(
                train_paths, train_labels, val_paths, val_labels,
                build_efficientnet_b4, "EfficientNet-B4",
                criterion, epochs=15, batch_size=args.batch_size,
            )
            all_results["efficientnet_b4"]["learning_curve"] = lc

    # ── ViT-B/16 ──
    if args.model in ("vit", "both", "all"):
        logger.info("\n=== ViT-B/16 TRAINING ===")
        vit_model = build_vit()
        history = train_deep_model(
            vit_model, "vit_b16", train_loader, val_loader,
            criterion, args.epochs, args.model_dir
        )
        logger.info("\nViT-B/16 TEST EVALUATION:")
        vit_results = full_evaluation(vit_model, test_loader, "test_vit_b16")
        vit_results["training_history"] = history
        all_results["vit_b16"] = vit_results

    # ── Save results JSON ──
    results_path = os.path.join(args.model_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✓ All results saved to {results_path}")

    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    for model_name, res in all_results.items():
        if isinstance(res, dict) and "macro_f1" in res:
            logger.info(f"  {model_name}: macro_F1={res['macro_f1']:.4f}, AUC={res.get('mean_auc', 0):.4f}")
        elif isinstance(res, dict):
            for k, v in res.items():
                if "f1" in k.lower():
                    logger.info(f"  {model_name} / {k}: {v:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
