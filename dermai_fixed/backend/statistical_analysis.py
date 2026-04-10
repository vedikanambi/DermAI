"""
statistical_analysis.py  —  DermAI Guard Dataset Statistical Analysis
=======================================================================
Run: python statistical_analysis.py --data_dir ./data

Produces:
  - Class distribution plots (train/val/test)
  - Imbalance ratio analysis
  - Pixel-level statistics (RGB means, std)
  - Image dimension statistics
  - Saved figures to analysis_output/

This satisfies the requirement: "Apply statistical analysis to understand
data characteristics" (Section 2, project brief).
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
from collections import Counter

try:
    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Skipping plots.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

CLASS_NAMES = [
    "Melanoma", "Melanocytic Nevi", "Basal Cell Carcinoma",
    "Actinic Keratosis", "Benign Keratosis", "Dermatofibroma",
    "Vascular Lesion", "Squamous Cell Carcinoma", "Unknown",
]
CLASS_ABBR = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
NUM_CLASSES = len(CLASS_NAMES)


def load_paths_labels(data_dir: str, split: str = "train") -> Tuple[List[str], List[int]]:
    paths, labels = [], []
    split_path = Path(data_dir) / split
    if not split_path.exists():
        split_path = Path(data_dir)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        folder = split_path / cls_name.replace(" ", "_")
        if not folder.exists():
            continue
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        paths.extend([str(p) for p in imgs])
        labels.extend([cls_idx] * len(imgs))
    return paths, labels


def class_distribution_analysis(splits_data: dict, output_dir: str) -> dict:
    """Compute and plot class distributions across all splits."""
    results = {}

    for split, (paths, labels) in splits_data.items():
        counts = Counter(labels)
        dist = {CLASS_NAMES[i]: counts.get(i, 0) for i in range(NUM_CLASSES)}
        total = sum(dist.values())
        results[split] = {
            "counts":      dist,
            "percentages": {k: round(100 * v / max(total, 1), 2) for k, v in dist.items()},
            "total":       total,
        }
        print(f"\n{split.upper()} split ({total} images):")
        for cls, cnt in dist.items():
            pct = 100 * cnt / max(total, 1)
            print(f"  {cls:<30} {cnt:>5}  ({pct:.1f}%)")

    # Imbalance ratio
    if "train" in results:
        counts = list(results["train"]["counts"].values())
        non_zero = [c for c in counts if c > 0]
        if non_zero:
            ratio = max(non_zero) / min(non_zero)
            max_cls = CLASS_NAMES[np.argmax(counts)]
            min_cls = CLASS_NAMES[np.argmin(counts)]
            results["imbalance_ratio"] = round(ratio, 2)
            results["majority_class"]  = max_cls
            results["minority_class"]  = min_cls
            print(f"\nClass imbalance ratio: {ratio:.1f}x ({max_cls} vs {min_cls})")

    # Plot
    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, len(splits_data), figsize=(6 * len(splits_data), 6))
        if len(splits_data) == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

        for ax, (split, data) in zip(axes, results.items()):
            if split in ("imbalance_ratio", "majority_class", "minority_class"):
                continue
            counts = [data["counts"].get(n, 0) for n in CLASS_NAMES]
            bars = ax.bar(CLASS_ABBR, counts, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_title(f"{split.capitalize()} Split (n={data['total']:,})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Class")
            ax.set_ylabel("Number of Images")
            ax.tick_params(axis="x", rotation=45)
            for bar, cnt in zip(bars, counts):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                            str(cnt), ha="center", va="bottom", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("DermAI Guard — Class Distribution Analysis (ISIC 2019)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "class_distribution.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n✓ Saved: {out_path}")

    return results


def pixel_statistics_analysis(
    train_paths: List[str],
    train_labels: List[int],
    output_dir: str,
    sample_size: int = 500,
) -> dict:
    """Per-class pixel statistics (mean, std, R/G/B channels)."""
    if not PIL_AVAILABLE:
        return {}

    sample_indices = np.random.choice(len(train_paths), min(sample_size, len(train_paths)), replace=False)
    sample_paths   = [train_paths[i]  for i in sample_indices]
    sample_labels  = [train_labels[i] for i in sample_indices]

    # Per-class stats
    class_stats = {i: {"R": [], "G": [], "B": [], "contrast": []} for i in range(NUM_CLASSES)}

    for path, label in zip(sample_paths, sample_labels):
        try:
            img = Image.open(path).convert("RGB").resize((128, 128))
            arr = np.array(img).astype(np.float32) / 255.0
            class_stats[label]["R"].append(arr[:, :, 0].mean())
            class_stats[label]["G"].append(arr[:, :, 1].mean())
            class_stats[label]["B"].append(arr[:, :, 2].mean())
            # Contrast: std of grayscale
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            class_stats[label]["contrast"].append(gray.std())
        except Exception:
            pass

    results = {}
    print("\nPer-class pixel statistics (sample):")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        s = class_stats[cls_idx]
        if not s["R"]:
            continue
        r, g, b, cont = (np.array(s[k]) for k in ("R", "G", "B", "contrast"))
        results[cls_name] = {
            "mean_R": round(float(r.mean()), 4),
            "mean_G": round(float(g.mean()), 4),
            "mean_B": round(float(b.mean()), 4),
            "std_R":  round(float(r.std()),  4),
            "std_G":  round(float(g.std()),  4),
            "std_B":  round(float(b.std()),  4),
            "mean_contrast": round(float(cont.mean()), 4),
            "n_sampled": len(r),
        }
        print(f"  {cls_name:<30} R={r.mean():.3f} G={g.mean():.3f} B={b.mean():.3f} contrast={cont.mean():.3f}")

    # Plot RGB means per class
    if MATPLOTLIB_AVAILABLE and results:
        classes = list(results.keys())
        means_r = [results[c]["mean_R"] for c in classes]
        means_g = [results[c]["mean_G"] for c in classes]
        means_b = [results[c]["mean_B"] for c in classes]
        contrasts = [results[c]["mean_contrast"] for c in classes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x     = np.arange(len(classes))
        width = 0.27
        abbrs = [CLASS_ABBR[CLASS_NAMES.index(c)] if c in CLASS_NAMES else c for c in classes]

        ax1.bar(x - width, means_r, width, label="Red",   color="tomato",     alpha=0.85)
        ax1.bar(x,         means_g, width, label="Green", color="mediumseagreen", alpha=0.85)
        ax1.bar(x + width, means_b, width, label="Blue",  color="steelblue",  alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(abbrs, rotation=45)
        ax1.set_ylabel("Mean Pixel Value (0–1)")
        ax1.set_title("Mean RGB Intensity per Class")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(x, contrasts, color="mediumpurple", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(abbrs, rotation=45)
        ax2.set_ylabel("Mean Grayscale Std Dev")
        ax2.set_title("Image Contrast per Class")
        ax2.grid(axis="y", alpha=0.3)

        plt.suptitle("DermAI Guard — Pixel Statistics by Class (ISIC 2019 Sample)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "pixel_statistics.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {out_path}")

    return results


def image_size_analysis(train_paths: List[str], output_dir: str, sample_size: int = 300) -> dict:
    """Analyse image dimensions in the dataset."""
    if not PIL_AVAILABLE:
        return {}

    sample = train_paths[:sample_size]
    widths, heights = [], []
    for path in sample:
        try:
            img = Image.open(path)
            widths.append(img.width)
            heights.append(img.height)
        except Exception:
            pass

    if not widths:
        return {}

    results = {
        "mean_width":   round(float(np.mean(widths)),  1),
        "mean_height":  round(float(np.mean(heights)), 1),
        "std_width":    round(float(np.std(widths)),   1),
        "std_height":   round(float(np.std(heights)),  1),
        "min_width":    int(min(widths)),
        "min_height":   int(min(heights)),
        "max_width":    int(max(widths)),
        "max_height":   int(max(heights)),
        "n_sampled":    len(widths),
    }

    print(f"\nImage size statistics (sample of {len(widths)}):")
    print(f"  Width:  mean={results['mean_width']:.0f} ± {results['std_width']:.0f}  "
          f"[{results['min_width']}, {results['max_width']}]")
    print(f"  Height: mean={results['mean_height']:.0f} ± {results['std_height']:.0f}  "
          f"[{results['min_height']}, {results['max_height']}]")

    if MATPLOTLIB_AVAILABLE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(widths, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax1.set_xlabel("Width (pixels)")
        ax1.set_ylabel("Count")
        ax1.set_title("Image Width Distribution")
        ax1.axvline(np.mean(widths), color="red", linestyle="--", label=f"Mean={np.mean(widths):.0f}")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.hist(heights, bins=30, color="mediumpurple", edgecolor="white", alpha=0.8)
        ax2.set_xlabel("Height (pixels)")
        ax2.set_ylabel("Count")
        ax2.set_title("Image Height Distribution")
        ax2.axvline(np.mean(heights), color="red", linestyle="--", label=f"Mean={np.mean(heights):.0f}")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.suptitle("DermAI Guard — Image Dimension Analysis (ISIC 2019 Sample)", fontsize=13)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "image_dimensions.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {out_path}")

    return results


def class_imbalance_impact(train_labels: List[int], output_dir: str) -> dict:
    """Illustrate class imbalance and its effect on naive accuracy."""
    counts  = np.bincount(np.array(train_labels), minlength=NUM_CLASSES)
    total   = counts.sum()
    probs   = counts / max(total, 1)

    # Naive accuracy = accuracy if you always predict majority class
    majority_cls    = int(counts.argmax())
    naive_accuracy  = float(probs[majority_cls])

    # Gini impurity as imbalance measure
    gini = 1.0 - float(np.sum(probs ** 2))

    results = {
        "majority_class":   CLASS_NAMES[majority_cls],
        "majority_pct":     round(100 * naive_accuracy, 2),
        "naive_accuracy":   round(naive_accuracy, 4),
        "gini_impurity":    round(gini, 4),
        "effective_n_classes": round(1.0 / max(np.sum(probs ** 2), 1e-9), 2),
    }

    print(f"\nClass imbalance analysis:")
    print(f"  Majority class: {results['majority_class']} ({results['majority_pct']}% of training data)")
    print(f"  Naive accuracy (always predict majority): {naive_accuracy:.4f}")
    print(f"  Gini impurity: {gini:.4f} (0=pure, ~0.89=balanced 9-class)")
    print(f"  Effective number of classes: {results['effective_n_classes']:.2f}")
    print(f"  → This explains why accuracy is a misleading metric;")
    print(f"    a trivial classifier can achieve {naive_accuracy:.0%} accuracy.")
    print(f"    We use Macro F1 (treats all classes equally) as primary metric.")

    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, NUM_CLASSES))
        bars = ax.bar(CLASS_ABBR, counts, color=colors, edgecolor="black", linewidth=0.5)

        ax.axhline(counts.mean(), color="navy", linestyle="--", linewidth=2,
                   label=f"Mean per class = {counts.mean():.0f}")
        ax.set_ylabel("Number of Training Images")
        ax.set_xlabel("Class")
        ax.set_title("ISIC 2019 Training Set — Class Imbalance\n"
                     f"Imbalance ratio: {counts.max()//max(counts.min(),1)}:1  |  "
                     f"Naive accuracy: {naive_accuracy:.1%}  |  "
                     f"Primary metric: Macro F1", fontsize=11)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{cnt:,}", ha="center", va="bottom", fontsize=9)

        # Add class names as x-tick labels with rotation
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_ABBR, rotation=45)

        plt.tight_layout()
        out_path = os.path.join(output_dir, "class_imbalance.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="DermAI Guard Statistical Analysis")
    parser.add_argument("--data_dir",    type=str, default="./data")
    parser.add_argument("--output_dir",  type=str, default="./analysis_output")
    parser.add_argument("--sample_size", type=int, default=500,
                        help="Number of images to sample for pixel/size analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("DermAI Guard — Statistical Data Analysis")
    print("=" * 60)

    # Load data splits
    splits_data = {}
    for split in ("train", "val", "test"):
        paths, labels = load_paths_labels(args.data_dir, split)
        if paths:
            splits_data[split] = (paths, labels)
            print(f"Loaded {split}: {len(paths)} images")

    if not splits_data:
        print(f"\nNo data found in {args.data_dir}. Run dataset download first.")
        return

    all_results = {}

    # Class distribution
    print("\n=== CLASS DISTRIBUTION ===")
    dist_results = class_distribution_analysis(splits_data, args.output_dir)
    all_results["class_distribution"] = dist_results

    # Pixel statistics (train only for speed)
    if "train" in splits_data:
        train_paths, train_labels = splits_data["train"]
        print("\n=== PIXEL STATISTICS ===")
        pixel_results = pixel_statistics_analysis(
            train_paths, train_labels, args.output_dir, args.sample_size
        )
        all_results["pixel_statistics"] = pixel_results

        # Image size analysis
        print("\n=== IMAGE DIMENSIONS ===")
        size_results = image_size_analysis(train_paths, args.output_dir, args.sample_size)
        all_results["image_size"] = size_results

        # Class imbalance impact
        print("\n=== CLASS IMBALANCE IMPACT ===")
        imbalance_results = class_imbalance_impact(train_labels, args.output_dir)
        all_results["imbalance_analysis"] = imbalance_results

    # Save JSON summary
    out_json = os.path.join(args.output_dir, "statistical_analysis.json")
    import json
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Full results saved to {out_json}")
    print(f"✓ Figures saved to {args.output_dir}/")
    print("\nKey findings for report:")
    if "class_distribution" in all_results and "train" in all_results["class_distribution"]:
        d  = all_results["class_distribution"]
        ir = d.get("imbalance_ratio", "N/A")
        print(f"  - Imbalance ratio: {ir}:1")
        print(f"  - Majority class: {d.get('majority_class', 'N/A')}")
        print(f"  - Minority class: {d.get('minority_class', 'N/A')}")
    if "imbalance_analysis" in all_results:
        ia = all_results["imbalance_analysis"]
        print(f"  - Naive accuracy if always predicting majority: {ia.get('naive_accuracy', 'N/A'):.2%}")
        print(f"  - Gini impurity: {ia.get('gini_impurity', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
