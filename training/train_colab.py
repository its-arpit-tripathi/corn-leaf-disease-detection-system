#!/usr/bin/env python3
"""Google Colab / Standalone Training Script for Maize (Corn) Leaf Disease Classifier.

Features:
  1. Distinguishes Maize vs Non-Maize (Not_Leaf / Not_Corn class) + 4 target conditions:
     - Blight (Northern Leaf Blight)
     - Common_Rust (Common Rust)
     - Gray_Leaf_Spot (Gray Leaf Spot)
     - Healthy (Healthy Maize)
     - Not_Leaf (Non-Maize Leaf, other crops, background, or invalid images)
  2. Auto-maps PlantVillage, Kaggle, and arbitrary folder names to canonical classes.
     Optionally treats any other plant species (Tomato, Potato, Apple, etc.) or background folders as Not_Leaf.
  3. Preprocessing: EXIF orientation correction, deduplication (MD5 exact + dHash visual),
     and leakage prevention before stratified 70/15/15 split.
  4. Two-stage transfer learning with MobileNetV2 (ImageNet backbone, fine-tuning top 40 layers).
  5. Built-in feature-space Out-of-Distribution (OOD) gate calculation (ood_stats.npz).
  6. Exports:
     - corn_leaf_disease_model_v3.keras
     - class_names.json
     - ood_stats.npz
     - metrics.json
     - confusion_matrix.png
     - model_export.zip (ready for one-click download and drop into backend/model/)

Usage in Colab or terminal:
  python train_colab.py --data /path/to/dataset --out ./output
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

IMG = 224
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
CACHE_MAX_SIDE = 512
NEG_CLASS = "Not_Leaf"


# ---------------------------------------------------------------- class mapping
def canon(folder: str, overrides: dict[str, str], map_unrecognized_to_not_corn: bool = False) -> str | None:
    """Map a dataset folder name to one of our 5 classes."""
    if folder in overrides:
        return overrides[folder]

    k = re.sub(r"[^a-z0-9]+", "_", folder.lower()).strip("_")
    tokens = set(k.split("_"))

    # Explicit negative / non-corn tokens
    if tokens & {"not", "non", "other", "others", "negative", "background", "not_corn", "not_leaf", "non_corn"}:
        return NEG_CLASS

    # Non-corn plant species from PlantVillage / Kaggle
    non_corn_plants = {
        "apple", "blueberry", "cherry", "grape", "orange", "peach",
        "pepper", "potato", "raspberry", "soybean", "squash", "strawberry", "tomato"
    }
    if tokens & non_corn_plants:
        return NEG_CLASS

    # Corn disease conditions
    if "rust" in k:
        return "Common_Rust"
    if "blight" in k:
        return "Blight"
    if tokens & {"gray", "grey", "cercospora", "spot"}:
        return "Gray_Leaf_Spot"
    if "healthy" in k and ("corn" in k or "maize" in k or len(tokens) == 1):
        return "Healthy"

    if map_unrecognized_to_not_corn:
        return NEG_CLASS

    return None


def find_images(roots: list[str], overrides: dict[str, str], skip: set[str], auto_negative: bool = False):
    items, unknown = [], Counter()
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"Warning: path does not exist: {root}")
            continue

        for d in sorted(root_path.rglob("*")):
            if not d.is_dir() or d.name in skip:
                continue
            files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in EXTS]
            if not files:
                continue
            cls = canon(d.name, overrides, map_unrecognized_to_not_corn=auto_negative)
            if cls is None:
                unknown[str(d)] += len(files)
                continue
            items += [(f, cls) for f in files]

    if unknown and not auto_negative:
        lines = "\n".join(f"  {p}  ({n} images)" for p, n in list(unknown.items())[:15])
        print("Note: Some folders could not be automatically mapped to corn classes:")
        print(lines)
        print("If these are non-corn images, run with --auto-negative to classify them as Not_Leaf.")

    return items


# ---------------------------------------------------------------- deduplication
def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def process_image(args):
    src, cls, cache = args
    try:
        raw = Path(src).read_bytes()
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
        if min(img.size) < 32:
            return None
        md5 = hashlib.md5(raw).hexdigest()
        h = dhash(img)
        img.thumbnail((CACHE_MAX_SIDE, CACHE_MAX_SIDE), Image.Resampling.BILINEAR)
        out = Path(cache) / cls / f"{md5}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            img.save(out, "JPEG", quality=95)
        return dict(path=str(out), label=cls, md5=md5, dhash=h, src=str(src))
    except Exception:
        return None


def clean(items, cache, workers):
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(process_image, [(s, c, cache) for s, c in items]))
    ok = [r for r in rows if r]
    report = {"found": len(items), "unreadable_or_tiny": len(items) - len(ok)}

    def dedupe(rows_in, key):
        groups = defaultdict(list)
        for r in rows_in:
            groups[r[key]].append(r)
        kept, dupes, conflicts = [], 0, 0
        for g in groups.values():
            if len({r["label"] for r in g}) > 1:
                conflicts += len(g)
                continue
            kept.append(g[0])
            dupes += len(g) - 1
        return kept, dupes, conflicts

    uniq, exact, c1 = dedupe(ok, "md5")
    final, visual, c2 = dedupe(uniq, "dhash")
    report["exact_duplicates_removed"] = exact
    report["visual_duplicates_removed"] = visual
    report["label_conflicts_removed"] = c1 + c2
    report["kept"] = len(final)
    return final, report


# ---------------------------------------------------------------- tf.data pipeline
def make_ds(paths, y, batch, training, seed):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(p, label):
        img = tf.io.decode_jpeg(tf.io.read_file(p))
        img = tf.image.resize(img, (IMG, IMG))
        img = tf.cast(img, tf.float32)  # RAW 0-255 pixels
        img.set_shape((IMG, IMG, 3))
        return img, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch)
    if training:
        aug = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2, value_range=(0, 255)),
        ])
        ds = ds.map(lambda x, l: (aug(x, training=True), l), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- model architecture
def build_model(n_classes: int, pretrained: bool = True, dropout: float = 0.25):
    """Builds MobileNetV2 architecture strictly matching the backend deployment contract."""
    inp = keras.Input((IMG, IMG, 3), name="input_image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescaling")(inp)
    base = keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    base._name = "mobilenetv2_1.00_224"
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", name="dense")(x)
    return keras.Model(inp, out), base


def fit_model(model, train, val, lr, epochs, cw, patience):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=max(1, patience // 2), min_lr=1e-7),
    ]
    h = model.fit(train, validation_data=val, epochs=epochs, class_weight=cw, callbacks=cbs, verbose=1)
    return {k: [float(v) for v in vs] for k, vs in h.history.items()}


# ---------------------------------------------------------------- OOD Gate
def fit_ood_gate(model, train_paths, train_labels, val_paths, val_labels, out_path, percentile=99.0):
    """Compute Mahalanobis distance parameters on corn-leaf features."""
    corn_classes = sorted({c for c in train_labels if c != NEG_CLASS})
    if not corn_classes:
        print("Warning: no corn classes found for OOD fitting.")
        return {}

    ext = keras.Model(model.input, model.get_layer("global_average_pooling2d").output)

    def extract_features(paths, labels, only_corn=True):
        feats, kept_lbls = [], []
        for i in range(0, len(paths), 64):
            batch_paths = paths[i:i + 64]
            batch_lbls = labels[i:i + 64]
            arr = []
            for p, l in zip(batch_paths, batch_lbls):
                if only_corn and l == NEG_CLASS:
                    continue
                try:
                    im = Image.open(p).convert("RGB").resize((IMG, IMG), Image.Resampling.BILINEAR)
                    arr.append(np.asarray(im, dtype="float32"))
                    kept_lbls.append(l)
                except Exception:
                    pass
            if arr:
                batch_arr = np.stack(arr)
                feats.append(ext.predict(batch_arr, verbose=0))
        return (np.concatenate(feats).astype(np.float64) if feats else np.zeros((0, 1280))), kept_lbls

    f_tr, y_tr = extract_features(train_paths, train_labels, only_corn=True)
    f_va, _ = extract_features(val_paths, val_labels, only_corn=True)

    if len(f_tr) < 30 or len(f_va) < 10:
        print("Warning: Insufficient corn images for robust OOD gate; skipping ood_stats.npz")
        return {}

    classes = sorted(set(y_tr))
    means = np.stack([f_tr[np.array(y_tr) == c].mean(0) for c in classes])
    centred = f_tr - means[[classes.index(c) for c in y_tr]]
    cov = centred.T @ centred / max(len(f_tr) - 1, 1)
    d = cov.shape[0]
    shrinkage = 0.1
    cov = (1 - shrinkage) * cov + shrinkage * (np.trace(cov) / d) * np.eye(d)
    precision = np.linalg.inv(cov)

    dists = [np.einsum("nd,nd->n", (f_va - mu) @ precision, f_va - mu) for mu in means]
    val_dists = np.min(np.stack(dists, axis=0), axis=0) if dists else np.zeros((0,), dtype=np.float64)
    threshold = float(np.percentile(val_dists, percentile))

    np.savez(
        out_path,
        means=means.astype("float32"),
        precision=precision.astype("float32"),
        threshold=np.float64(threshold),
        classes=np.array(classes),
    )
    print(f"OOD stats saved to {out_path} (Threshold: {threshold:.2f} at {percentile}th percentile)")
    return {"classes": classes, "threshold": threshold, "percentile": percentile}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="Path(s) to dataset root directory")
    ap.add_argument("--negatives", nargs="*", default=[], help="Optional directory with non-corn photos for Not_Leaf class")
    ap.add_argument("--out", default="output", help="Output directory for trained model and artifacts")
    ap.add_argument("--auto-negative", action="store_true", help="Automatically assign other plant species/background folders to Not_Leaf")
    ap.add_argument("--max-per-class", type=int, default=1500, help="Cap images per class to prevent imbalance (e.g. 1500)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs1", type=int, default=10, help="Stage 1 epochs (head only)")
    ap.add_argument("--epochs2", type=int, default=15, help="Stage 2 epochs (fine-tuning)")
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-5)
    ap.add_argument("--unfreeze", type=int, default=40, help="Number of backbone layers to unfreeze in stage 2")
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    tf.random.set_seed(a.seed)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    overrides = {}
    items = find_images(a.data, overrides, skip=set(), auto_negative=a.auto_negative)

    if a.negatives:
        for neg_dir in a.negatives:
            p = Path(neg_dir)
            if p.exists():
                files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in EXTS]
                items += [(f, NEG_CLASS) for f in files]

    if not items:
        sys.exit(f"No valid images found in {a.data}. Check your dataset directory path.")

    print(f"Found {len(items)} raw image files. Starting preprocessing and deduplication...")
    rows, report = clean(items, out / "clean_cache", a.workers)

    if a.max_per_class:
        rng = random.Random(a.seed)
        rng.shuffle(rows)
        kept, counter = [], Counter()
        for r in rows:
            if counter[r["label"]] < a.max_per_class:
                counter[r["label"]] += 1
                kept.append(r)
        rows = kept

    counts = Counter(r["label"] for r in rows)
    class_names = sorted(counts.keys())
    print("\nDataset Cleaning Summary:", json.dumps(report, indent=2))
    print("Class Counts:", dict(counts))

    if len(class_names) < 2:
        sys.exit("Error: Need at least 2 classes to train. Check dataset folder structure.")

    if NEG_CLASS not in class_names:
        print("\n" + "=" * 70)
        print("WARNING: Dataset does NOT include a Not_Leaf / Not_Corn class.")
        print("To distinguish maize vs non-maize, supply a folder of other plants/background")
        print("or pass --auto-negative if your dataset contains other plant species.")
        print("=" * 70 + "\n")

    idx = {c: i for i, c in enumerate(class_names)}
    paths = np.array([r["path"] for r in rows])
    y = np.array([idx[r["label"]] for r in rows])

    # Stratified split: 70% Train, 15% Validation, 15% Test
    tr_p, tmp_p, tr_y, tmp_y = train_test_split(paths, y, test_size=0.30, stratify=y, random_state=a.seed)
    va_p, te_p, va_y, te_y = train_test_split(tmp_p, tmp_y, test_size=0.50, stratify=tmp_y, random_state=a.seed)

    print(f"\nSplits: Train={len(tr_p)}, Val={len(va_p)}, Test={len(te_p)}")

    train_ds = make_ds(tr_p, tr_y, a.batch, training=True, seed=a.seed)
    val_ds = make_ds(va_p, va_y, a.batch, training=False, seed=a.seed)
    test_ds = make_ds(te_p, te_y, a.batch, training=False, seed=a.seed)

    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(class_names)), y=tr_y)))

    # Build model
    print(f"\nBuilding MobileNetV2 for {len(class_names)} classes: {class_names}")
    model, base = build_model(len(class_names), pretrained=True, dropout=a.dropout)

    # Stage 1: Train Head Only
    base.trainable = False
    print("\n=== Stage 1: Training Classification Head (Backbone Frozen) ===")
    h1 = fit_model(model, train_ds, val_ds, lr=a.lr1, epochs=a.epochs1, cw=cw, patience=4)

    # Stage 2: Fine-Tuning Top Layers
    base.trainable = True
    for layer in base.layers[:-a.unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    print(f"\n=== Stage 2: Fine-Tuning Top {a.unfreeze} Backbone Layers (lr={a.lr2}) ===")
    h2 = fit_model(model, train_ds, val_ds, lr=a.lr2, epochs=a.epochs2, cw=cw, patience=5)

    # Test Evaluation
    print("\n=== Final Evaluation on Held-Out Test Set ===")
    prob = model.predict(test_ds, verbose=0)
    pred = prob.argmax(axis=1)

    rep = classification_report(te_y, pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(te_y, pred)

    acc = float((pred == te_y).mean())
    macro_f1 = float(f1_score(te_y, pred, average="macro"))

    print(f"\n>>> TEST ACCURACY: {acc * 100:.2f}% | MACRO-F1: {macro_f1:.4f} <<<")
    print(classification_report(te_y, pred, target_names=class_names, zero_division=0))

    # Save artifacts
    model_path = out / "corn_leaf_disease_model_v3.keras"
    model.save(model_path)
    (out / "class_names.json").write_text(json.dumps(class_names, indent=4))

    # Fit OOD statistics for backend
    names = lambda yy: [class_names[i] for i in yy]
    ood_info = fit_ood_gate(
        model,
        tr_p,
        names(tr_y),
        va_p,
        names(va_y),
        out / "ood_stats.npz",
    )

    # Metrics
    metrics = {
        "classes": class_names,
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "per_class": {c: rep[c] for c in class_names},
        "confusion_matrix": cm.tolist(),
        "class_counts": dict(counts),
        "split_sizes": {"train": len(tr_p), "val": len(va_p), "test": len(te_p)},
        "ood_info": ood_info,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Plot confusion matrix
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Corn Leaf Disease Confusion Matrix")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=160)
        plt.close(fig)
    except Exception as e:
        print("Confusion matrix plotting skipped:", e)

    # Package files for one-click download
    zip_path = out / "model_export.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ["corn_leaf_disease_model_v3.keras", "class_names.json", "ood_stats.npz", "metrics.json", "confusion_matrix.png"]:
            f = out / fname
            if f.exists():
                zf.write(f, arcname=fname)

    print("\n" + "=" * 70)
    print("SUCCESS! Training and export complete.")
    print(f"Generated artifacts in '{out}/':")
    print("  1. corn_leaf_disease_model_v3.keras (Trained neural network)")
    print("  2. class_names.json (5-class labels)")
    print("  3. ood_stats.npz (Mahalanobis Out-of-Distribution gate)")
    print("  4. metrics.json (Evaluation performance)")
    print("  5. confusion_matrix.png (Visual matrix)")
    print(f"  6. model_export.zip (Packaged for deployment)")
    print("\nTo deploy to your local application:")
    print("Extract the contents of model_export.zip into 'backend/model/'")
    print("=" * 70)


if __name__ == "__main__":
    main()
