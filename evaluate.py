"""
evaluate.py — Detailed evaluation of the trained model on the validation set.

Produces:
  - Per-class accuracy table
  - Confusion matrix (saved as confusion_matrix.png)
  - Classification report (precision / recall / F1)

Usage:
    python evaluate.py
    python evaluate.py --data C:\\path\\to\\dataset
"""

import os
import argparse
import json

import numpy as np
import tensorflow as tf

from dataset_loader import get_class_names, build_dataset, IMAGE_SIZE, CHANNELS, BATCH_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained gesture recognition model")
    parser.add_argument("--data",    type=str, default=r"C:\Users\riyan\Desktop\dataset")
    parser.add_argument("--model",   type=str, default="saved_model/gesture_model.keras")
    parser.add_argument("--classes", type=str, default="class_names.json")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from: {args.model}")
    model = tf.keras.models.load_model(args.model)

    # ── Load class names ──────────────────────────────────────────────────────
    with open(args.classes) as f:
        class_names = json.load(f)
    print(f"Classes ({len(class_names)}): {class_names}\n")

    # ── Build validation dataset ──────────────────────────────────────────────
    val_dir = os.path.join(args.data, "val")
    val_ds  = build_dataset(val_dir, class_names, batch_size=args.batch, is_training=False)

    # ── Collect all predictions ───────────────────────────────────────────────
    print("Running predictions on validation set...")
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        probs  = model.predict(images, verbose=0)
        preds  = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    overall_acc = np.mean(y_true == y_pred)
    print(f"\n{'='*50}")
    print(f"  Overall Accuracy: {overall_acc*100:.2f}%")
    print(f"{'='*50}\n")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    print(f"{'Class':<15} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print("-" * 45)
    for idx, name in enumerate(class_names):
        mask    = y_true == idx
        correct = np.sum(y_pred[mask] == idx)
        total   = np.sum(mask)
        acc     = correct / total if total > 0 else 0
        print(f"{name:<15} {correct:>8} {total:>7} {acc*100:>9.1f}%")

    # ── Sklearn report (if available) ─────────────────────────────────────────
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        print("\n\n── Classification Report ──\n")
        print(classification_report(y_true, y_pred, target_names=class_names))

        # ── Confusion matrix plot ─────────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(max(10, len(class_names)), max(8, len(class_names)-2)))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=150)
            plt.close()
            print("\nConfusion matrix saved → confusion_matrix.png")
        except ImportError:
            pass

    except ImportError:
        print("\nTip: install scikit-learn for detailed per-class metrics:")
        print("  pip install scikit-learn")


if __name__ == "__main__":
    main()
