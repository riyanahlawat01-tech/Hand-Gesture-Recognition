"""
train.py — Train the custom CNN gesture recognition model from scratch.

Usage:
    python train.py                                  # uses default dataset path below
    python train.py --data C:\\path\\to\\your\\dataset  # custom path
    python train.py --epochs 30 --batch 64           # custom hyperparameters

What this script does:
  1. Scans your dataset folder and auto-detects all gesture classes
  2. Builds the tf.data pipelines (no RAM overload)
  3. Creates the custom CNN with randomly initialized weights
  4. Trains for N epochs using Adam optimizer
  5. Uses callbacks: early stopping, learning rate scheduler, model checkpointing
  6. Saves the best model to  saved_model/gesture_model.h5
  7. Plots training curves and saves them as training_history.png
"""

import os
import argparse
import json

import numpy as np
import tensorflow as tf

# ── Local modules (same folder) ───────────────────────────────────────────────
from dataset_loader import get_datasets, BATCH_SIZE, IMAGE_SIZE, CHANNELS
from model import build_custom_cnn


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing  (easy to override everything from the command line)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hand Gesture CNN from scratch")

    parser.add_argument(
        "--data",
        type=str,
        default=r"C:\Users\riyan\Desktop\dataset",
        help="Path to your dataset root folder (must contain train/ and val/ sub-folders)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (early stopping may halt sooner)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
        help="Batch size (reduce to 16 if you see memory errors)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for the Adam optimizer"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_model",
        help="Directory to save the trained model"
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule  (halve LR when validation loss plateaus)
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(save_dir: str) -> list:
    """
    Build the list of Keras callbacks used during training.

    Callbacks:
    - ModelCheckpoint  : saves the model whenever val_accuracy improves
    - EarlyStopping    : stops training if val_loss doesn't improve for 10 epochs
    - ReduceLROnPlateau: halves the learning rate after 5 stagnant epochs
    - CSVLogger        : writes per-epoch metrics to training_log.csv

    Args:
        save_dir : Folder where the best model will be saved

    Returns:
        List of tf.keras.callbacks
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "gesture_model.h5")

    callbacks = [
        # ── Save best model ─────────────────────────────────────────────────
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        # ── Stop early if no improvement ────────────────────────────────────
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,          # wait 10 epochs before giving up
            restore_best_weights=True,
            verbose=1
        ),

        # ── Reduce learning rate on plateau ─────────────────────────────────
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,           # new_lr = lr * 0.5
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),

        # ── CSV log file ─────────────────────────────────────────────────────
        tf.keras.callbacks.CSVLogger("training_log.csv"),
    ]

    return callbacks, model_path


# ─────────────────────────────────────────────────────────────────────────────
# Plot training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history):
    """
    Save accuracy and loss curves as a PNG image.
    Uses matplotlib if available; skips gracefully otherwise.

    Args:
        history : Keras History object returned by model.fit()
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (works on servers)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training History — Hand Gesture CNN", fontsize=14)

        # ── Accuracy ────────────────────────────────────────────────────────
        axes[0].plot(history.history["accuracy"],     label="Train Acc",  linewidth=2)
        axes[0].plot(history.history["val_accuracy"], label="Val Acc",    linewidth=2)
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ── Loss ────────────────────────────────────────────────────────────
        axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
        axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150)
        plt.close()
        print("\n[Train] Training curves saved → training_history.png")

    except ImportError:
        print("\n[Train] matplotlib not found — skipping plot. Install with: pip install matplotlib")


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  Hand Gesture Recognition — Training from Scratch")
    print("=" * 60)
    print(f"  Dataset  : {args.data}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  LR       : {args.lr}")
    print(f"  Save dir : {args.save_dir}")
    print("=" * 60)

    # ── Step 1: Build data pipelines ─────────────────────────────────────────
    print("\n[Step 1/4] Loading dataset...")
    train_ds, val_ds, class_names, num_classes = get_datasets(
        dataset_root=args.data,
        batch_size=args.batch
    )

    print(f"\n  Classes detected ({num_classes}): {class_names}")

    # ── Step 2: Build model ───────────────────────────────────────────────────
    print("\n[Step 2/4] Building custom CNN from scratch...")
    input_shape = IMAGE_SIZE + (CHANNELS,)   # (64, 64, 3)

    model = build_custom_cnn(
        num_classes=num_classes,
        input_shape=input_shape
    )

    # Override learning rate from CLI argument
    model.optimizer.learning_rate.assign(args.lr)

    model.summary()

    total_params = model.count_params()
    print(f"\n  Total trainable parameters: {total_params:,}")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("\n[Step 3/4] Training...")
    callbacks, model_path = get_callbacks(args.save_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # ── Step 4: Evaluate + Save ───────────────────────────────────────────────
    print("\n[Step 4/4] Final evaluation on validation set...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n  ✔  Validation Loss     : {val_loss:.4f}")
    print(f"  ✔  Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"\n  Best model saved to   : {model_path}")
    print(f"  Class names saved to  : class_names.json")

    # Save training results summary
    summary = {
        "num_classes"       : num_classes,
        "class_names"       : class_names,
        "total_epochs"      : len(history.history["accuracy"]),
        "best_val_accuracy" : float(max(history.history["val_accuracy"])),
        "final_val_loss"    : float(val_loss),
    }
    with open("training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Training summary saved: training_summary.json")

    # Plot curves
    plot_history(history)

    print("\n" + "=" * 60)
    print("  Training complete! Next step: run predict.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
