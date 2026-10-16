"""
dataset_loader.py — Efficient data loading using tf.data pipelines.

Key features:
- Reads images directly from disk (never loads entire dataset into RAM)
- Automatically detects class names from folder structure
- Applies augmentation only to training data
- Normalizes pixel values to [0, 1]
"""

import os
import json
import tensorflow as tf


# ─── Configuration ────────────────────────────────────────────────────────────

IMAGE_SIZE = (64, 64)      # Resize all images to this size
CHANNELS   = 3             # RGB
BATCH_SIZE = 32            # Images per batch (lower if you run out of RAM)
AUTOTUNE   = tf.data.AUTOTUNE


# ─── Class Discovery ──────────────────────────────────────────────────────────

def get_class_names(dataset_root: str) -> list:
    """
    Scan the 'train' directory and return a sorted list of class names.
    Each sub-folder name becomes one gesture class.

    Example: dataset/train/A, dataset/train/hello  →  ['A', 'hello', ...]

    Args:
        dataset_root : Path to the top-level dataset folder

    Returns:
        Sorted list of class name strings
    """
    train_dir = os.path.join(dataset_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Could not find 'train' folder inside: {dataset_root}\n"
            "Make sure your path is correct and the folder structure is:\n"
            "  dataset/\n    train/\n      A/\n      B/\n      ...\n    val/\n      ..."
        )

    # Only include entries that are directories (skip stray files)
    class_names = sorted([
        entry for entry in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, entry))
    ])

    if len(class_names) == 0:
        raise ValueError(f"No class sub-folders found in: {train_dir}")

    print(f"[Dataset] Found {len(class_names)} classes: {class_names}")
    return class_names


def save_class_names(class_names: list, save_path: str = "class_names.json"):
    """
    Save class names to a JSON file so you can load them later during inference.

    Args:
        class_names : List of class name strings
        save_path   : File path for the JSON output
    """
    with open(save_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[Dataset] Class names saved to: {save_path}")


def load_class_names(path: str = "class_names.json") -> list:
    """
    Load previously saved class names from JSON.

    Args:
        path : Path to class_names.json

    Returns:
        List of class name strings
    """
    with open(path, "r") as f:
        return json.load(f)


# ─── Image Loading Helpers ────────────────────────────────────────────────────

def load_and_preprocess(image_path: tf.Tensor, label: tf.Tensor) -> tuple:
    """
    Read one image file, decode it, resize, and normalize to [0, 1].
    This function runs inside the tf.data pipeline (on CPU, lazily).

    Args:
        image_path : String tensor — path to an image file
        label      : Integer tensor — class index

    Returns:
        Tuple of (normalized_image_tensor, label)
    """
    # Read raw bytes from disk
    raw = tf.io.read_file(image_path)

    # Decode JPEG or PNG automatically; force 3 colour channels
    image = tf.image.decode_image(raw, channels=CHANNELS, expand_animations=False)

    # Resize to fixed size (tf.data requires static shapes)
    image = tf.image.resize(image, IMAGE_SIZE)

    # Normalize pixel values: 0–255  →  0.0–1.0
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ─── Augmentation (Training Only) ─────────────────────────────────────────────

def augment(image: tf.Tensor, label: tf.Tensor) -> tuple:
    """
    Apply random augmentations to make the model more robust.
    These transforms are applied on-the-fly so they never repeat exactly.

    Augmentations used:
    - Random horizontal flip   (gesture mirrors)
    - Random brightness shift  (different lighting conditions)
    - Random contrast shift    (shadows / highlights)
    - Random hue shift         (slight skin tone variation)

    Args:
        image : Float tensor [H, W, C] in [0, 1]
        label : Integer class label

    Returns:
        Tuple of (augmented_image, label)
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)

    # Clip back to valid range after augmentations
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_dataset(
    split_dir: str,
    class_names: list,
    batch_size: int = BATCH_SIZE,
    is_training: bool = True,
    shuffle_buffer: int = 2000,
) -> tf.data.Dataset:
    """
    Build an efficient tf.data.Dataset for one split (train or val).

    Pipeline:
      file paths  →  shuffle  →  load+preprocess  →  [augment]  →  batch  →  prefetch

    Args:
        split_dir      : Path to the split folder (e.g. "dataset/train")
        class_names    : Ordered list of class names (index = label integer)
        batch_size     : Number of images per batch
        is_training    : If True, apply shuffling + augmentation
        shuffle_buffer : How many samples to hold in the shuffle buffer

    Returns:
        A tf.data.Dataset that yields (batch_of_images, batch_of_labels)
    """
    # ── Collect all file paths and their integer labels ──────────────────────
    all_paths  = []
    all_labels = []

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[Dataset] WARNING: Expected folder not found: {class_dir}")
            continue

        label = class_to_idx[class_name]
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(label)

    total = len(all_paths)
    if total == 0:
        raise ValueError(f"No images found in: {split_dir}")

    print(f"[Dataset] '{os.path.basename(split_dir)}' split — {total} images across {len(class_names)} classes")

    # ── Create Dataset from lists of paths + labels ───────────────────────────
    ds = tf.data.Dataset.from_tensor_slices((all_paths, all_labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(shuffle_buffer, total), reshuffle_each_iteration=True)

    # ── Map: load & preprocess (runs in parallel threads) ────────────────────
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    # ── Augmentation for training only ───────────────────────────────────────
    if is_training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    # ── Batch + Prefetch ─────────────────────────────────────────────────────
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)   # overlap data loading with model training

    return ds


# ─── Convenience Function ─────────────────────────────────────────────────────

def get_datasets(dataset_root: str, batch_size: int = BATCH_SIZE):
    """
    One-stop function: discover classes, build train + val datasets.

    Args:
        dataset_root : Path to the top-level dataset folder
        batch_size   : Batch size for both datasets

    Returns:
        train_ds     : Training tf.data.Dataset
        val_ds       : Validation tf.data.Dataset
        class_names  : List of class name strings
        num_classes  : Number of unique gesture classes
    """
    class_names = get_class_names(dataset_root)
    num_classes = len(class_names)

    # Save class names so predict.py can load them later
    save_class_names(class_names, "class_names.json")

    train_dir = os.path.join(dataset_root, "train")
    val_dir   = os.path.join(dataset_root, "val")

    train_ds = build_dataset(train_dir, class_names, batch_size=batch_size, is_training=True)
    val_ds   = build_dataset(val_dir,   class_names, batch_size=batch_size, is_training=False)

    return train_ds, val_ds, class_names, num_classes


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    dataset_path = r"C:\Users\riyan\Desktop\dataset"
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    train_ds, val_ds, classes, n = get_datasets(dataset_path)
    print(f"\nClasses ({n}): {classes}")

    # Peek at one batch
    for images, labels in train_ds.take(1):
        print(f"Batch shape : {images.shape}")
        print(f"Labels      : {labels.numpy()}")
        print(f"Pixel range : [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
