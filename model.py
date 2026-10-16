"""
model.py — Custom CNN built completely from scratch.
No pretrained weights, no transfer learning.
All weights are randomly initialized and trained only on your dataset.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_custom_cnn(num_classes: int, input_shape: tuple = (64, 64, 3)) -> tf.keras.Model:
    """
    Build a custom Convolutional Neural Network from scratch.

    Architecture overview:
    - 4 convolutional blocks (each: Conv2D → BatchNorm → ReLU → MaxPool)
    - Global Average Pooling to reduce spatial dimensions
    - 2 fully connected (Dense) layers with Dropout for regularization
    - Softmax output layer for multi-class classification

    Args:
        num_classes  : Number of gesture classes (auto-detected from dataset folders)
        input_shape  : (height, width, channels) — default 64×64 RGB

    Returns:
        A compiled Keras Model ready for training
    """

    # ─── Input ────────────────────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ─── Block 1: Learn basic edges / colors ──────────────────────────────────
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",          # keep spatial size
        kernel_initializer="he_normal",   # good random init for ReLU nets
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv1"
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)   # 64→32

    # ─── Block 2: Learn simple textures / shapes ──────────────────────────────
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)   # 32→16

    # ─── Block 3: Learn finger / hand part patterns ───────────────────────────
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv3"
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)   # 16→8

    # ─── Block 4: Learn high-level gesture representations ────────────────────
    x = layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv4"
    )(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)   # 8→4

    # ─── Global Average Pooling (replaces Flatten, fewer parameters) ──────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ─── Fully Connected Head ─────────────────────────────────────────────────
    x = layers.Dense(512, kernel_initializer="he_normal", name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Activation("relu", name="relu_fc1")(x)
    x = layers.Dropout(0.5, name="dropout1")(x)   # 50% dropout prevents overfitting

    x = layers.Dense(256, kernel_initializer="he_normal", name="fc2")(x)
    x = layers.BatchNormalization(name="bn_fc2")(x)
    x = layers.Activation("relu", name="relu_fc2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)

    # ─── Output Layer ─────────────────────────────────────────────────────────
    outputs = layers.Dense(
        num_classes,
        activation="softmax",    # probabilities across all gesture classes
        name="predictions"
    )(x)

    # ─── Build Model ──────────────────────────────────────────────────────────
    model = models.Model(inputs=inputs, outputs=outputs, name="GestureCNN_Scratch")

    # ─── Compile ──────────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # Quick sanity check — run this file directly to preview the architecture
    model = build_custom_cnn(num_classes=32)
    model.summary()
