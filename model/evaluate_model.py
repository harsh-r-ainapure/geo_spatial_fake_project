"""
evaluate_model.py
=================
Evaluate the trained Siamese CNN checkpoint.

Loads the Keras 3 weights directly from best_model.keras,
including the nested shared CNN encoder.
"""

import os
import zipfile
import h5py
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from data_pipeline import get_siamese_dataset
from siamese_cnn import build_siamese_damage_classifier


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = (
    r"D:\geo_spatial_fake_project\model\checkpoints\best_model.keras"
)

TEMP_WEIGHTS = (
    r"D:\geo_spatial_fake_project\model\checkpoints"
    r"\best_model_weights.h5"
)

BATCH_SIZE = 32

CLASS_NAMES = [
    "No Damage",
    "Damaged",
    "Destroyed",
]


# ============================================================
# EXTRACT WEIGHTS FROM .KERAS
# ============================================================

print("Extracting trained weights from best_model.keras...")

with zipfile.ZipFile(MODEL_PATH, "r") as z:

    with z.open("model.weights.h5") as src:

        with open(TEMP_WEIGHTS, "wb") as dst:
            dst.write(src.read())

print("Weights extracted.")


# ============================================================
# BUILD CURRENT ARCHITECTURE
# ============================================================

print("\nBuilding model architecture...")

model = build_siamese_damage_classifier(
    input_shape=(256, 256, 3),
    feature_dim=256,
    num_classes=3,
    dropout_rate=0.3,
)

print("Model architecture created.")


# ============================================================
# HELPER: LOAD VARIABLES FROM A CHECKPOINT LAYER
# ============================================================

def assign_layer_weights(model_layer, checkpoint_layer):

    if "vars" not in checkpoint_layer:
        return

    checkpoint_vars = checkpoint_layer["vars"]

    if len(model_layer.weights) == 0:
        return

    stored = [
        checkpoint_vars[k][()]
        for k in sorted(
            checkpoint_vars.keys(),
            key=lambda x: int(x)
        )
    ]

    if len(stored) != len(model_layer.weights):
        raise RuntimeError(
            f"Variable count mismatch for {model_layer.name}: "
            f"model={len(model_layer.weights)}, "
            f"checkpoint={len(stored)}"
        )

    for model_weight, stored_weight in zip(
        model_layer.weights,
        stored
    ):

        if tuple(model_weight.shape) != tuple(
            stored_weight.shape
        ):
            raise RuntimeError(
                f"Shape mismatch for {model_layer.name}: "
                f"model={tuple(model_weight.shape)}, "
                f"checkpoint={stored_weight.shape}"
            )

        model_weight.assign(stored_weight)


# ============================================================
# LOAD WEIGHTS
# ============================================================

print("\nLoading trained weights...")

with h5py.File(TEMP_WEIGHTS, "r") as f:

    root_layers = f["layers"]

    # --------------------------------------------------------
    # TOP-LEVEL CLASSIFIER LAYERS
    # --------------------------------------------------------

    top_level_mapping = {
        "dense_1": "dense",
        "dense_2": "dense_1",
        "damage_probabilities": "dense_2",

        "bn_dense_1": "batch_normalization",
        "bn_dense_2": "batch_normalization_1",
    }

    for model_name, checkpoint_name in top_level_mapping.items():

        model_layer = model.get_layer(model_name)

        checkpoint_layer = root_layers[checkpoint_name]

        assign_layer_weights(
            model_layer,
            checkpoint_layer
        )

    # --------------------------------------------------------
    # SHARED CNN ENCODER
    # --------------------------------------------------------

    encoder = model.get_layer("shared_cnn_encoder")

    checkpoint_encoder = root_layers["functional"]

    encoder_layers = checkpoint_encoder["layers"]

    encoder_mapping = {
        "conv1": "conv2d",
        "bn1": "batch_normalization",

        "conv2": "conv2d_1",
        "bn2": "batch_normalization_1",

        "conv3": "conv2d_2",
        "bn3": "batch_normalization_2",

        "conv4": "conv2d_3",
        "bn4": "batch_normalization_3",
    }

    for model_name, checkpoint_name in encoder_mapping.items():

        model_layer = encoder.get_layer(model_name)

        checkpoint_layer = encoder_layers[checkpoint_name]

        assign_layer_weights(
            model_layer,
            checkpoint_layer
        )


print("Trained weights loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = get_siamese_dataset(
    split="test",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Test dataset ready.")


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\nRunning predictions on test set...")

y_true = []
y_pred = []


for batch_inputs, batch_labels in test_ds:

    probabilities = model(
        batch_inputs,
        training=False,
    )

    predictions = tf.argmax(
        probabilities,
        axis=1,
    ).numpy()

    y_pred.extend(
        predictions.tolist()
    )

    y_true.extend(
        batch_labels.numpy().tolist()
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2],
)

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print("Rows    = Actual")
print("Columns = Predicted\n")

print(
    f"{'':15}"
    f"{'No Damage':>12}"
    f"{'Damaged':>12}"
    f"{'Destroyed':>12}"
)

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{name:<15}"
        f"{cm[i, 0]:>12}"
        f"{cm[i, 1]:>12}"
        f"{cm[i, 2]:>12}"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("PER-CLASS PERFORMANCE")
print("=" * 65)

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

print("=" * 65)
print(
    f"OVERALL TEST ACCURACY : "
    f"{accuracy:.4f}"
)

print(
    f"OVERALL TEST ACCURACY : "
    f"{accuracy * 100:.2f}%"
)

print("=" * 65)