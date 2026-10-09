"""
Misclassification Analysis
===========================

Analyzes the trained Siamese CNN on the test set.

Focus:
1. Destroyed -> No Damage
2. Destroyed -> Damaged
3. Damaged -> No Damage
4. No Damage -> Damaged

For each error type:
- saves a montage of BEFORE / AFTER images
- shows actual class
- shows predicted class
- shows prediction confidence
- saves metadata for every selected error

Uses the exact same checkpoint-loading method
as evaluate_model.py.
"""

import os
import zipfile
import h5py
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

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

OUTPUT_DIR = (
    r"D:\geo_spatial_fake_project\model"
    r"\misclassification_results"
)

BATCH_SIZE = 32

CLASS_NAMES = [
    "No Damage",
    "Damaged",
    "Destroyed",
]

# Number of examples to show for each error category
MAX_EXAMPLES_PER_TYPE = 20

# Number of columns in each montage
N_COLS = 2


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. EXTRACT WEIGHTS
# ============================================================

print("Extracting trained weights from best_model.keras...")

with zipfile.ZipFile(MODEL_PATH, "r") as z:
    with z.open("model.weights.h5") as src:
        with open(TEMP_WEIGHTS, "wb") as dst:
            dst.write(src.read())

print("Weights extracted.")


# ============================================================
# 2. BUILD MODEL
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
# 3. HELPER FOR CHECKPOINT WEIGHTS
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
            key=lambda x: int(x),
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
        stored,
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
# 4. LOAD TRAINED WEIGHTS
# ============================================================

print("\nLoading trained weights...")

with h5py.File(TEMP_WEIGHTS, "r") as f:

    root_layers = f["layers"]

    # --------------------------------------------------------
    # TOP-LEVEL CLASSIFIER
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
            checkpoint_layer,
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
            checkpoint_layer,
        )


print("Trained weights loaded successfully.")


# ============================================================
# 5. LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = get_siamese_dataset(
    split="test",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Test dataset ready.")


# ============================================================
# 6. RUN PREDICTIONS
# ============================================================

print("\nRunning predictions on test set...")

records = []

sample_index = 0

for batch_inputs, batch_labels in test_ds:

    probabilities = model(
        batch_inputs,
        training=False,
    ).numpy()

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    before_batch = batch_inputs[0].numpy()
    after_batch = batch_inputs[1].numpy()
    labels_batch = batch_labels.numpy()

    for i in range(len(labels_batch)):

        actual = int(labels_batch[i])
        predicted = int(predictions[i])

        confidence = float(
            probabilities[i][predicted]
        )

        records.append(
            {
                "test_index": sample_index,
                "actual": actual,
                "actual_name": CLASS_NAMES[actual],
                "predicted": predicted,
                "predicted_name": CLASS_NAMES[predicted],
                "confidence": confidence,
                "before": before_batch[i],
                "after": after_batch[i],
            }
        )

        sample_index += 1


print(f"Total test samples evaluated: {len(records)}")


# ============================================================
# 7. DEFINE ERROR TYPES
# ============================================================

error_types = {

    "destroyed_to_no_damage": {
        "actual": 2,
        "predicted": 0,
        "title": "Destroyed -> No Damage",
    },

    "destroyed_to_damaged": {
        "actual": 2,
        "predicted": 1,
        "title": "Destroyed -> Damaged",
    },

    "damaged_to_no_damage": {
        "actual": 1,
        "predicted": 0,
        "title": "Damaged -> No Damage",
    },

    "no_damage_to_damaged": {
        "actual": 0,
        "predicted": 1,
        "title": "No Damage -> Damaged",
    },
}


# ============================================================
# 8. SAVE VISUAL MONTAGES
# ============================================================

for error_name, info in error_types.items():

    matching = [
        r
        for r in records
        if r["actual"] == info["actual"]
        and r["predicted"] == info["predicted"]
    ]

    # Sort by confidence, highest first
    matching.sort(
        key=lambda r: r["confidence"],
        reverse=True,
    )

    selected = matching[
        :MAX_EXAMPLES_PER_TYPE
    ]

    print("\n" + "=" * 70)
    print(info["title"])
    print("=" * 70)
    print(f"Total errors: {len(matching)}")
    print(
        f"Visualizing: {len(selected)}"
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_rows = []

    for r in matching:

        metadata_rows.append(
            {
                "test_index": r["test_index"],
                "actual": r["actual_name"],
                "predicted": r["predicted_name"],
                "confidence": r["confidence"],
            }
        )

    metadata_path = os.path.join(
        OUTPUT_DIR,
        f"{error_name}_metadata.csv",
    )

    pd.DataFrame(
        metadata_rows
    ).to_csv(
        metadata_path,
        index=False,
    )

    # --------------------------------------------------------
    # No examples
    # --------------------------------------------------------

    if len(selected) == 0:

        print("No examples found.")

        continue

    # --------------------------------------------------------
    # Create montage
    # --------------------------------------------------------

    n_rows = len(selected)

    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(10, 4 * n_rows),
    )

    if n_rows == 1:
        axes = np.expand_dims(
            axes,
            axis=0,
        )

    for row, r in enumerate(selected):

        # BEFORE
        axes[row, 0].imshow(
            np.clip(
                r["before"],
                0,
                1,
            )
        )

        axes[row, 0].set_title(
            f"BEFORE\n"
            f"Actual: {r['actual_name']}\n"
            f"Predicted: {r['predicted_name']}\n"
            f"Confidence: {r['confidence']:.2%}"
        )

        axes[row, 0].axis("off")

        # AFTER
        axes[row, 1].imshow(
            np.clip(
                r["after"],
                0,
                1,
            )
        )

        axes[row, 1].set_title(
            f"AFTER\n"
            f"Test index: {r['test_index']}"
        )

        axes[row, 1].axis("off")

    fig.suptitle(
        f"Misclassification: {info['title']}",
        fontsize=16,
        y=0.995,
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{error_name}.png",
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved montage: {output_path}"
    )


# ============================================================
# 9. SAVE COMPLETE PREDICTION TABLE
# ============================================================

prediction_rows = []

for r in records:

    prediction_rows.append(
        {
            "test_index": r["test_index"],
            "actual": r["actual"],
            "actual_name": r["actual_name"],
            "predicted": r["predicted"],
            "predicted_name": r["predicted_name"],
            "confidence": r["confidence"],
            "correct": (
                r["actual"] == r["predicted"]
            ),
        }
    )

prediction_csv = os.path.join(
    OUTPUT_DIR,
    "all_test_predictions.csv",
)

pd.DataFrame(
    prediction_rows
).to_csv(
    prediction_csv,
    index=False,
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MISCLASSIFICATION ANALYSIS COMPLETE")
print("=" * 70)

for error_name, info in error_types.items():

    count = sum(
        1
        for r in records
        if r["actual"] == info["actual"]
        and r["predicted"] == info["predicted"]
    )

    print(
        f"{info['title']:<30}: {count}"
    )

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    print("  ", filename)

print("=" * 70)