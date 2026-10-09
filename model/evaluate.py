import os
import sys

# Add model directory to path so imports resolve
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from data_pipeline import get_siamese_dataset
from siamese_cnn import build_siamese_damage_classifier

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
MODEL_PATH = r"D:\geo_spatial_fake_project\model\checkpoints\best_model.keras"
OUTPUT_DIR = r"D:\geo_spatial_fake_project\model"
CONFUSION_IMG_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
BATCH_SIZE = 32

CLASS_NAMES = ["No Damage (0)", "Damaged (1)", "Destroyed (2)"]

# -----------------------------------------------------------------------
# STEP 1: LOAD MODEL
# -----------------------------------------------------------------------
print("=" * 65)
print("Loading model from:", MODEL_PATH)

# Rebuild architecture (registers AbsDifference custom layer)
# then load the saved weights from the .keras file
model = build_siamese_damage_classifier()

# Warm up model with a dummy batch so weights are initialized
dummy = tf.zeros((1, 256, 256, 3))
_ = model([dummy, dummy], training=False)

# Now load saved weights
model.load_weights(MODEL_PATH)
print("Model weights loaded successfully.")
print("=" * 65)


# -----------------------------------------------------------------------
# STEP 2: RUN INFERENCE ON FULL TEST SET
# -----------------------------------------------------------------------
print("\nRunning inference on test split...")
test_ds = get_siamese_dataset(split="test", batch_size=BATCH_SIZE, shuffle=False)

all_true_labels = []
all_pred_labels = []

for (before_batch, after_batch), label_batch in test_ds:
    preds = model([before_batch, after_batch], training=False)
    pred_classes = tf.argmax(preds, axis=1).numpy()
    all_true_labels.extend(label_batch.numpy().tolist())
    all_pred_labels.extend(pred_classes.tolist())

y_true = np.array(all_true_labels)
y_pred = np.array(all_pred_labels)
print(f"Total test samples evaluated: {len(y_true)}")

# -----------------------------------------------------------------------
# STEP 3: OVERALL ACCURACY
# -----------------------------------------------------------------------
overall_accuracy = np.mean(y_true == y_pred) * 100.0
print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%")

# -----------------------------------------------------------------------
# STEP 4: CONFUSION MATRIX
# -----------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

print("\n" + "=" * 65)
print("3x3 CONFUSION MATRIX (rows = True, cols = Predicted)")
print("=" * 65)
header = f"{'':20s} {'Pred: No Dmg':>14} {'Pred: Damaged':>14} {'Pred: Destroyed':>16}"
print(header)
print("-" * 65)
row_labels = ["True: No Damage   ", "True: Damaged     ", "True: Destroyed   "]
for i, row_label in enumerate(row_labels):
    row_str = f"{row_label:20s}"
    for j in range(3):
        row_str += f"{cm[i, j]:>14}"
    print(row_str)
print("=" * 65)

# -----------------------------------------------------------------------
# STEP 5: CLASSIFICATION REPORT (Precision, Recall, F1)
# -----------------------------------------------------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)
print("\nPER-CLASS PRECISION, RECALL, F1:")
print("-" * 65)
print(report)

# -----------------------------------------------------------------------
# STEP 6: SAVE CONFUSION MATRIX HEATMAP
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

# Normalize for colour (row-wise = true class distribution)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Proportion of True Class Predicted", fontsize=11)

tick_marks = [0, 1, 2]
short_names = ["No Damage\n(0)", "Damaged\n(1)", "Destroyed\n(2)"]
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(short_names, fontsize=12)
ax.set_yticklabels(short_names, fontsize=12)
ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
ax.set_ylabel("True Label", fontsize=13, labelpad=10)
ax.set_title(
    f"Siamese CNN — Confusion Matrix\nTest Accuracy: {overall_accuracy:.2f}%",
    fontsize=14,
    fontweight="bold",
    pad=14
)

# Annotate each cell with raw count and percentage
for i in range(3):
    for j in range(3):
        count = cm[i, j]
        pct = cm_norm[i, j] * 100.0
        text_color = "white" if cm_norm[i, j] > 0.55 else "black"
        ax.text(
            j, i,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color=text_color
        )

plt.tight_layout()
plt.savefig(CONFUSION_IMG_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Confusion matrix heatmap saved to: {CONFUSION_IMG_PATH}")
print("=" * 65)
