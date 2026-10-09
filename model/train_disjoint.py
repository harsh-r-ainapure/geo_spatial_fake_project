"""
main_event_disjoint.py
======================

Experiment 2:
Event-Disjoint Training of the Siamese CNN.

This keeps the exact same:
    - Data pipeline
    - Siamese CNN architecture
    - Batch size
    - Epochs
    - Learning rate
    - Optimizer
    - Loss
    - Callbacks

The ONLY experimental change is the dataset split:

TRAIN:
    guatemala-volcano
    hurricane-harvey
    hurricane-michael
    midwest-flooding
    santa-rosa-wildfire
    socal-fire

VALIDATION:
    hurricane-matthew
    mexico-earthquake

TEST:
    hurricane-florence
    palu-tsunami
"""

import os
import tensorflow as tf

# Phase 1: Data Pipeline
from data_pipeline import get_siamese_dataset

# Phase 2: Model Architecture
from siamese_cnn import build_siamese_damage_classifier


# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

NUM_CLASSES = 3
INPUT_SHAPE = (256, 256, 3)

# Event-disjoint metadata
EVENT_DISJOINT_DIR = r"D:\geo_spatial_fake_project\data\event_disjoint"

TRAIN_METADATA = os.path.join(
    EVENT_DISJOINT_DIR,
    "train_metadata.csv"
)

VAL_METADATA = os.path.join(
    EVENT_DISJOINT_DIR,
    "val_metadata.csv"
)

TEST_METADATA = os.path.join(
    EVENT_DISJOINT_DIR,
    "test_metadata.csv"
)

# IMPORTANT:
# Separate checkpoint directory so Experiment 1 is untouched.
CHECKPOINT_DIR = (
    r"D:\geo_spatial_fake_project"
    r"\model\checkpoints_event_disjoint"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# STEP 0: VERIFY METADATA FILES
# -----------------------------------------------------------------------

print("=" * 70)
print("EXPERIMENT 2: EVENT-DISJOINT SIAMESE CNN TRAINING")
print("=" * 70)

print("\nChecking event-disjoint metadata files...")

for path in [
    TRAIN_METADATA,
    VAL_METADATA,
    TEST_METADATA,
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metadata file not found:\n{path}"
        )

    print(f"  OK: {path}")


# -----------------------------------------------------------------------
# STEP 1: LOAD DATASETS
# -----------------------------------------------------------------------

print("\nLoading event-disjoint datasets...")

train_ds = get_siamese_dataset(
    split="train",
    metadata_csv=TRAIN_METADATA,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_ds = get_siamese_dataset(
    split="val",
    metadata_csv=VAL_METADATA,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = get_siamese_dataset(
    split="test",
    metadata_csv=TEST_METADATA,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Datasets ready.")


# -----------------------------------------------------------------------
# STEP 2: BUILD MODEL
# -----------------------------------------------------------------------

print("\nBuilding Siamese CNN model...")

model = build_siamese_damage_classifier(
    input_shape=INPUT_SHAPE,
    feature_dim=256,
    num_classes=NUM_CLASSES,
    dropout_rate=0.3
)


# -----------------------------------------------------------------------
# STEP 3: COMPILE
# -----------------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)


print("\nModel summary:")
model.summary()


# -----------------------------------------------------------------------
# STEP 4: CALLBACKS
# -----------------------------------------------------------------------

callbacks = [

    # Save best model according to validation accuracy
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            CHECKPOINT_DIR,
            "best_model.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    # Same early stopping as Experiment 1
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Same learning-rate reduction as Experiment 1
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]


# -----------------------------------------------------------------------
# STEP 5: TRAIN
# -----------------------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING EVENT-DISJOINT TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# -----------------------------------------------------------------------
# STEP 6: EVALUATE ON TEST SET
# -----------------------------------------------------------------------

print("\n" + "=" * 70)
print("EVALUATING ON EVENT-DISJOINT TEST SET")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(test_ds)

print()
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# -----------------------------------------------------------------------
# STEP 7: FINAL INFORMATION
# -----------------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 2 COMPLETE")
print("=" * 70)

print("\nEVENT-DISJOINT SPLIT")
print("--------------------")

print("\nTRAIN EVENTS:")
print("  - guatemala-volcano")
print("  - hurricane-harvey")
print("  - hurricane-michael")
print("  - midwest-flooding")
print("  - santa-rosa-wildfire")
print("  - socal-fire")

print("\nVALIDATION EVENTS:")
print("  - hurricane-matthew")
print("  - mexico-earthquake")

print("\nTEST EVENTS:")
print("  - hurricane-florence")
print("  - palu-tsunami")

print("\nBest model:")
print(
    os.path.join(
        CHECKPOINT_DIR,
        "best_model.keras"
    )
)

print("\nExperiment 1 checkpoint was NOT modified.")