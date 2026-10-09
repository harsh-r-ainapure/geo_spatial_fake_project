"""
main.py
=======
Entry point that wires together:
  - data_pipeline.py  (Phase 1: tf.data loading & batching)
  - siamese_model.py  (Phase 2: Siamese CNN architecture)
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
CHECKPOINT_DIR = r"D:\geo_spatial_fake_project\model\checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# STEP 1: LOAD DATASETS (Phase 1 pipeline)
# -----------------------------------------------------------------------
print("Loading datasets...")
train_ds = get_siamese_dataset(split="train", batch_size=BATCH_SIZE, shuffle=True)
val_ds = get_siamese_dataset(split="val", batch_size=BATCH_SIZE, shuffle=False)
test_ds = get_siamese_dataset(split="test", batch_size=BATCH_SIZE, shuffle=False)
print("Datasets ready.")

# -----------------------------------------------------------------------
# STEP 2: BUILD MODEL (Phase 2 architecture)
# -----------------------------------------------------------------------
print("Building Siamese CNN model...")
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()

# -----------------------------------------------------------------------
# STEP 4: CALLBACKS
# -----------------------------------------------------------------------
callbacks = [
    # Save the best model (by val accuracy)
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    # Stop early if val loss stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce LR when val loss plateaus
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
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# -----------------------------------------------------------------------
# STEP 6: EVALUATE ON TEST SET
# -----------------------------------------------------------------------
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")