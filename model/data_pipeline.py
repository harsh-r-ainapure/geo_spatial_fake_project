"""
Phase 1: Bi-Temporal Data Pipeline for Siamese CNN
===================================================
Loads 256x256 contextual crops from metadata CSVs, extracts paired Before/After
satellite imagery (TIFF/PNG/Parquet), normalizes pixel values to [0.0, 1.0],
and yields TensorFlow batches directly consumable by a Siamese CNN architecture.
"""

import os
from collections import OrderedDict
from io import BytesIO
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# Suppress verbose TensorFlow / oneDNN logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = r"D:\geo_spatial_fake_project\data"
DEFAULT_WORKSPACE_DIR = r"D:\geo_spatial_fake_project"

LABEL_MAP = {
    0: "No Damage",
    1: "Damaged",
    2: "Destroyed",
}

IMG_SIZE = 256
NUM_CHANNELS = 3


def resolve_metadata_path(split: str, custom_path: Optional[str] = None) -> str:
    """
    Locates the metadata CSV file for the requested split.
    Checks custom path, data/ directory, and workspace root.
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path

    filename = f"{split}_metadata.csv"
    candidate_paths = [
        os.path.join(DEFAULT_DATA_DIR, filename),
        os.path.join(DEFAULT_WORKSPACE_DIR, filename),
        os.path.join(DEFAULT_DATA_DIR, "filtered_256_crops_metadata.csv"),
    ]

    for p in candidate_paths:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Could not locate metadata file for split '{split}'. Checked candidates:\n"
        + "\n".join(candidate_paths)
    )


class SiameseDataLoader:
    """
    Handles file locating, image decoding, cropping, RGB verification,
    and normalization for bi-temporal image pairs.
    Includes an LRU shard cache to ensure fast I/O throughput.
    """

    def __init__(self, metadata_path: str, tiff_dir: Optional[str] = None, max_cached_shards: int = 6):
        self.metadata_path = metadata_path
        self.df = pd.read_csv(metadata_path)
        self.tiff_dir = tiff_dir
        self.max_cached_shards = max_cached_shards
        
        # LRU cache for parquet shards
        self._shard_cache = OrderedDict()

        self._validate_metadata()

    def _validate_metadata(self):
        required_cols = {"crop_ymin", "crop_xmin", "crop_ymax", "crop_xmax", "patch_class"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Metadata is missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _get_shard_data(self, shard_file: str) -> pd.DataFrame:
        """Retrieves parquet shard DataFrame from LRU cache or disk."""
        if shard_file in self._shard_cache:
            self._shard_cache.move_to_end(shard_file)
            return self._shard_cache[shard_file]

        # Load from disk
        shard_df = pd.read_parquet(shard_file, columns=["t1_image", "t2_image"])
        self._shard_cache[shard_file] = shard_df

        # Evict oldest if exceeding max cache size
        if len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)

        return shard_df

    def load_pair_and_crop(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Locates and extracts Before (T1) and After (T2) images for row `idx`,
        crops them to the identical (256, 256) bounding box, ensures RGB,
        and normalizes to float32 in range [0.0, 1.0].
        """
        row = self.df.iloc[idx]
        ymin = int(row["crop_ymin"])
        xmin = int(row["crop_xmin"])
        ymax = int(row["crop_ymax"])
        xmax = int(row["crop_xmax"])
        label = int(row["patch_class"])

        t1_img = None
        t2_img = None

        # Option A: Check if loose TIFF/PNG files exist on disk in tiff_dir
        if self.tiff_dir and os.path.exists(self.tiff_dir):
            base_name = os.path.basename(row["image_name"])
            t1_candidates = [
                os.path.join(self.tiff_dir, f"{base_name}_pre_disaster.tif"),
                os.path.join(self.tiff_dir, f"{base_name}_pre_disaster.png"),
                os.path.join(self.tiff_dir, f"{base_name}_t1.tif"),
            ]
            t2_candidates = [
                os.path.join(self.tiff_dir, f"{base_name}_post_disaster.tif"),
                os.path.join(self.tiff_dir, f"{base_name}_post_disaster.png"),
                os.path.join(self.tiff_dir, f"{base_name}_t2.tif"),
            ]
            for p1, p2 in zip(t1_candidates, t2_candidates):
                if os.path.exists(p1) and os.path.exists(p2):
                    t1_img = Image.open(p1).convert("RGB")
                    t2_img = Image.open(p2).convert("RGB")
                    break

        # Option B: Load directly from the dataset parquet shard (source of truth)
        if t1_img is None or t2_img is None:
            shard_path = row["shard_file"]
            row_idx = int(row["shard_row_idx"])
            shard_df = self._get_shard_data(shard_path)
            sample_data = shard_df.iloc[row_idx]

            t1_bytes = sample_data["t1_image"]["bytes"]
            t2_bytes = sample_data["t2_image"]["bytes"]

            t1_img = Image.open(BytesIO(t1_bytes)).convert("RGB")
            t2_img = Image.open(BytesIO(t2_bytes)).convert("RGB")

        # Apply identical crop: PIL crop box is (left, upper, right, lower)
        t1_crop = t1_img.crop((xmin, ymin, xmax, ymax))
        t2_crop = t2_img.crop((xmin, ymin, xmax, ymax))

        # Enforce exact 256x256 dimensions
        if t1_crop.size != (IMG_SIZE, IMG_SIZE):
            t1_crop = t1_crop.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        if t2_crop.size != (IMG_SIZE, IMG_SIZE):
            t2_crop = t2_crop.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

        # Convert to numpy float32 and normalize [0, 255] -> [0.0, 1.0]
        t1_arr = np.asarray(t1_crop, dtype=np.float32) / 255.0
        t2_arr = np.asarray(t2_crop, dtype=np.float32) / 255.0

        return t1_arr, t2_arr, label


def create_dataset_generator(loader: SiameseDataLoader, shuffle: bool, seed: int = 42):
    """
    Generator function that yields individual ((t1, t2), label) samples.
    When shuffle is True, performs shard-aware block shuffling to maximize I/O throughput.
    """
    if not shuffle:
        for idx in range(len(loader)):
            t1, t2, label = loader.load_pair_and_crop(idx)
            yield (t1, t2), label
    else:
        rng = np.random.RandomState(seed)
        shards = loader.df["shard_file"].unique()
        rng.shuffle(shards)
        
        for shard in shards:
            shard_indices = loader.df[loader.df["shard_file"] == shard].index.to_numpy()
            rng.shuffle(shard_indices)
            for idx in shard_indices:
                t1, t2, label = loader.load_pair_and_crop(idx)
                yield (t1, t2), label


def get_siamese_dataset(
    split: str = "train",
    metadata_csv: Optional[str] = None,
    tiff_dir: Optional[str] = None,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    seed: int = 42,
    shuffle_buffer_size: int = 0,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """
    Factory function creating an optimized tf.data.Dataset ready for Siamese CNN training.

    Args:
        split: 'train', 'val', or 'test'.
        metadata_csv: Optional explicit path to metadata CSV.
        tiff_dir: Optional path to loose TIFF/image folder.
        batch_size: Batch size (default 32).
        shuffle: Whether to shuffle. If None, defaults to True for 'train', False for 'val'/'test'.
        seed: Random seed for shuffling.
        shuffle_buffer_size: Optional in-memory buffer shuffle (0 to disable extra buffer).
        prefetch: Whether to prefetch batches with tf.data.AUTOTUNE.

    Returns:
        tf.data.Dataset yielding:
          ((before_batch, after_batch), label_batch)
          where before_batch and after_batch have shape (B, 256, 256, 3), float32 in [0, 1]
          and label_batch has shape (B,), int32 in {0, 1, 2}.
    """
    csv_path = resolve_metadata_path(split, metadata_csv)
    loader = SiameseDataLoader(csv_path, tiff_dir=tiff_dir)

    is_train = split.lower().startswith("train")
    should_shuffle = (shuffle if shuffle is not None else is_train)

    output_signature = (
        (
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: create_dataset_generator(loader, shuffle=should_shuffle, seed=seed),
        output_signature=output_signature,
    )

    # Optional in-memory buffer shuffle
    if should_shuffle and shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

    # Batching
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # Prefetch for asynchronous pipeline execution
    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset