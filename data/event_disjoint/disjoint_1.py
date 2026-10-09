"""
Create Experiment 2: Event-Disjoint Dataset Split

Every complete event is assigned to exactly one split.
No event can appear in train, validation, and test simultaneously.

Experiment 2:
    TRAIN:
        hurricane-michael
        hurricane-harvey
        socal-fire
        santa-rosa-wildfire
        midwest-flooding
        guatemala-volcano

    VALIDATION:
        hurricane-matthew
        mexico-earthquake

    TEST:
        palu-tsunami
        hurricane-florence
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\geo_spatial_fake_project")
DATA_DIR = PROJECT_DIR / "data"

OUTPUT_DIR = DATA_DIR / "event_disjoint"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Existing Experiment 1 metadata
INPUT_FILES = [
    DATA_DIR / "train_metadata.csv",
    DATA_DIR / "val_metadata.csv",
    DATA_DIR / "test_metadata.csv",
]


# ============================================================
# EVENT ASSIGNMENT
# ============================================================

EVENT_SPLIT = {
    # -------------------------
    # TRAIN
    # -------------------------
    "hurricane-michael": "train",
    "hurricane-harvey": "train",
    "socal-fire": "train",
    "santa-rosa-wildfire": "train",
    "midwest-flooding": "train",
    "guatemala-volcano": "train",

    # -------------------------
    # VALIDATION
    # -------------------------
    "hurricane-matthew": "val",
    "mexico-earthquake": "val",

    # -------------------------
    # TEST
    # -------------------------
    "palu-tsunami": "test",
    "hurricane-florence": "test",
}


EXPECTED_COUNTS = {
    "train": 4399,
    "val": 750,
    "test": 902,
}


# ============================================================
# LOAD ORIGINAL METADATA
# ============================================================

print("=" * 70)
print("EXPERIMENT 2: EVENT-DISJOINT SPLIT")
print("=" * 70)

frames = []

for path in INPUT_FILES:

    if not path.exists():
        raise FileNotFoundError(
            f"Required metadata file not found:\n{path}"
        )

    print(f"Loading: {path}")

    df_part = pd.read_csv(path)

    print(f"  Rows: {len(df_part)}")

    frames.append(df_part)


df = pd.concat(frames, ignore_index=True)


print()
print(f"Combined rows: {len(df)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = {
    "event",
    "patch_class",
    "crop_id",
    "image_name",
    "shard_file",
    "shard_row_idx",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


if df["event"].isna().any():
    raise ValueError("Found rows with missing event values.")


df["event"] = df["event"].astype(str).str.strip()


# ============================================================
# CHECK EVENT LIST
# ============================================================

actual_events = set(df["event"].unique())
expected_events = set(EVENT_SPLIT.keys())

unknown_events = actual_events - expected_events
missing_events = expected_events - actual_events

if unknown_events:
    raise ValueError(
        "Found events that are not assigned to a split:\n"
        + "\n".join(sorted(unknown_events))
    )

if missing_events:
    raise ValueError(
        "Expected events are missing from the metadata:\n"
        + "\n".join(sorted(missing_events))
    )


print()
print(f"Unique events: {len(actual_events)}")

for event in sorted(actual_events):
    print(f"  {event:30s} -> {EVENT_SPLIT[event]}")


# ============================================================
# ASSIGN SPLIT
# ============================================================

df["split"] = df["event"].map(EVENT_SPLIT)


if df["split"].isna().any():
    raise ValueError("Some rows could not be assigned to a split.")


# ============================================================
# VERIFY EVENT ISOLATION
# ============================================================

event_split_counts = (
    df.groupby("event")["split"]
    .nunique()
)

violations = event_split_counts[event_split_counts > 1]

if len(violations) > 0:

    print()
    print("ERROR: EVENT LEAKAGE DETECTED")

    for event in violations.index:
        print(
            f"  {event}: "
            f"{df.loc[df['event'] == event, 'split'].unique()}"
        )

    raise RuntimeError(
        "At least one event appears in multiple splits."
    )

print()
print("EVENT ISOLATION CHECK: PASSED")
print("Every event belongs to exactly one split.")


# ============================================================
# CREATE OUTPUT FILES
# ============================================================

for split in ["train", "val", "test"]:

    split_df = df[df["split"] == split].copy()

    # Remove helper column.
    split_df = split_df.drop(columns=["split"])

    output_path = OUTPUT_DIR / f"{split}_metadata.csv"

    split_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"Created: {output_path}")
    print(f"Rows   : {len(split_df)}")


# ============================================================
# VERIFY EXPECTED COUNTS
# ============================================================

print()
print("=" * 70)
print("SPLIT COUNTS")
print("=" * 70)

actual_counts = {}

for split in ["train", "val", "test"]:

    path = OUTPUT_DIR / f"{split}_metadata.csv"

    check_df = pd.read_csv(path)

    actual_counts[split] = len(check_df)

    print(
        f"{split.upper():12s}: "
        f"{len(check_df):5d} rows "
        f"(expected {EXPECTED_COUNTS[split]:5d})"
    )

    if len(check_df) != EXPECTED_COUNTS[split]:
        raise RuntimeError(
            f"{split} count mismatch: "
            f"expected {EXPECTED_COUNTS[split]}, "
            f"got {len(check_df)}"
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

CLASS_NAMES = {
    0: "No Damage",
    1: "Damaged",
    2: "Destroyed",
}

for split in ["train", "val", "test"]:

    split_df = pd.read_csv(
        OUTPUT_DIR / f"{split}_metadata.csv"
    )

    counts = (
        split_df["patch_class"]
        .value_counts()
        .reindex([0, 1, 2], fill_value=0)
    )

    print()
    print(split.upper())

    total = len(split_df)

    for class_id, count in counts.items():

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"  {CLASS_NAMES[class_id]:12s}: "
            f"{count:5d} ({percentage:6.2f}%)"
        )


# ============================================================
# EVENT DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("EVENT DISTRIBUTION")
print("=" * 70)

for split in ["train", "val", "test"]:

    split_df = pd.read_csv(
        OUTPUT_DIR / f"{split}_metadata.csv"
    )

    print()
    print(split.upper())

    event_counts = (
        split_df["event"]
        .value_counts()
        .sort_index()
    )

    for event, count in event_counts.items():

        print(
            f"  {event:30s}: {count:5d}"
        )


# ============================================================
# FINAL CHECKS
# ============================================================

train_events = set(
    pd.read_csv(
        OUTPUT_DIR / "train_metadata.csv"
    )["event"]
)

val_events = set(
    pd.read_csv(
        OUTPUT_DIR / "val_metadata.csv"
    )["event"]
)

test_events = set(
    pd.read_csv(
        OUTPUT_DIR / "test_metadata.csv"
    )["event"]
)


assert train_events.isdisjoint(val_events)
assert train_events.isdisjoint(test_events)
assert val_events.isdisjoint(test_events)


total_output_rows = (
    len(pd.read_csv(OUTPUT_DIR / "train_metadata.csv"))
    + len(pd.read_csv(OUTPUT_DIR / "val_metadata.csv"))
    + len(pd.read_csv(OUTPUT_DIR / "test_metadata.csv"))
)


assert total_output_rows == len(df)


print()
print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print("Train ∩ Validation : 0 events")
print("Train ∩ Test       : 0 events")
print("Validation ∩ Test  : 0 events")
print(f"Total output rows  : {total_output_rows}")
print(f"Original rows      : {len(df)}")

print()
print("ALL EVENT-DISJOINT CHECKS PASSED")
print()
print("Experiment 2 metadata created successfully.")
print("=" * 70)