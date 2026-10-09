"""
Error Audit for Siamese CNN
===========================

Connects test predictions back to the original crop metadata.

Analyzes:
1. Destroyed -> No Damage
2. Destroyed -> Damaged
3. Damaged -> No Damage
4. No Damage -> Damaged

Compares errors against correctly classified samples using:
- building coverage
- damaged pixels
- destroyed pixels
- total building pixels
- event
- number of target buildings
- cluster status
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

WORKSPACE = r"D:\geo_spatial_fake_project"

TEST_METADATA = os.path.join(
    WORKSPACE,
    "data",
    "test_metadata.csv"
)

PREDICTIONS = os.path.join(
    WORKSPACE,
    "model",
    "misclassification_results",
    "all_test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    WORKSPACE,
    "model",
    "error_audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "No Damage",
    1: "Damaged",
    2: "Destroyed",
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING TEST METADATA AND PREDICTIONS")
print("=" * 70)

print("\nTest metadata:")
print(TEST_METADATA)

print("\nPredictions:")
print(PREDICTIONS)

metadata = pd.read_csv(TEST_METADATA)
predictions = pd.read_csv(PREDICTIONS)

print("\nMetadata rows :", len(metadata))
print("Prediction rows:", len(predictions))


# ============================================================
# VERIFY INDEX ALIGNMENT
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING TEST INDEX ALIGNMENT")
print("=" * 70)

expected_indices = np.arange(len(metadata))

prediction_indices = predictions["test_index"].to_numpy()

if not np.array_equal(
    expected_indices,
    prediction_indices
):
    raise RuntimeError(
        "test_index does NOT match metadata row order."
    )

print("PASS: test_index matches test_metadata.csv row order.")


# ============================================================
# MERGE METADATA + PREDICTIONS
# ============================================================

df = metadata.copy()

df.insert(
    0,
    "test_index",
    np.arange(len(df))
)

prediction_columns = [
    "predicted",
    "predicted_name",
    "confidence",
    "correct",
]

for col in prediction_columns:

    if col not in predictions.columns:
        raise RuntimeError(
            f"Missing prediction column: {col}"
        )

    df[col] = predictions[col].values


# Actual label from metadata
df["actual"] = df["patch_class"].astype(int)

df["actual_name"] = df["actual"].map(
    CLASS_NAMES
)


# ============================================================
# VERIFY ACTUAL LABELS
# ============================================================

if "actual" in predictions.columns:

    if not np.array_equal(
        df["actual"].to_numpy(),
        predictions["actual"].to_numpy()
    ):
        raise RuntimeError(
            "Prediction actual labels do not match "
            "test_metadata.csv patch_class."
        )

print("PASS: actual labels also match.")


# ============================================================
# NORMALIZE NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "px_no_damage",
    "px_damaged",
    "px_destroyed",
    "total_building_pixels",
    "building_coverage_pct",
    "num_target_buildings",
]

for col in numeric_columns:

    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# DERIVED FEATURES
# ============================================================

df["damaged_fraction_of_buildings"] = np.where(
    df["total_building_pixels"] > 0,
    df["px_damaged"] / df["total_building_pixels"],
    0.0,
)

df["destroyed_fraction_of_buildings"] = np.where(
    df["total_building_pixels"] > 0,
    df["px_destroyed"] / df["total_building_pixels"],
    0.0,
)

df["changed_building_fraction"] = np.where(
    df["total_building_pixels"] > 0,
    (
        df["px_damaged"] +
        df["px_destroyed"]
    ) / df["total_building_pixels"],
    0.0,
)


# ============================================================
# ERROR TYPE
# ============================================================

def get_error_type(row):

    actual = int(row["actual"])
    predicted = int(row["predicted"])

    if actual == predicted:
        return "Correct"

    mapping = {
        (2, 0): "Destroyed -> No Damage",
        (2, 1): "Destroyed -> Damaged",
        (1, 0): "Damaged -> No Damage",
        (0, 1): "No Damage -> Damaged",
    }

    return mapping.get(
        (actual, predicted),
        f"{CLASS_NAMES[actual]} -> {CLASS_NAMES[predicted]}"
    )


df["error_type"] = df.apply(
    get_error_type,
    axis=1
)


# ============================================================
# BASIC COUNTS
# ============================================================

print("\n" + "=" * 70)
print("ERROR COUNTS")
print("=" * 70)

print(
    df["error_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# ANALYZE FOUR IMPORTANT ERROR GROUPS
# ============================================================

TARGET_ERRORS = [
    "Destroyed -> No Damage",
    "Destroyed -> Damaged",
    "Damaged -> No Damage",
    "No Damage -> Damaged",
]


analysis_columns = [
    "building_coverage_pct",
    "total_building_pixels",
    "px_no_damage",
    "px_damaged",
    "px_destroyed",
    "damaged_fraction_of_buildings",
    "destroyed_fraction_of_buildings",
    "changed_building_fraction",
    "num_target_buildings",
    "confidence",
]


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = []


groups = {
    "Correct": df[df["error_type"] == "Correct"]
}

for error_type in TARGET_ERRORS:
    groups[error_type] = df[
        df["error_type"] == error_type
    ]


for group_name, group in groups.items():

    row = {
        "group": group_name,
        "count": len(group),
    }

    for col in analysis_columns:

        if col not in group.columns:
            continue

        values = group[col].dropna()

        if len(values) == 0:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_median"] = np.nan
            row[f"{col}_min"] = np.nan
            row[f"{col}_max"] = np.nan

        else:
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_median"] = values.median()
            row[f"{col}_min"] = values.min()
            row[f"{col}_max"] = values.max()

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)


# ============================================================
# PRINT IMPORTANT COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("BUILDING COVERAGE COMPARISON")
print("=" * 70)

coverage_cols = [
    "group",
    "count",
    "building_coverage_pct_mean",
    "building_coverage_pct_median",
    "building_coverage_pct_min",
    "building_coverage_pct_max",
]

print(
    summary[
        [
            c
            for c in coverage_cols
            if c in summary.columns
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("BUILDING PIXEL COMPARISON")
print("=" * 70)

pixel_cols = [
    "group",
    "count",
    "total_building_pixels_mean",
    "px_damaged_mean",
    "px_destroyed_mean",
    "changed_building_fraction_mean",
]

print(
    summary[
        [
            c
            for c in pixel_cols
            if c in summary.columns
        ]
    ].to_string(index=False)
)


# ============================================================
# EVENT-WISE ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EVENT-WISE ERROR COUNTS")
print("=" * 70)

event_errors = pd.crosstab(
    df["event"],
    df["error_type"]
)

print(
    event_errors.to_string()
)


# ============================================================
# ERROR RATE BY EVENT
# ============================================================

print("\n" + "=" * 70)
print("EVENT-WISE ERROR RATE")
print("=" * 70)

event_summary = (
    df.groupby("event")
    .agg(
        samples=("test_index", "count"),
        errors=("correct", lambda x: (~x).sum()),
        accuracy=("correct", "mean"),
    )
    .reset_index()
)

event_summary["error_rate"] = (
    1.0 - event_summary["accuracy"]
)

event_summary = event_summary.sort_values(
    "error_rate",
    ascending=False
)

print(
    event_summary.to_string(
        index=False
    )
)


# ============================================================
# LOW BUILDING COVERAGE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("LOW BUILDING COVERAGE ANALYSIS")
print("=" * 70)

coverage_thresholds = [
    1,
    2,
    5,
    10,
    15,
    20,
]

for threshold in coverage_thresholds:

    low = df[
        df["building_coverage_pct"] <= threshold
    ]

    if len(low) == 0:
        continue

    error_rate = (
        1.0 - low["correct"].mean()
    )

    print(
        f"Coverage <= {threshold:>2}% : "
        f"{len(low):>4} samples | "
        f"error rate = {error_rate:.2%}"
    )


# ============================================================
# LOW-COVERAGE ERROR BREAKDOWN
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT ERRORS WITH LOW BUILDING COVERAGE")
print("=" * 70)

low_coverage_errors = df[
    (df["building_coverage_pct"] <= 10)
    & (df["error_type"].isin(TARGET_ERRORS))
].copy()

print(
    low_coverage_errors[
        [
            "test_index",
            "event",
            "actual_name",
            "predicted_name",
            "confidence",
            "building_coverage_pct",
            "total_building_pixels",
            "px_damaged",
            "px_destroyed",
            "num_target_buildings",
            "is_cluster",
        ]
    ]
    .sort_values(
        "building_coverage_pct"
    )
    .head(100)
    .to_string(index=False)
)


# ============================================================
# SAVE COMPLETE AUDIT
# ============================================================

merged_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions_with_metadata.csv"
)

df.to_csv(
    merged_path,
    index=False
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "error_group_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


event_path = os.path.join(
    OUTPUT_DIR,
    "event_error_analysis.csv"
)

event_summary.to_csv(
    event_path,
    index=False
)


# ============================================================
# SAVE FOUR ERROR GROUPS
# ============================================================

for error_type in TARGET_ERRORS:

    safe_name = (
        error_type
        .lower()
        .replace(" -> ", "_to_")
        .replace(" ", "_")
    )

    path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}.csv"
    )

    df[
        df["error_type"] == error_type
    ].to_csv(
        path,
        index=False
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ERROR AUDIT COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nFiles:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    print("  ", filename)

print("=" * 70)