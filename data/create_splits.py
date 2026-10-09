import os
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# CONFIG & PATHS
# -------------------------------------------------------------
INPUT_CSV = r"D:\geo_spatial_fake_project\filtered_256_crops_metadata.csv"
TRAIN_CSV = r"D:\geo_spatial_fake_project\train_metadata.csv"
VAL_CSV = r"D:\geo_spatial_fake_project\val_metadata.csv"
TEST_CSV = r"D:\geo_spatial_fake_project\test_metadata.csv"
SUMMARY_CSV = r"D:\geo_spatial_fake_project\split_summary.csv"

RANDOM_SEED = 42
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# -------------------------------------------------------------
# 1. LOAD DATA & AGGREGATE TO SOURCE-SCENE LEVEL
# -------------------------------------------------------------
df_crops = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df_crops):,} crops from {INPUT_CSV}")

# Group by unique source scene (image_name) and event
scene_agg = df_crops.groupby(["image_name", "event"]).agg(
    num_crops=("crop_id", "count"),
    num_c0=("patch_class", lambda s: (s == 0).sum()),
    num_c1=("patch_class", lambda s: (s == 1).sum()),
    num_c2=("patch_class", lambda s: (s == 2).sum()),
).reset_index()

# Categorize each scene by its damage composition signature
# e.g., '0_only', '1_only', '2_only', '0_1', '1_2', '0_1_2', etc.
def get_signature(row):
    has_0 = row["num_c0"] > 0
    has_1 = row["num_c1"] > 0
    has_2 = row["num_c2"] > 0
    parts = []
    if has_0: parts.append("c0")
    if has_1: parts.append("c1")
    if has_2: parts.append("c2")
    return "_".join(parts) if parts else "empty"

scene_agg["damage_signature"] = scene_agg.apply(get_signature, axis=1)

print(f"Total Unique Source Scenes: {len(scene_agg):,}")

# -------------------------------------------------------------
# 2. STRATIFIED GROUP SPLITTING AT SCENE LEVEL
# -------------------------------------------------------------
# Set reproducible random seed
rng = np.random.RandomState(RANDOM_SEED)

scene_split_map = {}

# We stratify within each event and damage signature
# To balance both class distribution and event representation:
grouped = scene_agg.groupby(["event", "damage_signature"])

# We can also track running crop and scene totals for train, val, test
split_names = ["train", "val", "test"]

for (event, sig), group in grouped:
    # Shuffle scenes within this stratum
    shuffled_scenes = group.sample(frac=1.0, random_state=rng)["image_name"].tolist()
    n = len(shuffled_scenes)

    if n == 1:
        # If only 1 scene exists in this rare event/signature combination:
        # Default to train
        scene_split_map[shuffled_scenes[0]] = "train"
    elif n == 2:
        # If 2 scenes: 1 to train, 1 to val (or test based on overall balance)
        scene_split_map[shuffled_scenes[0]] = "train"
        scene_split_map[shuffled_scenes[1]] = "val"
    else:
        # Calculate target counts
        n_train = int(round(n * TARGET_RATIOS["train"]))
        n_val = int(round(n * TARGET_RATIOS["val"]))
        
        # Ensure at least 1 in val and test if n >= 6
        if n >= 6:
            n_val = max(1, n_val)
        
        n_test = n - n_train - n_val
        
        # Adjust if rounding produced 0 or negative
        if n_test < 0:
            n_train += n_test
            n_test = 0
            
        train_scenes = shuffled_scenes[:n_train]
        val_scenes = shuffled_scenes[n_train:n_train + n_val]
        test_scenes = shuffled_scenes[n_train + n_val:]

        for s in train_scenes:
            scene_split_map[s] = "train"
        for s in val_scenes:
            scene_split_map[s] = "val"
        for s in test_scenes:
            scene_split_map[s] = "test"

# Map split back to crops
df_crops["split"] = df_crops["image_name"].map(scene_split_map)

# -------------------------------------------------------------
# 3. VERIFY ZERO DATA LEAKAGE & INTEGRITY
# -------------------------------------------------------------
train_scenes_set = set(df_crops[df_crops["split"] == "train"]["image_name"])
val_scenes_set = set(df_crops[df_crops["split"] == "val"]["image_name"])
test_scenes_set = set(df_crops[df_crops["split"] == "test"]["image_name"])

overlap_tv = train_scenes_set.intersection(val_scenes_set)
overlap_tt = train_scenes_set.intersection(test_scenes_set)
overlap_vt = val_scenes_set.intersection(test_scenes_set)

assert len(overlap_tv) == 0, f"Leakage detected between train and val: {overlap_tv}"
assert len(overlap_tt) == 0, f"Leakage detected between train and test: {overlap_tt}"
assert len(overlap_vt) == 0, f"Leakage detected between val and test: {overlap_vt}"
assert df_crops["split"].isna().sum() == 0, "Unassigned crops found!"

print("\n" + "=" * 65)
print("ZERO DATA LEAKAGE VERIFIED: No scene overlap across splits!")
print("=" * 65)

# -------------------------------------------------------------
# 4. SAVE INDIVIDUAL SPLIT CSVs
# -------------------------------------------------------------
df_train = df_crops[df_crops["split"] == "train"].copy()
df_val = df_crops[df_crops["split"] == "val"].copy()
df_test = df_crops[df_crops["split"] == "test"].copy()

df_train.to_csv(TRAIN_CSV, index=False)
df_val.to_csv(VAL_CSV, index=False)
df_test.to_csv(TEST_CSV, index=False)

print(f"Saved: {TRAIN_CSV} ({len(df_train):,} crops, {len(train_scenes_set):,} scenes)")
print(f"Saved: {VAL_CSV} ({len(df_val):,} crops, {len(val_scenes_set):,} scenes)")
print(f"Saved: {TEST_CSV} ({len(df_test):,} crops, {len(test_scenes_set):,} scenes)")

# -------------------------------------------------------------
# 5. GENERATE & SAVE SPLIT_SUMMARY.CSV
# -------------------------------------------------------------
total_crops = len(df_crops)
total_scenes = len(scene_agg)

summary_rows = []

# Overall Metrics
for s_name, d, sc_set in [("Train", df_train, train_scenes_set), 
                          ("Val", df_val, val_scenes_set), 
                          ("Test", df_test, test_scenes_set)]:
    summary_rows.append({
        "category": "Overall",
        "split": s_name,
        "scenes": len(sc_set),
        "scenes_pct": round(len(sc_set) / total_scenes * 100, 2),
        "total_crops": len(d),
        "crops_pct": round(len(d) / total_crops * 100, 2),
        "class_0_crops": int((d["patch_class"] == 0).sum()),
        "class_0_pct": round((d["patch_class"] == 0).sum() / len(d) * 100, 2),
        "class_1_crops": int((d["patch_class"] == 1).sum()),
        "class_1_pct": round((d["patch_class"] == 1).sum() / len(d) * 100, 2),
        "class_2_crops": int((d["patch_class"] == 2).sum()),
        "class_2_pct": round((d["patch_class"] == 2).sum() / len(d) * 100, 2),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(SUMMARY_CSV, index=False)
print(f"Saved: {SUMMARY_CSV}")

# -------------------------------------------------------------
# 6. PRINT SUMMARY AUDIT TABLES TO TERMINAL
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("SPLIT DISTRIBUTION OVERVIEW")
print("=" * 70)
print(f"{'Split':<8} {'Scenes':>9} {'Scenes %':>10} {'Crops':>9} {'Crops %':>10} {'Class 0 (No)':>14} {'Class 1 (Dam)':>14} {'Class 2 (Dest)':>15}")
print("-" * 92)

for _, r in df_summary.iterrows():
    c0_str = f"{r['class_0_crops']:,} ({r['class_0_pct']}%)"
    c1_str = f"{r['class_1_crops']:,} ({r['class_1_pct']}%)"
    c2_str = f"{r['class_2_crops']:,} ({r['class_2_pct']}%)"
    print(f"{r['split']:<8} {r['scenes']:>9,} {r['scenes_pct']:>9.1f}% {r['total_crops']:>9,} {r['crops_pct']:>9.1f}% {c0_str:>14} {c1_str:>14} {c2_str:>15}")

print("-" * 92)
print(f"{'Total':<8} {total_scenes:>9,} {100.0:>9.1f}% {total_crops:>9,} {100.0:>9.1f}% "
      f"{(df_crops['patch_class']==0).sum():,} ({((df_crops['patch_class']==0).sum()/total_crops*100):.1f}%) "
      f"{(df_crops['patch_class']==1).sum():,} ({((df_crops['patch_class']==1).sum()/total_crops*100):.1f}%) "
      f"{(df_crops['patch_class']==2).sum():,} ({((df_crops['patch_class']==2).sum()/total_crops*100):.1f}%)")
print("=" * 92)

print("\nEVENT-WISE SPLIT BREAKDOWN (Crops per event in Train / Val / Test):")
print("-" * 75)
event_cross = pd.crosstab(df_crops["event"], df_crops["split"], margins=True)
# Reorder columns
event_cross = event_cross[["train", "val", "test", "All"]]
print(event_cross.to_string())
print("=" * 75)

