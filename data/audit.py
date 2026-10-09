import pandas as pd
import numpy as np

# -------------------------------------------------------------
# CONFIG & PATHS
# -------------------------------------------------------------
INPUT_CSV = r"D:\geo_spatial_fake_project\fixed_256_crops_metadata.csv"
OUTPUT_FILTERED_CSV = r"D:\geo_spatial_fake_project\filtered_256_crops_metadata.csv"

# Total pixels in a 256x256 patch
TOTAL_PATCH_PIXELS = 256 * 256  # 65,536 pixels

# Minimum building coverage percentage threshold:
# 3.0% = ~1,966 pixels (approx. 2-3 decent sized houses or 1 large building)
# You can adjust this threshold if you want strictly >= 5% (3,277 pixels)
MIN_BUILDING_PCT_THRESHOLD = 3.0
MIN_BUILDING_PIXELS = int((MIN_BUILDING_PCT_THRESHOLD / 100.0) * TOTAL_PATCH_PIXELS)

# -------------------------------------------------------------
# LOAD & FILTER
# -------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
initial_count = len(df)
print(f"Loaded {initial_count:,} crops from {INPUT_CSV}")

# Calculate building coverage percentage for each crop
df["building_coverage_pct"] = (df["total_building_pixels"] / TOTAL_PATCH_PIXELS) * 100.0

# Apply the minimum building presence filter
df_filtered = df[df["total_building_pixels"] >= MIN_BUILDING_PIXELS].copy()
filtered_count = len(df_filtered)
dropped_count = initial_count - filtered_count

# -------------------------------------------------------------
# SAVE FILTERED CSV
# -------------------------------------------------------------
df_filtered.to_csv(OUTPUT_FILTERED_CSV, index=False)

# -------------------------------------------------------------
# DETAILED AUDIT & COMPARISON REPORT
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("BUILDING PRESENCE FILTER REPORT")
print("=" * 65)
print(f"Total patch pixels      : 65,536 (256x256)")
print(f"Threshold applied       : >= {MIN_BUILDING_PCT_THRESHOLD:.1f}% ({MIN_BUILDING_PIXELS:,} building pixels)")
print(f"Original total crops    : {initial_count:,}")
print(f"Crops kept              : {filtered_count:,} ({(filtered_count / initial_count) * 100:.1f}%)")
print(f"Crops dropped (< {MIN_BUILDING_PCT_THRESHOLD:.1f}%): {dropped_count:,} ({(dropped_count / initial_count) * 100:.1f}%)")

print("\n" + "-" * 65)
print(f"{'Class':<25} {'Before':>10} {'After':>10} {'Retention':>12}")
print("-" * 65)

classes = sorted(df["patch_label"].unique())
for cls in classes:
    before = (df["patch_label"] == cls).sum()
    after = (df_filtered["patch_label"] == cls).sum()
    retention = (after / before * 100.0) if before > 0 else 0
    print(f"{cls:<25} {before:>10,} {after:>10,} {retention:>11.1f}%")

print("-" * 65)
print(f"{'TOTAL':<25} {initial_count:>10,} {filtered_count:>10,} {(filtered_count / initial_count) * 100:>11.1f}%")
print("=" * 65)

print("\nBuilding Coverage Statistics (Filtered Dataset):")
print(f"  Min building coverage : {df_filtered['building_coverage_pct'].min():.2f}% ({df_filtered['total_building_pixels'].min():,} px)")
print(f"  Median coverage       : {df_filtered['building_coverage_pct'].median():.2f}% ({int(df_filtered['total_building_pixels'].median()):,} px)")
print(f"  Mean coverage         : {df_filtered['building_coverage_pct'].mean():.2f}% ({int(df_filtered['total_building_pixels'].mean()):,} px)")
print(f"  Max building coverage : {df_filtered['building_coverage_pct'].max():.2f}% ({df_filtered['total_building_pixels'].max():,} px)")

print(f"\nSaved filtered dataset to: {OUTPUT_FILTERED_CSV}")