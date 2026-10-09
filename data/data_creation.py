import glob
import os
import re
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import find_objects, label
from sklearn.cluster import DBSCAN

# Silence loky warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# =====================================================================
# CONFIGURATION & HYPERPARAMETERS
# =====================================================================
DATA_DIR = r"D:\geo_spatial_fake_project\xview2\data"
OUTPUT_CROPS_CSV = r"D:\geo_spatial_fake_project\fixed_256_crops_metadata.csv"
PREVIEW_DIR = r"D:\geo_spatial_fake_project\crops_256_preview"
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Fixed patch dimension
CROP_SIZE = 256
IMAGE_SIZE = 1024

# Spatial clustering hyperparameters
CLUSTER_EPS_PIXELS = 100       # Max distance in pixels to group buildings
MIN_CLUSTER_SAMPLES = 2        # >=2 forms a cluster, 1 is isolated building

MAX_CONTROL_SAMPLES = 500      # Max no-damage control images to process
NUM_PREVIEWS_PER_CLASS = 3     # Preview images per class to save


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def extract_building_footprints(mask, target_classes):
    """
    Extracts individual connected component building masks matching target classes.
    """
    binary_mask = np.isin(mask, target_classes).astype(np.uint8)
    labeled_mask, _ = label(binary_mask)
    slices = find_objects(labeled_mask)

    buildings = []
    for i, slc in enumerate(slices, 1):
        component_area = np.sum(labeled_mask[slc] == i)
        if component_area < 15:  # Filter noise artifacts
            continue

        ymin, ymax = slc[0].start, slc[0].stop
        xmin, xmax = slc[1].start, slc[1].stop
        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0

        classes = np.unique(mask[slc][labeled_mask[slc] == i])
        classes = [int(c) for c in classes if c > 0]

        buildings.append({
            "id": i,
            "center": (cy, cx),
            "ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax,
            "classes": classes,
            "area": component_area,
        })
    return buildings


def compute_fixed_256_box(buildings_group, crop_size=256, img_size=1024):
    """
    Calculates a fixed crop_size x crop_size (256x256) bounding box centered
    around the building cluster, strictly clamped inside [0, img_size].
    """
    # Cluster center of mass
    center_y = int(np.mean([b["center"][0] for b in buildings_group]))
    center_x = int(np.mean([b["center"][1] for b in buildings_group]))

    # Center the 256x256 window
    half_size = crop_size // 2
    ymin = center_y - half_size
    xmin = center_x - half_size
    ymax = ymin + crop_size
    xmax = xmin + crop_size

    # Clamp to image boundaries
    if ymin < 0:
        ymin = 0
        ymax = crop_size
    elif ymax > img_size:
        ymax = img_size
        ymin = img_size - crop_size

    if xmin < 0:
        xmin = 0
        xmax = crop_size
    elif xmax > img_size:
        xmax = img_size
        xmin = img_size - crop_size

    return ymin, xmin, ymax, xmax


# =====================================================================
# STEP 1: SCAN DATASET AND CATEGORIZE IMAGES
# =====================================================================
files = sorted(glob.glob(DATA_DIR + r"\*.parquet"))
print(f"Found {len(files)} parquet shards in {DATA_DIR}")

damage_sample_pool = []
no_damage_sample_pool = []

print("Scanning samples for damage vs clean scenes...")
for shard_idx, path in enumerate(files, 1):
    df_shard = pd.read_parquet(path, columns=["image_name", "t2_mask"])

    for row_idx, row in df_shard.iterrows():
        img_name = row["image_name"]
        match = re.search(r"images/([^_]+(?:-[^_]+)*)_", img_name)
        event = match.group(1) if match else "unknown"

        t2_mask = np.array(Image.open(BytesIO(row["t2_mask"]["bytes"])))
        unique_vals = set(np.unique(t2_mask))

        has_no_damage = 1 in unique_vals
        has_damaged = (2 in unique_vals) or (3 in unique_vals)
        has_destroyed = 4 in unique_vals

        if not (has_no_damage or has_damaged or has_destroyed):
            continue  # No buildings

        item = {
            "image_name": img_name,
            "event": event,
            "shard_file": path,
            "shard_row_idx": row_idx,
        }

        # Damage target: has damaged or destroyed buildings
        if has_damaged or has_destroyed:
            item["is_control"] = False
            damage_sample_pool.append(item)
        # Control scene: has buildings, but ZERO damage
        elif has_no_damage and not has_damaged and not has_destroyed:
            item["is_control"] = True
            no_damage_sample_pool.append(item)

# Subsample no-damage control scenes
df_ctrl = pd.DataFrame(no_damage_sample_pool)
if not df_ctrl.empty:
    n_sample = min(MAX_CONTROL_SAMPLES, len(df_ctrl))
    sampled_ctrl = df_ctrl.sample(n=n_sample, random_state=42).to_dict("records")
else:
    sampled_ctrl = []

all_samples_to_crop = damage_sample_pool + sampled_ctrl
print(f"Total scenes selected: {len(all_samples_to_crop)} "
      f"(Damage: {len(damage_sample_pool)}, Clean Control: {len(sampled_ctrl)})\n")


# =====================================================================
# STEP 2: GENERATE FIXED 256x256 CONTEXTUAL CROPS
# =====================================================================
crops_metadata = []
df_to_crop = pd.DataFrame(all_samples_to_crop)
grouped_shards = df_to_crop.groupby("shard_file")

print("Generating fixed 256x256 contextual crops...")

for shard_path, group in grouped_shards:
    shard_df = pd.read_parquet(shard_path)

    for _, meta_row in group.iterrows():
        row_data = shard_df.iloc[meta_row["shard_row_idx"]]
        t2_mask = np.array(Image.open(BytesIO(row_data["t2_mask"]["bytes"])))

        is_control = meta_row["is_control"]

        # If control scene -> extract clusters of intact buildings (class 1)
        # If damage scene  -> extract clusters of damaged/destroyed buildings (classes 2, 3, 4)
        target_classes = [1] if is_control else [2, 3, 4]
        buildings = extract_building_footprints(t2_mask, target_classes)

        if not buildings:
            continue

        coords = np.array([b["center"] for b in buildings])

        # Cluster spatial groups
        if len(buildings) == 1:
            cluster_labels = np.array([-1])
        else:
            db = DBSCAN(eps=CLUSTER_EPS_PIXELS, min_samples=MIN_CLUSTER_SAMPLES, n_jobs=1).fit(coords)
            cluster_labels = db.labels_

        clusters = {}
        for b, label_id in zip(buildings, cluster_labels):
            clusters.setdefault(label_id, []).append(b)

        crop_idx = 0
        for cluster_id, b_list in clusters.items():
            crop_idx += 1
            # Compute exact fixed 256x256 box centered on cluster
            ymin, xmin, ymax, xmax = compute_fixed_256_box(b_list, crop_size=CROP_SIZE, img_size=IMAGE_SIZE)

            # Analyze the actual pixels inside this 256x256 patch
            patch_mask = t2_mask[ymin:ymax, xmin:xmax]
            px_no_damage = np.sum(patch_mask == 1)
            px_damaged = np.sum((patch_mask == 2) | (patch_mask == 3))
            px_destroyed = np.sum(patch_mask == 4)
            total_building_px = px_no_damage + px_damaged + px_destroyed

            # Ensure the 256x256 patch contains buildings
            if total_building_px < 50:
                continue

            # Assign 3-Class Label based on our project definition
            if is_control or (px_damaged == 0 and px_destroyed == 0):
                patch_class = 0
                patch_label = "Class 0: No Damage"
            elif px_destroyed > px_damaged:
                patch_class = 2
                patch_label = "Class 2: Destroyed"
            else:
                patch_class = 1
                patch_label = "Class 1: Damaged"

            crops_metadata.append({
                "crop_id": f"{meta_row['image_name']}_crop256_{crop_idx}",
                "image_name": meta_row["image_name"],
                "event": meta_row["event"],
                "shard_file": shard_path,
                "shard_row_idx": meta_row["shard_row_idx"],
                "patch_class": patch_class,
                "patch_label": patch_label,
                "is_cluster": int(len(b_list) >= 2),
                "num_target_buildings": len(b_list),
                # Identical bounding box for T1 and T2:
                "crop_ymin": ymin,
                "crop_xmin": xmin,
                "crop_ymax": ymax,
                "crop_xmax": xmax,
                "crop_height": ymax - ymin,
                "crop_width": xmax - xmin,
                "px_no_damage": int(px_no_damage),
                "px_damaged": int(px_damaged),
                "px_destroyed": int(px_destroyed),
                "total_building_pixels": int(total_building_px),
            })

# Save crops metadata
df_crops = pd.DataFrame(crops_metadata)
df_crops.to_csv(OUTPUT_CROPS_CSV, index=False)

print("\n" + "=" * 65)
print(f"SUCCESS: Saved fixed 256x256 crops metadata to: {OUTPUT_CROPS_CSV}")
print("=" * 65)
print(f"Total 256x256 crops generated: {len(df_crops)}")
print("\nClass Distribution:")
print(df_crops["patch_label"].value_counts().to_string())
print(f"\nCluster vs Isolated: {df_crops['is_cluster'].sum()} clusters, "
      f"{len(df_crops) - df_crops['is_cluster'].sum()} isolated")


# =====================================================================
# STEP 3: PREVIEW VERIFICATION ACROSS THE 3 CLASSES
# =====================================================================
print(f"\nGenerating preview images in {PREVIEW_DIR}...")

DAMAGE_COLORS = [
    (0.0, 0.0, 0.0, 0.0),     # 0: Background
    (0.0, 0.8, 0.2, 0.85),    # 1: No Damage (Green)
    (1.0, 0.65, 0.0, 0.85),   # 2: Damaged (Orange)
    (0.9, 0.1, 0.1, 0.85),    # 3: Destroyed (Red)
]
from matplotlib.colors import ListedColormap
cmap_damage = ListedColormap(DAMAGE_COLORS)

preview_samples = []
for c in [0, 1, 2]:
    subset = df_crops[df_crops["patch_class"] == c]
    if not subset.empty:
        preview_samples.append(subset.sample(n=min(len(subset), NUM_PREVIEWS_PER_CLASS), random_state=42))

df_preview = pd.concat(preview_samples, ignore_index=True)

for idx, crop in df_preview.iterrows():
    shard_df = pd.read_parquet(crop["shard_file"])
    row = shard_df.iloc[crop["shard_row_idx"]]

    t1_full = np.array(Image.open(BytesIO(row["t1_image"]["bytes"])))
    t2_full = np.array(Image.open(BytesIO(row["t2_image"]["bytes"])))
    t2_mask_full = np.array(Image.open(BytesIO(row["t2_mask"]["bytes"])))

    ymin, xmin, ymax, xmax = crop["crop_ymin"], crop["crop_xmin"], crop["crop_ymax"], crop["crop_xmax"]

    # Identical 256x256 slice
    t1_crop = t1_full[ymin:ymax, xmin:xmax]
    t2_crop = t2_full[ymin:ymax, xmin:xmax]
    mask_crop = t2_mask_full[ymin:ymax, xmin:xmax]

    # Map mask: 1->1, (2,3)->2, 4->3
    mapped_mask = np.zeros_like(mask_crop, dtype=np.uint8)
    mapped_mask[mask_crop == 1] = 1
    mapped_mask[(mask_crop == 2) | (mask_crop == 3)] = 2
    mapped_mask[mask_crop == 4] = 3

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.suptitle(
        f"{crop['patch_label']} | Buildings in cluster: {crop['num_target_buildings']} | Event: {crop['event']}\n"
        f"256x256 Box: [Y: {ymin}:{ymax}, X: {xmin}:{xmax}]",
        fontsize=11, fontweight="bold"
    )

    axes[0].imshow(t1_crop)
    axes[0].set_title(f"Before (T1 256x256)")
    axes[0].axis("off")

    axes[1].imshow(t2_crop)
    axes[1].set_title(f"After (T2 256x256)")
    axes[1].axis("off")

    axes[2].imshow(t2_crop)
    overlay = np.ma.masked_where(mapped_mask == 0, mapped_mask)
    axes[2].imshow(overlay, cmap=cmap_damage, vmin=0, vmax=3, interpolation="nearest")
    axes[2].set_title("T2 + Damage Mask Overlay\n(Green:Intact, Orange:Damaged, Red:Destroyed)", fontsize=9)
    axes[2].axis("off")

    plt.tight_layout()
    out_file = os.path.join(PREVIEW_DIR, f"preview_{idx+1:02d}_class{crop['patch_class']}.png")
    plt.savefig(out_file, dpi=130)
    plt.close()

print(f"All previews saved to: {PREVIEW_DIR}")
print("Done!")