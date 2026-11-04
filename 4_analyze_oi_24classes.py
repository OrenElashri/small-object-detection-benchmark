import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ========================= CONFIG =========================
BASE_DIR = "/p/composml/object-detection-benchmark"

# --- Open Images V7 (24 classes) ---
YOLO_ROOT = os.path.join(BASE_DIR, "openimages_v7_24classes")
TRAIN_LABELS = os.path.join(YOLO_ROOT, "labels/train")
TRAIN_IMGS = os.path.join(YOLO_ROOT, "images/train")

# --- COCO 2017 ---
COCO_ANN_TRAIN = os.path.join(BASE_DIR, "coco_2017/annotations/instances_train2017.json")

# --- Output directories ---
OI_OUTPUT_DIR = "/home/elashrio/bachelor_thesis/bachelor_thesis_small_objects/oi_24classes_analysis/oi_24classes_analysis"
COCO_OUTPUT_DIR = "/home/elashrio/bachelor_thesis/bachelor_thesis_small_objects/oi_24classes_analysis/coco_analysis_compare"

os.makedirs(OI_OUTPUT_DIR, exist_ok=True)
os.makedirs(COCO_OUTPUT_DIR, exist_ok=True)

small_area_threshold = 32 * 32

# ====================== 24 SHARED CLASSES ======================
MATCHED_CLASSES = {
    'apple', 'backpack', 'baseball glove', 'bird', 'boat', 'book',
    'bottle', 'car', 'clock', 'handbag', 'kite', 'knife', 'mouse',
    'orange', 'parking meter', 'person', 'sheep', 'snowboard',
    'spoon', 'tie', 'toothbrush', 'traffic light', 'vase', 'wine glass'
}

COCO_TO_OUR = {
    "baseball glove": "baseball glove",
    "wine glass": "wine glass",
    "traffic light": "traffic light",
    "parking meter": "parking meter",
    "mouse": "mouse",
}
# ===========================================================

def print_summary(df, small_areas, dataset_name):
    max_small = df['num_small_objects'].max()
    
    print(f"\n{'='*75}")
    print(f" DATASET: {dataset_name}")
    print(f"{'='*75}")
    print(f"Total images processed : {len(df):,}")
    has_small = (df['num_small_objects'] > 0).sum()
    print(f"Images with ≥1 small obj: {has_small:,} ({100 * has_small / len(df):.2f}%)")
    print(f"Max small objects in one image : **{max_small:,}**")

    print("\n--- IMAGE SIZE STATISTICS ---")
    print(df[['width', 'height']].describe().round(2))

    total_objs = df['num_objects'].sum()
    total_small = df['num_small_objects'].sum()
    print("\n--- SMALL OBJECT STATISTICS ---")
    print(f"Avg objects per image      : {df['num_objects'].mean():.3f}")
    print(f"Avg small objects per image: {df['num_small_objects'].mean():.3f}")
    print(f"Percentage of small objects: {100 * total_small / total_objs:.2f}%")

    small_df = df[df['num_small_objects'] > 0]
    print("\n--- SMALL OBJECT PIXEL OCCUPATION ---")
    print(f"Avg small object area      : {np.mean(small_areas):.2f} px²")
    print(f"Median small object area   : {np.median(small_areas):.2f} px²")
    print(f"Avg small area ratio       : {small_df['small_area_ratio'].mean()*100:.3f}%")
    print(f"Median small area ratio    : {small_df['small_area_ratio'].median()*100:.3f}%")


def save_plots(df, small_areas, output_dir, dataset_name):
    small_df = df[df['num_small_objects'] > 0]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['num_small_objects'], bins=50)
    plt.title(f'[{dataset_name}] Distribution of Small Objects per Image')
    plt.xlabel('Number of Small Objects')
    plt.ylabel('Number of Images')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "01_small_objects_per_image.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(small_df['small_area_ratio'] * 100, bins=50)
    plt.title(f'[{dataset_name}] Small Objects – Pixel Occupation (%)')
    plt.xlabel('Percentage of Image Area Occupied by Small Objects')
    plt.ylabel('Number of Images')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "02_small_objects_pixel_occupation.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ===========================================================
# 1. OPEN IMAGES V7 Train (24 classes)
# ===========================================================
print("\n" + "="*60)
print("🔍 Analysing Open Images V7 Train (24 classes)…")
print("="*60)

label_files = [f for f in os.listdir(TRAIN_LABELS) if f.endswith('.txt')]
oi_stats = []
oi_small_areas = []

batch_size = 4000
for i in tqdm(range(0, len(label_files), batch_size), desc="OI Train Processing"):
    batch_files = label_files[i:i + batch_size]
    for label_file in batch_files:
        img_id = label_file.replace('.txt', '')
        img_path = os.path.join(TRAIN_IMGS, f"{img_id}.jpg")
        
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            continue

        with open(os.path.join(TRAIN_LABELS, label_file)) as f:
            lines = f.readlines()

        num_objs = len(lines)
        small_objs = 0
        small_pixel_area = 0.0
        for line in lines:
            if not line.strip(): continue
            _, x, y, bw, bh = map(float, line.strip().split())
            pixel_w = bw * w
            pixel_h = bh * h
            area = pixel_w * pixel_h
            if area < small_area_threshold:
                small_objs += 1
                small_pixel_area += area
                oi_small_areas.append(area)

        oi_stats.append({
            'image_id': img_id,
            'width': w,
            'height': h,
            'num_objects': num_objs,
            'num_small_objects': small_objs,
            'small_area_ratio': small_pixel_area / (w * h) if (w * h) > 0 else 0.0,
        })

oi_df = pd.DataFrame(oi_stats)
print_summary(oi_df, oi_small_areas, "Open Images V7 Train (24 classes)")
oi_df.to_csv(os.path.join(OI_OUTPUT_DIR, "oi_24classes_train_stats.csv"), index=False)

# ===========================================================
# 2. COCO 2017 Train (24 classes only)
# ===========================================================
print("\n" + "="*60)
print("🔍 Analysing COCO 2017 Train (24 classes)…")
print("="*60)

def process_coco_train(ann_path):
    with open(ann_path) as f:
        coco = json.load(f)

    img_info = {img['id']: (img['width'], img['height']) for img in coco['images']}
    cat_id_to_name = {cat['id']: cat['name'].lower() for cat in coco['categories']}

    img_stats = defaultdict(lambda: {'num_objects': 0, 'num_small_objects': 0, 'small_pixel_area': 0.0})
    small_areas = []
    matched_count = 0

    for ann in tqdm(coco['annotations'], desc="COCO Train (24 classes)"):
        img_id = ann['image_id']
        if img_id not in img_info:
            continue

        cat_name = cat_id_to_name.get(ann['category_id'], '')
        our_class = COCO_TO_OUR.get(cat_name, cat_name)

        if our_class not in MATCHED_CLASSES:
            continue

        matched_count += 1
        _, _, bw, bh = ann['bbox']
        area = bw * bh

        img_stats[img_id]['num_objects'] += 1
        if area < small_area_threshold:
            img_stats[img_id]['num_small_objects'] += 1
            img_stats[img_id]['small_pixel_area'] += area
            small_areas.append(area)

    print(f"   → Kept {matched_count:,} annotations from the 24 classes")

    rows = []
    for img_id, s in img_stats.items():
        w, h = img_info[img_id]
        rows.append({
            'image_id': img_id,
            'width': w,
            'height': h,
            'num_objects': s['num_objects'],
            'num_small_objects': s['num_small_objects'],
            'small_area_ratio': s['small_pixel_area'] / (w * h) if (w * h) > 0 else 0.0,
        })

    return pd.DataFrame(rows), small_areas


coco_train_df, coco_small_areas = process_coco_train(COCO_ANN_TRAIN)
print_summary(coco_train_df, coco_small_areas, "COCO 2017 Train (24 classes)")
coco_train_df.to_csv(os.path.join(COCO_OUTPUT_DIR, "coco_2017_train_24classes_stats.csv"), index=False)

# ===========================================================
# 3. Reduced Comparison Plot (Top-Left + Bottom-Left style)
# ===========================================================
print("\n📊 Generating comparison plot...")

fig, axes = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle("Comparison: Open Images V7 Train vs COCO 2017 Train\n(24 shared classes)", 
             fontsize=16, fontweight='bold')

datasets = [
    ("Open Images V7", oi_df, oi_small_areas, "steelblue"),
    ("COCO 2017 Train", coco_train_df, coco_small_areas, "darkorange"),
]

ax = axes[0]
for label, df, _, color in datasets:
    sns.kdeplot(df['num_small_objects'], ax=ax, label=label, color=color, fill=True, alpha=0.35)
ax.set_title("Small Objects per Image")
ax.set_xlabel("Number of Small Objects")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for label, _, areas, color in datasets:
    sns.kdeplot(areas, ax=ax, label=label, color=color, fill=True, alpha=0.35)
ax.set_title("Small Object Area Distribution (px²)")
ax.set_xlabel("Area (px²)")
ax.set_ylabel("Density")
ax.axvline(small_area_threshold, color='red', linestyle='--', label=f'Threshold ({small_area_threshold} px²)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OI_OUTPUT_DIR, "05_comparison_train_24classes.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(COCO_OUTPUT_DIR, "05_comparison_train_24classes.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("🔍 FINAL VERIFICATION - ONLY 24 CLASSES?")
print("="*80)

print(f"Open Images - Total objects counted : {oi_df['num_objects'].sum():,}  "
      f"({oi_df['num_objects'].sum()/len(oi_df):.2f} per image)")

print(f"COCO 24cls   - Total objects counted : {coco_df['num_objects'].sum():,}  "
      f"({coco_df['num_objects'].sum()/len(coco_df):.2f} per image)")

# Check how many images have zero objects (should be none)
print(f"Images with 0 objects in OI   : {(oi_df['num_objects'] == 0).sum()}")
print(f"Images with 0 objects in COCO : {(coco_df['num_objects'] == 0).sum()}")

print("✅ All done! Comparison saved.")
