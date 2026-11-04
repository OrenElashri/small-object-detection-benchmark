import os
import pandas as pd
from tqdm import tqdm
import shutil
from pathlib import Path

# ========================= CONFIG =========================
BASE_DIR = "/p/composml/object-detection-benchmark"
YOLO_DATA_ROOT = os.path.join(BASE_DIR, "openimages_v7_24classes")

MATCHED_CLASSES = {
    'apple', 'backpack', 'baseball glove', 'bird', 'boat', 'book',
    'bottle', 'car', 'clock', 'handbag', 'kite', 'knife', 'mouse',
    'orange', 'parking meter', 'person', 'sheep', 'snowboard',
    'spoon', 'tie', 'toothbrush', 'traffic light', 'vase', 'wine glass'
}

# Special mapping for Open Images internal names
OPENIMAGES_CLASS_MAPPING = {
    "mouse": "Computer mouse"
}
# =========================================================

def prepare_oi_yolo_split(oi_base: str, split: str, yolo_root: str):
    data_dir = os.path.join(oi_base, "data")
    labels_csv = os.path.join(oi_base, "labels", "detections.csv")
    classes_csv = os.path.join(oi_base, "metadata", "classes.csv")

    labels_dir = os.path.join(yolo_root, "labels", split)
    images_dir = os.path.join(yolo_root, "images", split)

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Consistent class ordering (important for YOLO)
    classes_sorted = sorted(MATCHED_CLASSES)
    class_to_id = {name: idx for idx, name in enumerate(classes_sorted)}

    # Load Open Images class mapping
    classes_df = pd.read_csv(classes_csv, header=None, names=["LabelName", "DisplayName"])
    class_map = dict(zip(classes_df["DisplayName"].str.lower(), classes_df["LabelName"]))

    # Build MID -> our class mapping
    matched_mids = {}
    for cls in classes_sorted:
        oi_name = OPENIMAGES_CLASS_MAPPING.get(cls.lower(), cls.lower()).lower()
        if oi_name in class_map:
            mid = class_map[oi_name]
            matched_mids[mid] = cls
        else:
            print(f"Warning: Class '{cls}' not found in Open Images metadata.")

    mid_to_id = {mid: class_to_id[cls] for mid, cls in matched_mids.items()}

    print(f"Preparing {split} split with {len(matched_mids)} matched classes...")

    # Load detections (very large file → we filter early)
    dets_df = pd.read_csv(labels_csv, usecols=["ImageID", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax"])
    dets_df = dets_df[dets_df["Confidence"] == 1]
    dets_df = dets_df[dets_df["LabelName"].isin(matched_mids.keys())]

    grouped = dets_df.groupby("ImageID")
    print(f"Found {len(grouped)} images with at least one of the 24 target classes.")

    for img_id, group in tqdm(grouped, desc=f"Processing {split}", total=len(grouped)):
        label_path = os.path.join(labels_dir, f"{img_id}.txt")

        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                mid = row["LabelName"]
                if mid not in mid_to_id:
                    continue

                cls_id = mid_to_id[mid]
                xmin, xmax, ymin, ymax = row[["XMin", "XMax", "YMin", "YMax"]]

                # YOLO format: center x, center y, width, height (normalized)
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Copy/symlink image
        img_src = os.path.join(data_dir, f"{img_id}.jpg")
        img_dest = os.path.join(images_dir, f"{img_id}.jpg")

        if os.path.exists(img_src) and not os.path.exists(img_dest):
            try:
                os.symlink(img_src, img_dest)
            except OSError:
                shutil.copy2(img_src, img_dest)

    print(f"✓ Prepared {len(grouped)} images for {split}\n")


def create_yaml(yolo_root: str):
    classes_sorted = sorted(MATCHED_CLASSES)
    yaml_path = os.path.join(yolo_root, "data.yaml")

    content = f"""path: {yolo_root}
train: images/train
val: images/val
nc: {len(classes_sorted)}
names: {classes_sorted}
"""

    with open(yaml_path, 'w') as f:
        f.write(content)

    print(f"✓ data.yaml created with {len(classes_sorted)} classes at {yaml_path}")
    return yaml_path


# ====================== MAIN ======================
if __name__ == "__main__":
    yaml_path = os.path.join(YOLO_DATA_ROOT, "data.yaml")

    if not os.path.exists(yaml_path):
        os.makedirs(YOLO_DATA_ROOT, exist_ok=True)

        train_path = os.path.join(BASE_DIR, "datasets/open-images-v7/open-images-v7/train")
        val_path = os.path.join(BASE_DIR, "datasets/open-images-v7/open-images-v7/validation")

        prepare_oi_yolo_split(train_path, "train", YOLO_DATA_ROOT)
        prepare_oi_yolo_split(val_path, "val", YOLO_DATA_ROOT)

        create_yaml(YOLO_DATA_ROOT)
    else:
        print("Dataset already prepared. Delete data.yaml if you want to re-process.")