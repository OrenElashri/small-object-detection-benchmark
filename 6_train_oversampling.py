from ultralytics import YOLO
import os
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from PIL import Image
import argparse

# ========================= Argument Parser =========================
parser = argparse.ArgumentParser()
parser.add_argument('--factor', type=int, default=2, choices=[2, 3])
args = parser.parse_args()

OVERSAMPLE_LEVELS = [args.factor]   # Override to run only one factor


# ========================= CONFIG =========================
BASE_DIR = "/p/composml/object-detection-benchmark"
ORIGINAL_DATA_ROOT = Path(BASE_DIR) / "openimages_v7_24classes"
PROJECT_DIR = Path(BASE_DIR) / "oi_finetune_experiments/oversample_experiments"

EPOCHS = 35
IMG_SIZE = 640
DEVICE = 0

MODELS = [
    # ("yolo11l.pt", "yolo11l"),
    ("rtdetr-l.pt", "rtdetr-l"),
]


# Small object = 32x32 pixels or smaller
SMALL_OBJECT_THRESHOLD = 32 * 32  

print(f"🚀 Starting Improved Small Object Oversampling - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
# =========================================================

def safe_link_or_copy(src, dst):
    # Use a hard link if possible to avoid extra disk usage.
    # Fall back to a normal copy if hard links are not supported.
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def get_small_object_images():
    # Find training images that contain at least one object smaller than the threshold.
    label_train_dir = ORIGINAL_DATA_ROOT / "labels/train"
    image_train_dir = ORIGINAL_DATA_ROOT / "images/train"
    small_images = []

    for label_file in label_train_dir.glob("*.txt"):
        img_file = image_train_dir / f"{label_file.stem}.jpg"
        if not img_file.exists():
            continue

        try:
            # Read the real image size because Open Images has mixed resolutions.
            with Image.open(img_file) as im:
                img_w, img_h = im.size

            # Check all boxes in the label file.
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO labels are normalized, so convert box size back to pixels.
                        box_w = float(parts[3]) * img_w
                        box_h = float(parts[4]) * img_h
                        area = box_w * box_h

                        # Mark image if it contains at least one small object.
                        if area <= SMALL_OBJECT_THRESHOLD:
                            small_images.append(label_file.stem)
                            break
        except Exception:
            continue

    print(f"Found {len(small_images)} images with small objects (<=32x32).")
    return small_images



def create_oversampled_dataset(oversample_factor):
    # Create a dataset where selected small-object images appear multiple times.
    dataset_name = f"openimages_v7_24classes_oversampled{oversample_factor}"
    dataset_dir = Path(BASE_DIR) / dataset_name
    data_yaml_path = dataset_dir / "data.yaml"
    train_img_dir = dataset_dir / "images" / "train"
    train_lbl_dir = dataset_dir / "labels" / "train"

    # Reuse the dataset if it already exists and looks complete.
    if data_yaml_path.exists() and train_img_dir.exists():
        expected_min = len(list((ORIGINAL_DATA_ROOT / "images/train").glob("*.jpg")))
        current = len(list(train_img_dir.glob("*.jpg")))
        if current >= expected_min:
            print(f"✅ Using existing dataset: {dataset_name}")
            return str(data_yaml_path)

    # Rebuild the dataset from scratch.
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    src_images = ORIGINAL_DATA_ROOT / "images/train"
    src_labels = ORIGINAL_DATA_ROOT / "labels/train"

    # Copy or link all original training images and labels.
    print("Copying original images...")
    for img_path in src_images.glob("*.jpg"):
        dst_img = train_img_dir / img_path.name
        safe_link_or_copy(str(img_path), str(dst_img))

        lbl = src_labels / f"{img_path.stem}.txt"
        if lbl.exists():
            dst_lbl = train_lbl_dir / lbl.name
            safe_link_or_copy(str(lbl), str(dst_lbl))

    # Duplicate only images that contain small objects.
    small_images = get_small_object_images()
    print(f"Oversampling {len(small_images)} images by {oversample_factor}x...")

    for img_name in small_images:
        src_img = src_images / f"{img_name}.jpg"
        src_lbl = src_labels / f"{img_name}.txt"

        for i in range(1, oversample_factor):
            new_name = f"{img_name}_os{i}"
            safe_link_or_copy(str(src_img), str(train_img_dir / f"{new_name}.jpg"))
            if src_lbl.exists():
                safe_link_or_copy(str(src_lbl), str(train_lbl_dir / f"{new_name}.txt"))

    # Write a dataset YAML that points to the new training folder.
    with open(ORIGINAL_DATA_ROOT / "data.yaml", "r") as f:
        data = yaml.safe_load(f)

    data["train"] = str(train_img_dir)
    data["val"] = str(ORIGINAL_DATA_ROOT / "images/val")

    with open(data_yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"✅ Dataset {dataset_name} created!")
    return str(data_yaml_path)



# ===================== MAIN =====================
os.makedirs(PROJECT_DIR, exist_ok=True)

for model_pt, model_name in MODELS:
    for factor in OVERSAMPLE_LEVELS:
        run_name = f"{model_name}_oversample_{factor}x"
        print(f"\n🔬 Training {run_name}\n")

        # Build or reuse the oversampled dataset.
        oversampled_yaml = create_oversampled_dataset(factor)     
        model = YOLO(model_pt)
                        
        # Train with a small oversampling factor to keep compute cost low.
        model.train(
            data=oversampled_yaml,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch= 8,
            device=DEVICE,
            project=str(PROJECT_DIR),
            name=run_name,
            exist_ok=True,
            patience=20,
            amp=True,
            cache="disk",
            workers=4,
            seed=42,
            resume=False,
            
            # No augmentations
            augment=False,
            mosaic=0.0, copy_paste=0.0, mixup=0.0,
            translate=0.0, scale=0.0, shear=0.0, degrees=0.0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3.0,
            weight_decay=0.0005,
            label_smoothing=0.0,
            
            val=True,
            save=True,
            save_period=5,
        )
        
        print(f" ✓ Finished {run_name}\n")

print("\n🎉 All Small Object Oversampling Experiments Completed!")
