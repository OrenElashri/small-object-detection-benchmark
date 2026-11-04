from ultralytics import YOLO
 
import os
import json
import copy
import time
 
import torch
import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
from datetime import datetime
from tqdm import tqdm
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import yaml

# ============================================================
# CONFIG
# ============================================================
 
BASE_DIR      = "/p/composml/object-detection-benchmark"
PROJECT_DIR   = os.path.join(BASE_DIR, "oi_finetune_experiments")
OUTPUT_DIR    = os.path.join(PROJECT_DIR, "Evaluation_comparison")
OUT_DIR       = OUTPUT_DIR
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
IMGSZ      = 640
BATCH_SIZE = 16
NUM_WORKERS = 4
RUNTIME_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
 
print("=" * 100)
print(f"FULL BENCHMARK STARTED\n{datetime.now()}")
print("=" * 100)
 
# ============================================================
# OPENIMAGES
# ============================================================

OI_DATA_YAML = os.path.join(BASE_DIR, "openimages_v7_24classes/data.yaml")
OI_IMGS = os.path.join(BASE_DIR, "openimages_v7_24classes/images/val")
OI_LABELS = os.path.join(BASE_DIR, "openimages_v7_24classes/labels/val")

# ============================================================
# COCO
# ============================================================
 
COCO_DIR  = os.path.join(BASE_DIR, "datasets/coco")
COCO_IMGS = os.path.join(COCO_DIR, "images/val2017")
COCO_ANN  = os.path.join(COCO_DIR, "annotations/instances_val2017.json")
COCO_YAML = os.path.join(BASE_DIR, "ultralytics/ultralytics/cfg/datasets/coco.yaml")

MATCHED_COCO_DIR = os.path.join(OUTPUT_DIR, "matched_coco")
MATCHED_COCO_YAML = os.path.join(OUTPUT_DIR,"matched_coco.yaml")

# ============================================================
# MATCHED CLASSES
# ============================================================
 
MATCHED_CLASSES = {
    'apple', 'backpack', 'baseball glove', 'bird',
    'boat', 'book', 'bottle', 'car', 'clock', 'handbag', 'kite', 'knife',
    'mouse', 'orange', 'parking meter', 'person', 'sheep', 'snowboard',
    'spoon', 'tie', 'toothbrush', 'traffic light', 'vase', 'wine glass'
}
  
# ============================================================
# REMAPPING
# ============================================================

MODEL_CLASS_MAP = {
'apple':0,
'backpack':1,
'baseball glove':2,
'bird':3,
'boat':4,
'book':5,
'bottle':6,
'car':7,
'clock':8,
'handbag':9,
'kite':10,
'knife':11,
'mouse':12,
'orange':13,
'parking meter':14,
'person':15,
'sheep':16,
'snowboard':17,
'spoon':18,
'tie':19,
'toothbrush':20,
'traffic light':21,
'vase':22,
'wine glass':23,
}

# ============================================================
# COCO CATEGORY IDS
# ============================================================

COCO_MATCHED_IDS = {
    'person': 0, 'car': 2, 'boat': 8, 'traffic light': 9, 'parking meter': 12,
    'bird': 14, 'sheep': 18, 'backpack': 24, 'handbag': 26, 'tie': 27,
    'snowboard': 31, 'kite': 33, 'baseball glove':35, 'bottle': 39,
    'wine glass': 40, 'knife': 43, 'spoon': 44, 'apple': 47, 'orange': 49,
    'mouse': 64, 'book': 73, 'clock': 74, 'vase': 75, 'toothbrush': 79,
}



COCO_TO_MODEL = {}

for name, coco_id in COCO_MATCHED_IDS.items():
    COCO_TO_MODEL[coco_id] = MODEL_CLASS_MAP[name]

# ============================================================
# MODELS
# ============================================================
 
MODEL_COLORS = {
    # ── YOLO11l — purple/magenta family ──────────────────────────────────────
    "yolo11l_baseline":      "#9400D3",
    "yolo11l_mosaic":        "#DDA0DD",
    "yolo11l_mixup":         "#7B68EE",
    "yolo11l_oversample_2x": "#8B008B",
    "yolo11l_oversample_3x": "#C8A2C8",
    # ── RT-DETR-l — green family ─────────────────────────────────────────────
    "rtdetr-l_baseline":     "#228B22",
    "rtdetr-l_mosaic":       "#32CD32",
    "rtdetr-l_mixup":      "#3CB371",
    "rtdetr-l_oversample_2x": "#00FF00",
    "rtdetr-l_oversample_3x": "#98FB98",
}
 
MODELS_TO_EVAL = [
    # (path, label, family, impl_type)
    # ── YOLO11l ──────────────────────────────────────────────────────────────
    (f"{PROJECT_DIR}/yolo11l_baseline/weights/best.pt",                             "yolo11l_baseline",      "YOLO11",              "ultralytics"),
    (f"{PROJECT_DIR}/yolo11l_mosaic/weights/best.pt",                               "yolo11l_mosaic",        "YOLO11",              "ultralytics"),
    (f"{PROJECT_DIR}/yolo11l_mixup/weights/best.pt",                                "yolo11l_mixup",         "YOLO11",              "ultralytics"),
    (f"{PROJECT_DIR}/oversample_experiments/yolo11l_oversample_2x/weights/best.pt", "yolo11l_oversample_2x", "YOLO11",              "ultralytics"),
    (f"{PROJECT_DIR}/oversample_experiments/yolo11l_oversample_3x/weights/best.pt", "yolo11l_oversample_3x", "YOLO11",              "ultralytics"),
    # ── RT-DETR-l ────────────────────────────────────────────────────────────
    (f"{PROJECT_DIR}/rtdetr-l_baseline/weights/best.pt",                            "rtdetr-l_baseline",     "RT-DETR-Ultralytics", "ultralytics"),
    (f"{PROJECT_DIR}/rtdetr-l_mosaic/weights/best.pt",                              "rtdetr-l_mosaic",       "RT-DETR-Ultralytics", "ultralytics"),
    (f"{PROJECT_DIR}/rtdetr-l_mixup/weights/best.pt",                             "rtdetr-l_mixup",        "RT-DETR-Ultralytics", "ultralytics"),
    (f"{PROJECT_DIR}/oversample_experiments/rtdetr-l_oversample_2x/weights/best.pt","rtdetr-l_oversample_2x","RT-DETR-Ultralytics","ultralytics"),
    (f"{PROJECT_DIR}/oversample_experiments/rtdetr-l_oversample_3x/weights/best.pt","rtdetr-l_oversample_3x","RT-DETR-Ultralytics","ultralytics"),
]

# ============================================================
# DATASET CLASSES
# ============================================================

class CocoSmallObjectDataset(Dataset):
    def __init__(self, img_dir, ann_file, matched_classes, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.coco = COCO(ann_file)
        self.matched_cat_ids = []
        self.cat_id_to_name = {}
        
        for cat_id, cat_info in self.coco.cats.items():
            if cat_info['name'] in matched_classes:
                self.matched_cat_ids.append(cat_id)
                self.cat_id_to_name[cat_id] = cat_info['name']
        
        self.img_ids = []
        for cat_id in self.matched_cat_ids:
            self.img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
        self.img_ids = sorted(list(set(self.img_ids)))
       
        self.class_name_to_id = {v: k for k, v in self.cat_id_to_name.items()}
        
        sorted_classes = sorted(matched_classes)
        self.names = sorted_classes   # Model was trained on these 24 classes
        
        print(f" → COCO: {len(self.img_ids)} images with matched classes")
   
    def __len__(self):
        return len(self.img_ids)
   
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        orig_size = img.size
        if self.transform:
            img_tensor = self.transform(img)
        return img_tensor, img_id, (orig_size[0], orig_size[1])


class OpenImagesSmallObjectDataset(Dataset):
    """YOLO-format dataset (labels/*.txt). No CSV needed."""
    def __init__(self, img_dir, labels_dir, data_yaml_path, matched_classes, transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Load class names from data.yaml
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        self.names = data['names']
        
        # COCO-style category mapping (sorted)
        sorted_classes = sorted(matched_classes)
        self.class_name_to_cat_id = {name: idx + 1 for idx, name in enumerate(sorted_classes)}
        self.cat_id_to_class_name = {idx + 1: name for idx, name in enumerate(sorted_classes)}
        
        # YOLO class_id (from data.yaml order) → our cat_id
        self.yolo_id_to_cat_id = {}
        for yolo_idx, name in enumerate(self.names):
            if name in self.class_name_to_cat_id:
                self.yolo_id_to_cat_id[yolo_idx] = self.class_name_to_cat_id[name]

        print(" → Loading Open Images YOLO labels...")
        self.img_ids = []
        self.img_id_to_labels = {}
        
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        for txt_file in tqdm(label_files, desc=" Scanning labels"):
            img_id = txt_file.replace('.txt', '')
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                self.img_ids.append(img_id)
                labels = []
                with open(os.path.join(labels_dir, txt_file), 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                cx, cy, w, h = map(float, parts[1:5])
                                labels.append((cls_id, cx, cy, w, h))
                self.img_id_to_labels[img_id] = labels

        self.img_ids = sorted(self.img_ids)
        self.img_id_to_idx = {img_id: idx + 1 for idx, img_id in enumerate(self.img_ids)}
        print(f" → Open Images: {len(self.img_ids)} images with matched classes")

    def __len__(self):
        return len(self.img_ids)
   
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Missing image → black placeholder: {img_path}")
            img = Image.new("RGB", (640, 640), (0, 0, 0))
        orig_size = img.size
        if self.transform:
            img_tensor = self.transform(img)
        return img_tensor, img_id, (orig_size[0], orig_size[1])

# ============================================================
# HELPERS
# ============================================================

def count_params(model, impl_type):
    if impl_type == "ultralytics":
        return round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

def create_exp_dir(label, dataset):
    d = os.path.join(OUT_DIR, f"{RUNTIME_ID}_{label}_{dataset}")
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================
# REMAPPED COCO LABLES
# ============================================================

def remap_coco_yolo_labels(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fn in os.listdir(src_dir):
        if not fn.endswith(".txt"):
            continue

        src = os.path.join(src_dir, fn)
        dst = os.path.join(dst_dir, fn)
        out_lines = []
        with open(src) as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                if cls not in COCO_TO_MODEL:
                    continue

                new_cls = COCO_TO_MODEL[cls]
                out_lines.append(str(new_cls) + " " + " ".join(parts[1:]))

        with open(dst,"w") as f:
            f.write("\n".join(out_lines))

# ============================================================
# FPS — YOLO
# ============================================================

def measure_fps_yolo(model, loader, warmup_batches=80, timing_batches=150):
    """Measure FPS for YOLO models with FP16."""
    model.model.eval()

    print("    Warming up...")
    with torch.inference_mode():
        for i, (img_batch, _, _) in enumerate(loader):
            if i >= warmup_batches:
                break
            _ = model(img_batch.to(DEVICE), verbose=False, half=True, imgsz=IMGSZ)
    torch.cuda.synchronize()
    time.sleep(0.5)

    print("    Measuring inference speed...")
    times = []
    with torch.inference_mode():
        for i, (img_batch, _, _) in enumerate(loader):
            if i >= timing_batches:
                break
            bs = img_batch.shape[0]
            if bs < BATCH_SIZE:          # skip incomplete tail batch
                continue
            torch.cuda.synchronize()
            start = time.time()
            _ = model(img_batch.to(DEVICE), verbose=False, half=True, imgsz=IMGSZ)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.extend([elapsed / bs] * bs)

    avg_time = np.median(times)
    return round(1 / avg_time if avg_time > 0 else 0.0, 2)

# ============================================================
# OPENIMAGES GT ANNOTATIONS
# ============================================================

def create_openimages_annotations(dataset, output_path):
    """Create COCO-format GT from YOLO .txt labels."""
    img_ids = dataset.img_ids
    img_id_to_idx = dataset.img_id_to_idx
    
    print(" Loading image sizes and converting YOLO labels to COCO...")
    images, annotations = [], []
    ann_id = 1
    
    for img_id in tqdm(img_ids, desc=" Building GT"):
        idx = img_id_to_idx[img_id]
        img_path = os.path.join(dataset.img_dir, f"{img_id}.jpg")
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            continue
        
        images.append({"id": idx, "file_name": f"{img_id}.jpg", "width": w, "height": h})
        
        for cls_id, cx, cy, bw, bh in dataset.img_id_to_labels.get(img_id, []):
            if cls_id not in dataset.yolo_id_to_cat_id:
                continue
            cat_id = dataset.yolo_id_to_cat_id[cls_id]
            
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            bbox_w = bw * w
            bbox_h = bh * h
            
            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": cat_id,
                "bbox": [x1, y1, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "iscrowd": 0,
            })
            ann_id += 1
    
    categories = [
        {"id": cat_id, "name": name, "supercategory": "object"}
        for cat_id, name in sorted(dataset.cat_id_to_class_name.items())
    ]
    
    with open(output_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    
    print(f" → Created {len(images)} images, {len(annotations)} annotations")
    return output_path


# ============================================================
# EVALUATION — YOLO COCO
# ============================================================

# def evaluate_coco_yolo(model, exp_dir, matched_classes):
#     print(" Running COCO validation (matched 24 classes)...")
#     print(f"Model classes: {len(model.names)}")

#     data_yaml = MATCHED_COCO_YAML
#     if not os.path.exists(data_yaml):
#         print("ERROR: matched_coco.yaml missing!")
#         return {"mAP50_95": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}

#     # Force clear ALL possible caches
#     for cache_path in [
#         os.path.join(MATCHED_COCO_DIR, "labels", "val2017.cache"),
#         os.path.join(COCO_DIR, "labels", "val2017.cache"),
#     ]:
#         if os.path.exists(cache_path):
#             os.remove(cache_path)
#             print(f"   → Cleared cache: {cache_path}")

#     metrics = model.val(
#         data=data_yaml,
#         imgsz=640,
#         batch=BATCH_SIZE,
#         half=True,
#         project=exp_dir,
#         name="val",
#         save_json=True,
#         verbose=False,
#         conf=0.001,
#         iou=0.6,
#         max_det=300,
#         rect=False,
#         plots=False
#     )

#     result_dict = {
#         "mAP50_95": float(metrics.box.map),
#         "AP_small": 0.0,
#         "AP_medium": 0.0,
#         "AP_large": 0.0,
#     }

#     pred_path = os.path.join(exp_dir, "val", "predictions.json")
#     if os.path.exists(pred_path):
#         try:
#             coco_gt = COCO(COCO_ANN)
#             coco_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
#             coco_name_to_id = {v: k for k, v in coco_id_to_name.items()}

#             matched_cat_ids = [coco_name_to_id[cls] for cls in matched_classes if cls in coco_name_to_id]
#             matched_img_ids = sorted(set(
#                 img_id for cat_id in matched_cat_ids 
#                 for img_id in coco_gt.getImgIds(catIds=[cat_id])
#             ))

#             coco_dt = coco_gt.loadRes(pred_path)
#             coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
#             coco_eval.params.catIds = matched_cat_ids
#             coco_eval.params.imgIds = matched_img_ids
#             coco_eval.evaluate()
#             coco_eval.accumulate()
#             coco_eval.summarize()

#             result_dict["AP_small"] = float(coco_eval.stats[3])
#             result_dict["AP_medium"] = float(coco_eval.stats[4]) if len(coco_eval.stats) > 4 else 0.0
#             result_dict["AP_large"] = float(coco_eval.stats[5]) if len(coco_eval.stats) > 5 else 0.0
#         except Exception as e:
#             print(f"   Warning: pycocotools eval failed: {e}")

#     return result_dict


def evaluate_coco_yolo(model, loader, dataset, exp_dir):
    """
    Manual inference + filtered COCOeval (reliable way)
    """
    print(" Collecting predictions on COCO (matched classes)...")
    model.model.eval()
    predictions = []
    
    with torch.inference_mode():
        for img_batch, img_id_batch, orig_sizes in tqdm(loader, desc=" Inference"):
            results = model(img_batch.to(DEVICE), verbose=False, half=True, imgsz=640)
            
            for b in range(img_batch.shape[0]):
                img_id = img_id_batch[b]
                orig_w, orig_h = orig_sizes[b]
                scale_x = orig_w / 640.0
                scale_y = orig_h / 640.0
                
                if results[b].boxes is None or len(results[b].boxes) == 0:
                    continue
                
                for box, conf, cls_idx in zip(
                    results[b].boxes.xyxy,
                    results[b].boxes.conf,
                    results[b].boxes.cls
                ):
                    cls_idx_int = int(cls_idx)
                    class_name = dataset.names[cls_idx_int] if cls_idx_int < len(dataset.names) else None
                    
                    if class_name not in MATCHED_CLASSES:
                        continue
                    
                    x1, y1, x2, y2 = box.cpu().numpy()
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y
                    
                    # Use original COCO category ID
                    category_id = dataset.class_name_to_id[class_name]
                    
                    predictions.append({
                        "image_id": int(img_id),
                        "category_id": int(category_id),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(conf)
                    })
    
    if not predictions:
        print(" Warning: No predictions collected!")
        return {"mAP50_95": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}
    
    print(f" Collected {len(predictions)} predictions")
    
    pred_path = os.path.join(exp_dir, "coco_predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
    
    # Final accurate evaluation
    coco_gt = COCO(COCO_ANN)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    
    matched_cat_ids = [dataset.class_name_to_id[cls] for cls in MATCHED_CLASSES]
    matched_img_ids = sorted(set(
        img_id for cat_id in matched_cat_ids 
        for img_id in coco_gt.getImgIds(catIds=[cat_id])
    ))
    
    coco_eval.params.catIds = matched_cat_ids
    coco_eval.params.imgIds = matched_img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return {
        "mAP50_95": float(coco_eval.stats[0]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]) if len(coco_eval.stats) > 4 else 0.0,
        "AP_large": float(coco_eval.stats[5]) if len(coco_eval.stats) > 5 else 0.0,
    }
    

def evaluate_openimages_yolo(model, loader, dataset, gt_ann_path, exp_dir):
    """
    Evaluate YOLO models on filtered Open Images (custom categories).
    """
    print(" Collecting predictions on Open Images (YOLO)...")
    model.model.eval()
    predictions = []
    
    with torch.inference_mode():
        for img_batch, img_id_batch, orig_sizes in tqdm(loader, desc=" Inference"):
            results = model(img_batch.to(DEVICE), verbose=False, half=True, imgsz=640)
            for b in range(img_batch.shape[0]):
                img_id = img_id_batch[b]
                coco_img_id = dataset.img_id_to_idx[img_id]
                orig_w, orig_h = orig_sizes[b]
                scale_x = orig_w / 640.0
                scale_y = orig_h / 640.0
                
                if results[b].boxes is None or len(results[b].boxes) == 0:
                    continue
                
                for box, conf, cls_idx in zip(
                    results[b].boxes.xyxy,
                    results[b].boxes.conf,
                    results[b].boxes.cls
                ):
                    cls_idx_int = int(cls_idx)
                    # Ultralytics YOLO class indices match training (data.yaml order)
                    class_name = dataset.names[cls_idx_int] if cls_idx_int < len(dataset.names) else None
                    if class_name not in dataset.class_name_to_cat_id:
                        continue
                    
                    x1, y1, x2, y2 = box.cpu().numpy()
                    x1 *= scale_x; y1 *= scale_y; x2 *= scale_x; y2 *= scale_y
                    
                    predictions.append({
                        "image_id": int(coco_img_id),
                        "category_id": int(dataset.class_name_to_cat_id[class_name]),
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        "score": float(conf)
                    })
    
    if not predictions:
        print(" Warning: No predictions collected!")
        return {"mAP50_95": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}
    
    print(f" Collected {len(predictions)} predictions")
    pred_path = os.path.join(exp_dir, "openimages_predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
    
    coco_gt = COCO(gt_ann_path)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return {
        "mAP50_95": float(coco_eval.stats[0]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]) if len(coco_eval.stats) > 4 else 0.0,
        "AP_large": float(coco_eval.stats[5]) if len(coco_eval.stats) > 5 else 0.0,
    }

# ============================================================
# COLLATE
# ============================================================

def custom_collate_fn(batch):
    imgs       = torch.stack([item[0] for item in batch], dim=0)
    img_ids    = [item[1] for item in batch]
    orig_sizes = [item[2] for item in batch]   # list of (W, H)
    return imgs, img_ids, orig_sizes

# ============================================================
# RESULT ROW BUILDER  — unified schema for both datasets
# ============================================================

def make_row(model_name, family, params, latency_ms, fps, metrics):
    """Build a single result dict with the target column schema."""
    return {
        "model":      model_name,
        "family":     family,
        "mAP50-95":   round(metrics["mAP50_95"],   6),
        "AP_small":   round(metrics["AP_small"],   6),
        "AP_medium":  round(metrics["AP_medium"],  4),
        "AP_large":   round(metrics["AP_large"],   4),
        "Latency_ms": round(latency_ms,            2),
        "FPS":        round(fps,                   2),
        "params_M":   params,
    }

# ============================================================
# PLOT — one per dataset
# ============================================================

TABLE_COLS = ["model", "mAP50-95", "AP_small", "AP_medium", "AP_large", "Latency_ms", "FPS", "params_M"]
 
def plot_latency_vs_ap_small(df, dataset_label, plot_dir, runtime_id):
    """
    One scatter plot per dataset.
    X = Latency_ms,  Y = AP_small,  color = per-model from MODEL_COLORS[label].
    Legend groups by family using the first model's color in that family.
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="black")
    ax.set_facecolor("black")
 
    # Plot each model with its own color
    for _, row in df.iterrows():
        color = MODEL_COLORS.get(row["model"], "#FFFFFF")
        ax.scatter(row["Latency_ms"], row["AP_small"],
                   color=color, s=140, zorder=5)
        ax.text(row["Latency_ms"] * 1.015, row["AP_small"],
                row["model"], fontsize=8, color="white",
                va="center", ha="left")
 
    # Legend — one entry per family, using first model's color in that family
    seen_families = {}
    for _, row in df.iterrows():
        fam = row["family"]
        if fam not in seen_families:
            seen_families[fam] = MODEL_COLORS.get(row["model"], "#FFFFFF")
 
    legend_handles = [
        mpatches.Patch(color=color, label=fam)
        for fam, color in seen_families.items()
    ]
 
    ax.set_xlabel("Latency (ms / image)", fontsize=12, fontweight="bold", color="white")
    ax.set_ylabel("AP Small (< 32×32 px, IoU=0.50:0.95)", fontsize=12, fontweight="bold", color="white")
    ax.set_title(
        f"Small Object Detection — {dataset_label.upper()}\n"
        f"24 Matched Classes | FP16 / AMP",
        fontsize=14, fontweight="bold", pad=20, color="white"
    )
    ax.grid(True, alpha=0.3, color="gray")
    ax.legend(handles=legend_handles, fontsize=11,
              facecolor="black", edgecolor="white", labelcolor="white")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.4f}"))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
 
    plt.tight_layout()
    out_path = os.path.join(plot_dir, f"{runtime_id}_{dataset_label}_latency_vs_AP_small.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"  ✓ Plot saved: {out_path}")
    return out_path

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("PREPARING DATASETS")
    print("=" * 80)
    transform = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
    ])
   
    print("\nLoading COCO dataset...")
    coco_dataset = CocoSmallObjectDataset(COCO_IMGS, COCO_ANN, MATCHED_CLASSES, transform)
    coco_loader = DataLoader(
        coco_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn,
    )
  
    COCO_LABELS_SRC = os.path.join(COCO_DIR, "labels", "val2017")
    MATCHED_IMAGES = os.path.join(MATCHED_COCO_DIR, "images", "val2017")
    MATCHED_LABELS = os.path.join(MATCHED_COCO_DIR, "labels", "val2017")
   
    os.makedirs(MATCHED_LABELS, exist_ok=True)
    os.makedirs(os.path.dirname(MATCHED_IMAGES), exist_ok=True)
   
    # symlink images
    if not os.path.exists(MATCHED_IMAGES):
        os.symlink(COCO_IMGS, MATCHED_IMAGES)
   
    # Remap labels (COCO class ids → your 24-class ids)
    print("Ensuring remapped COCO labels...")
    remap_coco_yolo_labels(COCO_LABELS_SRC, MATCHED_LABELS)
    
    def write_matched_coco_yaml():
        yaml_data = {
            "path": MATCHED_COCO_DIR,
            "train": "images/val2017",
            "val": "images/val2017",
            "test": "images/val2017",
            "names": {
                0: 'apple', 1: 'backpack', 2: 'baseball glove', 3: 'bird',
                4: 'boat', 5: 'book', 6: 'bottle', 7: 'car', 8: 'clock',
                9: 'handbag', 10: 'kite', 11: 'knife', 12: 'mouse',
                13: 'orange', 14: 'parking meter', 15: 'person',
                16: 'sheep', 17: 'snowboard', 18: 'spoon', 19: 'tie',
                20: 'toothbrush', 21: 'traffic light', 22: 'vase',
                23: 'wine glass'
            },
            "nc": 24
        }
        with open(MATCHED_COCO_YAML, "w") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False)
        print(f" → Created matched_coco.yaml with 24 classes at {MATCHED_COCO_YAML}")

    write_matched_coco_yaml()

    print("\nLoading Open Images dataset (YOLO format)...")
    oi_dataset = OpenImagesSmallObjectDataset(
        OI_IMGS, OI_LABELS, OI_DATA_YAML, MATCHED_CLASSES, transform
    )
    oi_loader = DataLoader(
        oi_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn,
    )
   
    print("\nCreating Open Images ground truth (from YOLO txt)...")
    oi_gt_path = os.path.join(OUT_DIR, f"{RUNTIME_ID}_openimages_gt.json")
    create_openimages_annotations(oi_dataset, oi_gt_path)
   
    # ------------------------------------------------------------------
    # Per-dataset result accumulators
    # ------------------------------------------------------------------
    coco_rows = []
    oi_rows = []
    print("\n" + "=" * 80)
    print("EVALUATING MODELS (FP16 + AMP)")
    print("=" * 80)
   
    for model_path, label, family, impl_type in MODELS_TO_EVAL:
        print(f"\n{'=' * 80}")
        print(f"Model: {label} ({family})")
        print(f"{'=' * 80}")
        if not os.path.exists(model_path):
            print(f" SKIPPED — file not found: {model_path}")
            continue
        try:
            model = YOLO(model_path).to(DEVICE)
            params = count_params(model, "ultralytics")
           
            # ── COCO ──────────────────────────────────────────────────────────
            print("\n COCO Evaluation (FP16):")
            exp_dir_coco = create_exp_dir(label, "coco")
            fps_coco = measure_fps_yolo(model, coco_loader)
            latency_coco = 1000 / fps_coco if fps_coco > 0 else 0.0
            print(f" FPS: {fps_coco:.2f} | Latency: {latency_coco:.2f} ms/img")
            
            coco_metrics = evaluate_coco_yolo(model, coco_loader, coco_dataset, exp_dir_coco)
            print(f"   COCO mAP: {coco_metrics['mAP50_95']:.4f} | AP_small: {coco_metrics['AP_small']:.4f}")
            coco_rows.append(make_row(label, family, params, latency_coco, fps_coco, coco_metrics))
           
            # ── Open Images ───────────────────────────────────────────────────
            print("\n Open Images Evaluation (FP16):")
            exp_dir_oi = create_exp_dir(label, "openimages")
            fps_oi = measure_fps_yolo(model, oi_loader)
            latency_oi = 1000 / fps_oi if fps_oi > 0 else 0.0
            print(f" FPS: {fps_oi:.2f} | Latency: {latency_oi:.2f} ms/img")
            oi_metrics = evaluate_openimages_yolo(
                model, oi_loader, oi_dataset, oi_gt_path, exp_dir_oi
            )
            oi_rows.append(make_row(label, family, params, latency_oi, fps_oi, oi_metrics))
           
            # Save per-model metrics
            for ds_name, exp_dir, fps, lat, m in [
                ("COCO", exp_dir_coco, fps_coco, latency_coco, coco_metrics),
                ("Open Images V7", exp_dir_oi, fps_oi, latency_oi, oi_metrics),
            ]:
                with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
                    json.dump({
                        "model": label, "family": family, "params_M": params,
                        "fps": fps, "latency_ms": lat,
                        "dataset": ds_name, "precision": "FP16",
                        **m,
                    }, f, indent=2)
           
            print(f"\n ✓ Done: {label}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # FINAL TABLES
    # ============================================================

    coco_df = pd.DataFrame(coco_rows)
    oi_df   = pd.DataFrame(oi_rows)

    if len(coco_df) > 0:
        coco_df = coco_df.sort_values("mAP50-95", ascending=False)
    if len(oi_df) > 0:
        oi_df = oi_df.sort_values("mAP50-95", ascending=False)

    print("\n")
    print("=" * 100)
    print("FINAL TABLE — COCO (24 Matched Classes)")
    print("=" * 100)
    if len(coco_df) > 0:
        print(coco_df[TABLE_COLS].to_string(index=False))
    else:
        print("  (no results)")

    print("\n")
    print("=" * 100)
    print("FINAL TABLE — OPEN IMAGES V7 (24 Matched Classes)")
    print("=" * 100)
    if len(oi_df) > 0:
        print(oi_df[TABLE_COLS].to_string(index=False))
    else:
        print("  (no results)")

    # ============================================================
    # SAVE CSVs
    # ============================================================

    coco_csv = os.path.join(OUT_DIR, f"{RUNTIME_ID}_coco_results.csv")
    oi_csv   = os.path.join(OUT_DIR, f"{RUNTIME_ID}_openimages_results.csv")
    coco_df.to_csv(coco_csv, index=False)
    oi_df.to_csv(oi_csv,     index=False)
    print(f"\n✓ CSV saved: {coco_csv}")
    print(f"✓ CSV saved: {oi_csv}")

    # ============================================================
    # PLOTS — one per dataset
    # ============================================================

    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plot_dir = os.path.join(OUT_DIR, f"{RUNTIME_ID}_plots")
    os.makedirs(plot_dir, exist_ok=True)

    if len(coco_df) > 0:
        plot_latency_vs_ap_small(coco_df, "coco", plot_dir, RUNTIME_ID)

    if len(oi_df) > 0:
        plot_latency_vs_ap_small(oi_df, "openimages", plot_dir, RUNTIME_ID)

    # ============================================================
    # SUMMARY JSON
    # ============================================================

    summary = {
        "coco":        coco_df.to_dict(orient="records") if len(coco_df) > 0 else [],
        "openimages":  oi_df.to_dict(orient="records")   if len(oi_df)   > 0 else [],
    }
    summary_path = os.path.join(OUT_DIR, f"{RUNTIME_ID}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Outputs:  {OUT_DIR}")
    print(f"✓ Plots:    {plot_dir}")
    print(f"✓ Summary:  {summary_path}")
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("KEY METRIC: AP_small — objects < 32×32 px, IoU=0.50:0.95")
    print("=" * 80)


if __name__ == "__main__":
    main()
