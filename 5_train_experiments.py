from ultralytics import YOLO
import os
from datetime import datetime

# ========================= CONFIG =========================
BASE_DIR = "/p/composml/object-detection-benchmark"
YOLO_DATA_ROOT = os.path.join(BASE_DIR, "openimages_v7_24classes")
PROJECT_DIR = os.path.join(BASE_DIR, "oi_finetune_experiments")

EPOCHS = 35
IMG_SIZE = 640
DEVICE = 0

STRATEGIES = {
    "baseline": {
        "mosaic": 0.0,
        "copy_paste": 0.0,
        "mixup": 0.0,
        "augment": False
    },
    "mosaic": {
        "mosaic": 1.0,
        "copy_paste": 0.0,
        "mixup": 0.0
    },
    "mixup": {
        "mosaic": 0.0,
        "copy_paste": 0.0,
        "mixup": 0.1
    },
}

MODELS = [
    ("yolo11l.pt",  "yolo11l"),
    ("rtdetr-l.pt", "rtdetr-l"),
]
# =========================================================

os.makedirs(PROJECT_DIR, exist_ok=True)

print(f"🚀 Starting Small Object Fine-Tuning Experiments - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*90)

no_augs = {
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "degrees": 0.0,
            "close_mosaic": 0,
        }


for model_pt, model_name in MODELS:
    print(f"\n🔬 Training {model_name.upper()} with aug strategies...\n")

    for strategy_name, aug_params in STRATEGIES.items():
        run_name = f"{model_name}_{strategy_name}"
        print(f"   → Running: {run_name}")
      
        model = YOLO(model_pt)
          
        model.train(
            data=os.path.join(YOLO_DATA_ROOT, "data.yaml"),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch= 8,                   # Use small batch for larger models
            device=DEVICE,
            project=PROJECT_DIR,
            name=run_name,
            exist_ok=True,
            patience=20,
            amp=True,
            cache="disk",                                                 # Safe for large dataset
            workers=4,
            seed=42,
            resume=False,
            fraction=0.9,

            # Learning Rate
            lr0=0.001,                                                    # Lower initial LR
            lrf=0.01,                                                     # Stronger final LR decay
            warmup_epochs=5.0,                                            # Gradual warmup
            warmup_bias_lr=0.1,
            weight_decay=0.0005,                                          # small penalty if the model’s weights become too large
            label_smoothing=0.0,                                          # less overconfident about the exact class
        
            # Save & Validate
            val=True,
            save=True,
            save_period=5,

            # Strategy-specific augmentations (mosaic, copy_paste, mixup)
            **aug_params,
            
            # Common augmentations (now cleanly handled)
            **no_augs,
        )

        print(f"   ✓ Finished {run_name}\n")

print("\n🎉 ALL EXPERIMENTS COMPLETED!")
print(f"Results saved in: {PROJECT_DIR}")
