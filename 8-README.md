# Small Object Detection & Dataset Analysis Benchmark

## Overview

Complete experimental pipeline to benchmark and optimize **small object detection** ($Area \le 32^2\text{ px}$) across **MS COCO 2017** and **OpenImages V7**.

The project covers dataset downloading, class-level small object distribution analysis, dataset matching across 24 shared classes, fine-tuning modern detectors (YOLO11l and RT-DETR-l), and GPU benchmarking with official COCO evaluation metrics.

---

## Repository Structure

* `1_download_datasets.py` — Unified dataset & pre-trained weight downloader


* `2_dataset_analysis.py` — Small-object statistics & visualizations for COCO and OpenImages


* `3_prepare_oi_yolo_24classes.py` — Prepares the 24-class matched OpenImages subset in YOLO format


* `4_analyze_oi_24classes.py` — Comparative analysis between COCO and OpenImages 24-class subsets


* `5_train_experiments.py` — Fine-tuning with augmentation strategies (Baseline, Mosaic, Mixup)


* `6_train_oversampling.py` — Fine-tuning with small-object image oversampling (2x, 3x)


* `7_compare_experiments.py` — GPU benchmark & COCOeval performance comparison



---

## 24 Matched Classes

To ensure fair cross-dataset benchmarking, the following 24 shared classes are used:

> `apple`, `backpack`, `baseball glove`, `bird`, `boat`, `book`, `bottle`, `car`, `clock`, `handbag`, `kite`, `knife`, `mouse`, `orange`, `parking meter`, `person`, `sheep`, `snowboard`, `spoon`, `tie`, `toothbrush`, `traffic light`, `vase`, `wine glass`
> 

---

## How to Run

### 1. Download Datasets & Weights

Download COCO 2017, OpenImages V7, and model checkpoints.

```bash
python 1_download_datasets.py

```

### 2. Dataset Analysis

Generate small-object distributions and charts for COCO and OpenImages.

```bash
python 2_dataset_analysis.py

```

### 3. Prepare 24-Class OpenImages Subset

Filter OpenImages to the 24 shared classes and export annotations in YOLO format.

```bash
python 3_prepare_oi_yolo_24classes.py

```

### 4. Analyze 24-Class Subset

Run comparative small-object distribution analysis between the 24-class OpenImages subset and COCO.

```bash
python 4_analyze_oi_24classes.py

```

### 5. Training Experiments

Fine-tune **YOLO11l** and **RT-DETR-l** for 35 epochs:

* Augmentation experiments (Baseline, Mosaic, Mixup):


```bash
python 5_train_experiments.py

```


* Oversampling experiments (2x, 3x):


```bash
python 6_train_oversampling.py --factor 2
python 6_train_oversampling.py --factor 3

```



### 6. Benchmark & Model Evaluation

Evaluate all trained models using FP16/AMP on COCO and OpenImages validation sets to generate metrics ($\text{mAP}_{50-95}$, $\text{AP}_{\text{small}}$, Latency, FPS) and plots.

```bash
python 7_compare_experiments.py

```

---

## Requirements

```bash
pip install torch torchvision ultralytics transformers fiftyone pycocotools pandas numpy matplotlib seaborn pillow tqdm pyyaml

```