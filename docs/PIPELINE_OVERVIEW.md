# Pipeline Overview

## Current Pipeline

```text
Dataset
  |
  v
Dataset structure parser
  |
  v
RGB video loader + H5 depth loader + GT parser + calibration parser
  |
  v
Alignment/debug targets
  |
  v
YOLO dataset export
  |
  v
YOLO training and inference
  |
  v
YOLO detections CSV
  |
  v
Observation3D JSONL
```

## Future Pipeline

```text
Observation3D JSONL
  |
  v
3D estimator / depth-supervised heads
  |
  v
ReID embeddings
  |
  v
Deep-OC-SORT3D single-camera tracking
  |
  v
MTMC global association
  |
  v
track1.txt submission
```

## Components

### 1. Dataset Loader

The dataset loader parses scene structure and exposes videos, depth maps, ground truth, calibration, and map paths. Train/val have depth and GT. Test does not.

### 2. RGB/Depth/GT/Calibration Validation

Validation scripts confirm that:

- video frames are 0-based;
- ground truth frame keys are 0-based strings;
- H5 depth is read lazily;
- calibration is loaded per camera;
- visible 2D boxes align visually with RGB frames.

### 3. YOLO Dataset Export

Ground-truth visible 2D boxes are exported into YOLO format. This includes:

- `images/train`;
- `labels/train`;
- `data.yaml`;
- manifests and summaries for curriculum exports.

### 4. YOLO Training/Inference

YOLO is the 2D detector front-end. Current plan:

- YOLO11s for smoke tests and debugging.
- YOLO11m for main curriculum training.
- YOLO11l only later if YOLO11m is still insufficient.

### 5. YOLO Detections CSV

Inference exports per-frame detections into CSV with frame, camera, class, confidence, and bbox fields.

### 6. Observation3D JSONL

Observation3D records are built from YOLO detections. On train/val, detections can be matched to GT using IoU and class match. On test, the builder must run without GT and depth.

### 7. Future 3D Estimator

The future 3D estimator should consume RGB crops/detections plus calibration/map features and predict 3D observation attributes. Depth remains auxiliary supervision during train/val, not a required test input.

### 8. Future ReID

ReID embeddings are needed for long-term identity continuity and MTMC association.

### 9. Future Deep-OC-SORT3D

The tracker should adapt Deep-OC-SORT logic to 3D/BEV observations, global IDs, 3D motion, and multi-camera consistency.

### 10. Future MTMC Global Association

Global association should merge camera-local tracks into shared global object IDs across cameras.

### 11. Future track1.txt Export

Final output formatting is not implemented yet. It should only be added after detector and 3D observations are stable.

