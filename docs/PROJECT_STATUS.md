# Project Status

## Goal

This repository is being extended from the original Deep-OC-SORT 2D tracker toward a trainable 3D MTMC pipeline for AI City / PhysicalAI SmartSpaces 2026.

The final target is a Deep-OC-SORT-inspired 3D multi-camera tracker that uses:

- YOLO as the 2D detector front-end.
- RGB, calibration, and map data at test time.
- Depth maps only during train/val for supervision and diagnostics.
- Observation3D records as the bridge between 2D detection and future 3D tracking.
- ReID embeddings and global multi-camera association in later stages.

We are not building a custom detector from scratch. YOLO remains the 2D detector.

## Current Architecture

```text
MTMC_Tracking_2026
  -> dataset parsers/loaders
  -> RGB/depth/GT/calibration validation
  -> YOLO dataset export
  -> YOLO training/inference
  -> YOLO detections CSV
  -> Observation3D JSONL
  -> future 3D estimator
  -> future ReID
  -> future Deep-OC-SORT3D association
  -> future global MTMC IDs
```

## Implemented Steps

| Step | Status | Notes |
| --- | --- | --- |
| Step 1: dataset inspection and parsers | Done, manually validated | Structure, calibration, and ground truth parsing work. |
| Step 2: real RGB/depth/GT/calibration loader | Done, manually validated | RGB video and H5 depth are read lazily. |
| Step 3: RGB-depth-GT-calibration alignment and target builder | Done | 3D targets and bbox visibility are available. |
| Step 4: depth quality and robust sampling | Done | Depth is millimeters-like and converted to meters. Default sampling is `center_median`. |
| Step 5: YOLO 2D front-end | Done | YOLO export, training wrapper, inference wrapper, CSV export, and visualizations exist. |
| Step 6: YOLO CSV to Observation3D JSONL | Done | GT matching works on train/val; test runs without GT/depth. |
| Step 7: YOLO class audit and balanced export | Done | Train has all classes; official val is incomplete for some classes. |
| Step 8: all-class split and internal holdout planning | Conceptually built | Internal holdout is diagnostic, not an official score. |
| Step 9A: bbox scale/visibility/difficulty audit | Done | Hard/small boxes are a major detector issue. |
| Step 9B: curriculum export | Done | `easy_allclass` and `medium_allclass` exports are functional. |

## Validated So Far

- Dataset structure for train/val/test.
- Ground truth parsing with real keys containing spaces.
- Calibration parsing.
- Lazy RGB video frame reading.
- Lazy H5 depth frame reading.
- Depth frame structure where each H5 dataset may represent one frame.
- Depth unit is likely millimeters and should be divided by 1000 for meters.
- `center_median` is the default robust depth sampling method.
- Target builder exports 3D targets and diagnostics.
- YOLO label export and visualization.
- YOLO inference CSV generation.
- Observation3D JSONL generation.
- BBox audit CSV and class-rich frame CSV generation.
- Curriculum export for easy and medium all-class datasets.

## Not Implemented Yet

- Neural 3D estimator used in the final detection-to-3D pipeline.
- ReID embedding model for MTMC identity continuity.
- Deep-OC-SORT3D association logic.
- Global multi-camera ID assignment.
- Final `track1.txt` export.
- Full final training/evaluation recipe.

## Current YOLO Status

The main bottleneck is still detector quality, especially for non-Person classes. Earlier YOLO experiments showed weak recall for Forklift and PalletTruck. The current response is to train with a curriculum:

- first on clean/easy all-class samples;
- then on medium all-class samples;
- then evaluate on official val and internal holdout.

## Next Concrete Step

Step 9C: controlled YOLO curriculum training.

The planned order is:

1. Smoke test YOLO11s on `easy_allclass`.
2. Train YOLO11m on `easy_allclass`.
3. Evaluate on official val and internal holdout.
4. Fine-tune YOLO11m on `medium_allclass`.
5. Run confidence sweeps per class.
6. Compare with the previous detector.
7. Decide whether the detector is good enough to move to 3D estimator/ReID.

