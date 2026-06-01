# Experiment Log

This log records high-level project progress. It intentionally avoids full raw outputs.

## Step 1: Dataset Inspection

What was done:

- Parsed dataset split/scene structure.
- Parsed `calibration.json`.
- Parsed `ground_truth.json`.
- Added dummy tests for structure and schema.

Result:

- Dataset inspection and parsers worked manually.

Conclusion:

- The project can reliably discover scenes and parse train/val metadata.

## Step 2: Sample Loader

What was done:

- Added lazy RGB video frame loading.
- Added lazy H5 depth loading.
- Added sample loader for RGB/depth/GT/calibration/map.

Result:

- RGB frames and depth frame 0 were read correctly.
- H5 depth structure required recursive dataset handling.

Conclusion:

- Loader works and does not preload large depth files.

## Step 3: Alignment and 3D Targets

What was done:

- Built target objects with class, 3D center, dimensions, yaw, object ID, visible bbox, and depth diagnostics.
- Added visual and numeric alignment scripts.

Result:

- Bboxes align visually with objects.
- Initial backprojection errors were large.

Conclusion:

- Target builder is useful, but backprojection should remain diagnostic for now.

## Step 4: Depth Sampling

What was done:

- Added robust depth sampling methods.
- Added depth unit checks.
- Evaluated sampling methods.

Result:

- Depth is likely in millimeters.
- Convert to meters with `/1000.0`.
- `center_median` was the best default.
- `bottom_center` remains a useful diagnostic alternative.

Conclusion:

- Depth should supervise/diagnose training, but not be required at test time.

## Step 5: YOLO Front-End

What was done:

- Exported GT visible boxes to YOLO labels.
- Added train and inference wrappers.
- Added visual inspection tools.

Result:

- YOLO pipeline runs manually.
- Early debug model mostly detected Person and missed non-Person classes.

Conclusion:

- Detector quality became the main blocker.

## Step 6: Observation3D

What was done:

- Converted YOLO detections CSV to Observation3D JSONL.
- Added train/val GT matching.
- Kept test support without GT/depth.

Result:

- Observation builder matched detections to GT when detector boxes were good.

Conclusion:

- Observation3D logic is healthy; detector quality must improve.

## Step 7: Class Audit

What was done:

- Audited class counts across train and official val.
- Investigated missing and rare classes.

Result:

- Train has all seven classes.
- Official val is incomplete for FourierGR1T2 and AgilityDigit.
- Transporter and NovaCarter are rare in official val.

Conclusion:

- Official val alone is not enough to validate all classes.

## Step 9A: BBox Audit

What was done:

- Audited bbox scale, visibility, difficulty, scene distribution, and camera distribution.
- Generated `train_bbox_audit.csv` and `class_rich_frames.csv`.

Result:

- Many objects are hard/small.
- PalletTruck, Forklift, NovaCarter, and Person have many hard examples.

Conclusion:

- A curriculum is needed before large detector training.

## Step 9B: Curriculum Export

What was done:

- Exported `easy_allclass`.
- Exported `medium_allclass`.
- Added manifest, summary, inspection, visualization, and comparison scripts.

Result:

- Both exports completed manually.
- Easy export had all classes present, no Person-only frames, and holdout scenes excluded.

Conclusion:

- The project is ready for controlled YOLO curriculum training.

## Step 9C: Curriculum Training

Status:

- TODO: run YOLO11s smoke test.
- TODO: run YOLO11m easy curriculum training.
- TODO: evaluate official val and internal holdout.
- TODO: fine-tune on medium curriculum.
- TODO: run confidence sweeps per class.
- TODO: compare against previous detector.

