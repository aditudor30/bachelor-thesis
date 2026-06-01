# YOLO Status

## Role of YOLO

YOLO is the 2D detector front-end for the 3D MTMC pipeline. We are not implementing a custom detector from scratch.

The detector produces 2D boxes that later become:

- YOLO detections CSV;
- Observation3D JSONL;
- inputs for future 3D estimation;
- inputs for future ReID and 3D tracking.

## Model Sizes

- YOLO11s: debug and smoke tests.
- YOLO11m: main training target.
- YOLO11l: possible later option if YOLO11m remains insufficient.

## What Worked

- GT visible boxes can be exported to YOLO format.
- YOLO debug datasets can be inspected and visualized.
- YOLO training wrapper works manually.
- YOLO inference exports CSV detections.
- YOLO detections can be matched to GT and converted into Observation3D records.
- Curriculum datasets `easy_allclass` and `medium_allclass` were exported and inspected successfully.

## Why YOLO Was Struggling

The earlier detector struggled mainly because:

- debug exports were too small;
- classes are imbalanced;
- several classes are rare in official val;
- many target objects are small or far away;
- PalletTruck has very small median bbox area;
- official val does not represent all classes;
- Camera_0000 in val is mostly useful for Person/Forklift/PalletTruck.

This means the issue was not only class imbalance. Bbox scale and visibility are also major factors.

## Audits and Fixes Done

### Class Audit

Class audits showed:

- train contains all seven classes;
- official val is incomplete;
- internal holdout is required for robotic classes.

### Balanced Export

Balanced export improved class coverage but did not fully solve detection quality, especially for Forklift and PalletTruck.

### BBox Difficulty Audit

The bbox audit showed:

- PalletTruck, Forklift, NovaCarter, and Person have many hard examples.
- PalletTruck median `area_norm` is approximately `0.00127`.
- Small/far objects are a major detector bottleneck.

### Curriculum Export

Step 9B created:

- `easy_allclass`
- `medium_allclass`

The `easy_allclass` export manually reported:

- all classes present;
- no Person-only frames;
- rare-class frames for all exported images;
- holdout scenes excluded;
- complete labels retained for selected frames.

Important interpretation: `easy_allclass` means easy-selected frames with complete labels, not only easy boxes. Complete labels avoid teaching YOLO that visible non-selected objects are background.

## Next YOLO Step

Step 9C is controlled curriculum training:

1. YOLO11s smoke test on `easy_allclass`.
2. YOLO11m warm-up on `easy_allclass`.
3. Evaluation on official val and internal holdout.
4. YOLO11m fine-tuning on `medium_allclass`.
5. Per-class confidence sweep.
6. Compare against previous YOLO results.

