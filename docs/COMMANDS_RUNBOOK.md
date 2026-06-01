# Commands Runbook

These commands are intended to be run manually on the Linux remote machine. Do not run them automatically from Codex while documenting status.

Set the dataset root as appropriate:

```bash
DATA_ROOT=/home/cl5-student1/PycharmProjects/PoseTrack1/dataset/MTMC_Tracking_2026
```

## 1. Inspect Dataset

```bash
python -m deep_oc_sort_3d.scripts.inspect_dataset \
  --root "$DATA_ROOT"
```

Expected check:

- train/val/test scenes are discovered;
- test missing depth/GT is reported as normal.

## 2. Debug Sample Loader

```bash
python -m deep_oc_sort_3d.scripts.debug_sample_loader \
  --root "$DATA_ROOT" \
  --split train \
  --scene Warehouse_000 \
  --camera-id Camera_0000 \
  --max-frames 3
```

```bash
python -m deep_oc_sort_3d.scripts.debug_sample_loader \
  --root "$DATA_ROOT" \
  --split test \
  --scene Warehouse_023 \
  --camera-id Camera_0000 \
  --max-frames 3
```

Expected check:

- train/val have RGB, depth, GT, calibration;
- test has RGB and calibration, with depth/GT absent as expected.

## 3. Debug Alignment

```bash
python -m deep_oc_sort_3d.scripts.debug_alignment \
  --root "$DATA_ROOT" \
  --split train \
  --scene Warehouse_000 \
  --camera-id Camera_0000 \
  --max-frames 5
```

Expected check:

- visible targets are found for cameras where GT has visible boxes;
- bbox and target counts look plausible.

## 4. Depth Statistics

```bash
python -m deep_oc_sort_3d.scripts.inspect_depth_statistics \
  --root "$DATA_ROOT" \
  --split train \
  --scene Warehouse_000 \
  --camera-id Camera_0000 \
  --max-frames 5
```

Expected check:

- depth unit guess should be `millimeters_likely`;
- use `/1000.0` for meters.

## 5. Export YOLO Dataset Debug

```bash
python -m deep_oc_sort_3d.scripts.export_yolo_dataset \
  --root "$DATA_ROOT" \
  --output output/yolo_dataset_debug \
  --max-frames-per-scene 20 \
  --camera-id Camera_0000
```

Expected check:

- `data.yaml` exists;
- images and labels are created;
- visual labels align with objects.

## 6. Train YOLO Debug

```bash
python -m deep_oc_sort_3d.scripts.train_yolo \
  --data output/yolo_dataset_debug/data.yaml \
  --model yolo11s.pt \
  --epochs 5 \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --workers 2 \
  --project output/yolo_runs \
  --name yolo11s_debug
```

Expected check:

- training starts and writes a run directory;
- this is only a smoke test.

## 7. Run YOLO Inference

```bash
python -m deep_oc_sort_3d.scripts.run_yolo_inference \
  --root "$DATA_ROOT" \
  --split val \
  --scene Warehouse_020 \
  --camera-id Camera_0000 \
  --model output/yolo_runs/yolo11s_debug/weights/best.pt \
  --output output/yolo_detections/Warehouse_020_Camera_0000.csv \
  --max-frames 500 \
  --conf 0.05 \
  --imgsz 960
```

Expected check:

- CSV detections are written;
- visual inspection should show multiple objects when confidence is low enough.

## 8. Build Observation3D

```bash
python -m deep_oc_sort_3d.scripts.build_observations3d \
  --root "$DATA_ROOT" \
  --split val \
  --scene Warehouse_020 \
  --camera-id Camera_0000 \
  --detections output/yolo_detections/Warehouse_020_Camera_0000.csv \
  --output output/observations3d/Warehouse_020_Camera_0000.jsonl \
  --max-frames 500
```

Expected check:

- observations are produced;
- train/val can include GT match diagnostics;
- test should run without GT/depth.

## 9. Audit BBox Scale

```bash
python -m deep_oc_sort_3d.scripts.audit_bbox_scale_visibility \
  --root "$DATA_ROOT" \
  --split train \
  --output output/bbox_audit/train_bbox_audit.csv
```

Expected check:

- CSV contains class, scene, camera, bbox scale, and difficulty fields.

## 10. Export Curriculum Easy

```bash
python -m deep_oc_sort_3d.scripts.export_yolo_curriculum \
  --config deep_oc_sort_3d/configs/yolo_curriculum_easy_allclass.yaml \
  --root "$DATA_ROOT"
```

Expected check:

- `output/yolo_curriculum/easy_allclass/data.yaml`;
- `curriculum_manifest.csv`;
- `curriculum_summary.json`;
- no Person-only frames;
- all expected classes if available after filtering.

## 11. Export Curriculum Medium

```bash
python -m deep_oc_sort_3d.scripts.export_yolo_curriculum \
  --config deep_oc_sort_3d/configs/yolo_curriculum_medium_allclass.yaml \
  --root "$DATA_ROOT"
```

Expected check:

- more diverse than easy;
- still excludes holdout scenes;
- labels inspect cleanly.

## 12. Inspect Curriculum

```bash
python -m deep_oc_sort_3d.scripts.inspect_yolo_curriculum \
  --dataset output/yolo_curriculum/easy_allclass
```

```bash
python -m deep_oc_sort_3d.scripts.inspect_yolo_curriculum \
  --dataset output/yolo_curriculum/medium_allclass
```

Expected check:

- `invalid labels: 0`;
- `missing label files: 0`;
- `duplicate frames: 0`.

## 13. Visualize Curriculum Samples

```bash
python -m deep_oc_sort_3d.scripts.visualize_yolo_curriculum_samples \
  --dataset output/yolo_curriculum/easy_allclass \
  --class-name PalletTruck \
  --max-images 16 \
  --output output/yolo_curriculum/easy_pallettruck_grid.jpg
```

Repeat with:

- `Forklift`
- `Transporter`
- `FourierGR1T2`
- `AgilityDigit`
- `NovaCarter`

Expected check:

- boxes align with objects;
- objects are visible enough to learn;
- false labels are not visually dominant.

## 14. Step 9C: YOLO Curriculum Training

### Smoke Test

```bash
python -m deep_oc_sort_3d.scripts.train_yolo \
  --data output/yolo_curriculum/easy_allclass/data.yaml \
  --model yolo11s.pt \
  --epochs 5 \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --workers 2 \
  --project output/yolo_runs \
  --name yolo11s_easy_curriculum_debug
```

### YOLO11m Easy

```bash
python -m deep_oc_sort_3d.scripts.train_yolo \
  --data output/yolo_curriculum/easy_allclass/data.yaml \
  --model yolo11m.pt \
  --epochs 15 \
  --imgsz 1280 \
  --batch 16 \
  --device 0,1 \
  --workers 4 \
  --project output/yolo_runs \
  --name yolo11m_easy_curriculum
```

### YOLO11m Medium

```bash
python -m deep_oc_sort_3d.scripts.train_yolo \
  --data output/yolo_curriculum/medium_allclass/data.yaml \
  --model output/yolo_runs/yolo11m_easy_curriculum/weights/best.pt \
  --epochs 30 \
  --imgsz 1280 \
  --batch 16 \
  --device 0,1 \
  --workers 4 \
  --project output/yolo_runs \
  --name yolo11m_medium_curriculum
```

