# YOLO Curriculum Step 9C

Step 9C is the controlled YOLO curriculum training stage.

Do not treat this as final tracking. The goal is to determine whether the detector starts producing usable non-Person detections before moving to 3D estimator/ReID/tracking.

## Files

- Config: `deep_oc_sort_3d/configs/yolo_curriculum_9c_plan.yaml`
- Command generator: `deep_oc_sort_3d/scripts/prepare_yolo_curriculum_9c.py`
- Split-wide threshold sweep: `deep_oc_sort_3d/scripts/compare_yolo_split_conf_thresholds.py`

## Generate The Manual Runbook

```bash
python -m deep_oc_sort_3d.scripts.prepare_yolo_curriculum_9c \
  --config deep_oc_sort_3d/configs/yolo_curriculum_9c_plan.yaml \
  --output output/yolo_curriculum_9c_commands.md
```

This only writes commands. It does not train, run inference, or evaluate by itself.

## Stage Order

1. YOLO11s smoke test on `easy_allclass`.
2. YOLO11m warm-up on `easy_allclass`.
3. Run inference for YOLO11m easy on official val and internal holdout.
4. Evaluate YOLO11m easy.
5. Fine-tune YOLO11m on `medium_allclass`.
6. Run inference for YOLO11m medium on official val and internal holdout.
7. Evaluate YOLO11m medium.
8. Run split-wide confidence threshold sweeps.
9. Compare easy vs medium metrics.

## What To Watch

Official val is incomplete for all-class diagnosis, so it is useful but not sufficient.

Internal holdout is diagnostic only, but it should show whether robotic classes are being detected at all.

The most important checks are:

- PalletTruck has matches greater than 0.
- Forklift recall improves above the previous baseline.
- Transporter, NovaCarter, FourierGR1T2, and AgilityDigit produce real detections on internal holdout.
- False positives do not dominate visualizations.
- Person does not degrade massively.

## Expected Decision

If easy/medium training improves non-Person detection, continue toward 3D estimator and ReID.

If it still fails, do not proceed to tracking yet. Investigate label quality, class ambiguity, image size, model size, confidence thresholds, and camera/scene sampling.

