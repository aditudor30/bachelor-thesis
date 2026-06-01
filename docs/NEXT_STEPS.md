# Next Steps

## Immediate Next Step: Step 9C

Step 9C is controlled YOLO curriculum training. The goal is not to solve MTMC yet. The goal is to determine whether the detector can produce real usable detections for non-Person classes.

The command plan for this step is defined in:

- `deep_oc_sort_3d/configs/yolo_curriculum_9c_plan.yaml`
- `docs/YOLO_CURRICULUM_9C.md`

## Plan

1. Smoke test YOLO11s on `easy_allclass`.
2. Train YOLO11m warm-up on `easy_allclass`.
3. Evaluate on official val.
4. Evaluate on internal holdout.
5. Fine-tune YOLO11m on `medium_allclass`.
6. Run confidence sweep per class.
7. Compare against the older YOLO model.
8. Decide whether to proceed to 3D estimator/ReID.

## Transition Criteria

Proceed to 3D estimator/ReID only if:

- PalletTruck has matches greater than 0.
- Forklift recall improves significantly above the previous baseline.
- Transporter, NovaCarter, FourierGR1T2, and AgilityDigit produce real detections in internal holdout.
- False positives do not visually dominate.
- Person performance does not degrade massively.

## If Curriculum Training Works

Then continue with:

1. 3D observation estimator from YOLO detections.
2. ReID embedding training/evaluation.
3. Deep-OC-SORT3D single-camera association.
4. MTMC global association.
5. Submission export.

## If Curriculum Training Does Not Work

Investigate:

- label quality;
- class ambiguity;
- image size;
- model size;
- confidence thresholds per class;
- small-object augmentation;
- class-specific camera/scene sampling;
- whether some classes need separate detector fine-tuning or oversampling.

## Rules Going Forward

- Do not use depth as a required test input.
- Use depth only for train/val supervision and diagnostics.
- Keep test pipeline compatible with RGB + calibration + map only.
- Do not modify original Deep-OC-SORT 2D behavior.
- Keep Python 3.9 compatibility.
- Do not use Python 3.10 union syntax.
- Do not use `match/case`.
- Treat official val as incomplete for all-class diagnosis.
- Use internal holdout for robotic class diagnostics.
