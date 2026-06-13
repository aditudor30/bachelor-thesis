"""Generate missing fit-train pseudo-3D sources without training or GT-derived predictions."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.data.calibration import load_calibration_json
from deep_oc_sort_3d.pipeline.batch_yolo_inference import BatchYoloInferenceRunner
from deep_oc_sort_3d.pipeline.run_config import PipelineRunConfig
from deep_oc_sort_3d.pseudo3d.pseudo3d_config import load_pseudo3d_config
from deep_oc_sort_3d.pseudo3d.pseudo3d_estimator import Pseudo3DEstimator
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import load_scene_camera_calibration, write_pseudo3d_predictions_jsonl
from deep_oc_sort_3d.pseudo3d.pseudo3d_priors import load_pseudo3d_priors
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DInput
from deep_oc_sort_3d.v51_geometry_calibration_refit.source_availability_audit import discover_source_files
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import output_root
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import iter_jsonl, progress_iter, write_json


def ensure_fit_train_sources(
    config: Dict[str, Any], progress: bool = True, overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate only missing Warehouse_000-013 sources using YOLO plus pseudo-3D."""
    root = output_root(config)
    fit_scenes = list(config.get("calibration_splits", {}).get("fit_train", {}).get("scenes", []))
    available, missing = _fit_scene_coverage(config, fit_scenes)
    summary: Dict[str, Any] = {
        "requested_fit_train_scenes": fit_scenes, "available_before_generation": sorted(available),
        "missing_before_generation": missing, "generated": False, "no_training": True,
        "gt_used_for_predictions": False, "depth_used_for_predictions": False,
    }
    rules = config.get("source_generation", {})
    if not missing:
        summary.update({"status": "existing_fit_train_sources_complete", "missing_after_generation": []})
        write_json(root / "audit" / "generated_train_sources_summary.json", summary)
        return summary
    if not bool(rules.get("generate_missing_fit_train_sources", True)):
        summary.update({"status": "missing_fit_train_sources_generation_disabled", "missing_after_generation": missing})
        write_json(root / "audit" / "generated_train_sources_summary.json", summary)
        return summary
    checkpoint, detector_settings = resolve_detector_checkpoint(config)
    if checkpoint is None or not checkpoint.is_file():
        message = "Detector checkpoint required for fit_train source generation was not found. Configure paths.detector_checkpoint or a valid paths.detector_config."
        summary.update({"status": "detector_checkpoint_missing", "error": message, "resolved_checkpoint": None if checkpoint is None else str(checkpoint)})
        write_json(root / "audit" / "generated_train_sources_summary.json", summary)
        if bool(rules.get("fail_if_detector_checkpoint_missing", True)):
            raise FileNotFoundError(message)
        return summary
    pipeline_root = root / "generated_train_pipeline"
    pipeline_config = PipelineRunConfig(
        root=Path(str(config.get("paths", {}).get("dataset_root", ""))),
        output_root=pipeline_root,
        run_name="v51_fit_train_sources",
        detector_model=checkpoint,
        conf_threshold=float(rules.get("detector_conf_threshold", detector_settings.get("conf_threshold", 0.01))),
        imgsz=int(rules.get("detector_imgsz", detector_settings.get("imgsz", 1280))),
        device=str(rules.get("detector_device", detector_settings.get("device", "0"))),
        frame_stride=int(rules.get("frame_stride", detector_settings.get("frame_stride", 1))),
        max_frames=_optional_int(rules.get("max_frames", detector_settings.get("max_frames"))),
        scenes_by_subset={"fit_train": missing}, camera_ids=None,
        split_by_subset={"fit_train": "train"}, build_observations=False,
        export_mot_like=False, iou_threshold=0.3,
        depth_sampling_method="disabled", class_must_match=True,
    )
    inference_rows = BatchYoloInferenceRunner(pipeline_config, overwrite=overwrite, show_progress=progress).run()
    failed_inference = [row for row in inference_rows if row.get("status") == "error"]
    generated_rows = _generate_pseudo3d(config, inference_rows, progress, overwrite)
    generated_scenes = set(row.get("scene_name") for row in generated_rows if row.get("status") in ("ok", "skipped_existing"))
    complete_after, missing_after = _fit_scene_coverage(config, fit_scenes)
    summary.update({
        "status": "ok" if not missing_after and not failed_inference else "incomplete",
        "generated": True, "detector_checkpoint": str(checkpoint),
        "inference_files": len(inference_rows), "inference_errors": failed_inference,
        "pseudo3d_files": len(generated_rows), "generated_scenes": sorted(generated_scenes),
        "complete_fit_train_scenes_after_generation": sorted(complete_after),
        "missing_after_generation": missing_after, "files": generated_rows,
    })
    write_json(root / "audit" / "generated_train_sources_summary.json", summary)
    return summary


def resolve_detector_checkpoint(config: Dict[str, Any]) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Resolve an existing checkpoint from explicit config, pipeline config or manifests."""
    paths = config.get("paths", {})
    explicit = paths.get("detector_checkpoint")
    if explicit:
        path = Path(str(explicit))
        return path, {}
    detector_config = Path(str(paths.get("detector_config", "")))
    settings: Dict[str, Any] = {}
    if detector_config.is_file():
        data = yaml.safe_load(detector_config.read_text(encoding="utf-8")) or {}
        settings = data.get("pipeline", {}) if isinstance(data, dict) else {}
        configured = settings.get("detector_model")
        if configured:
            path = Path(str(configured))
            if path.is_file():
                return path, settings
    candidates = [
        Path("runs/detect/output/yolo_runs/yolo11m_medium_curriculum/weights/best.pt"),
        Path("output/yolo_runs/yolo11m_medium_curriculum/weights/best.pt"),
    ]
    for path in candidates:
        if path.is_file():
            return path, settings
    pipeline_root = Path(str(paths.get("pipeline_runs_root", "output/pipeline_runs")))
    if pipeline_root.is_dir():
        for manifest in sorted(pipeline_root.rglob("run_config_resolved.yaml")):
            data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            configured = data.get("pipeline", {}).get("detector_model") if isinstance(data, dict) else None
            if configured and Path(str(configured)).is_file():
                return Path(str(configured)), data.get("pipeline", {})
    return Path(str(settings.get("detector_model"))) if settings.get("detector_model") else None, settings


def _generate_pseudo3d(
    config: Dict[str, Any], inference_rows: List[Dict[str, Any]], progress: bool, overwrite: bool,
) -> List[Dict[str, Any]]:
    paths = config.get("paths", {})
    dataset_root = Path(str(paths.get("dataset_root", "")))
    priors = load_pseudo3d_priors(Path(str(paths.get("class_priors_path", ""))))
    estimator = Pseudo3DEstimator(priors, load_pseudo3d_config(Path(str(paths.get("pseudo3d_config", "deep_oc_sort_3d/configs/pseudo3d_isolated_debug.yaml")))))
    output_rows: List[Dict[str, Any]] = []
    for row in progress_iter(inference_rows, progress, "V5.1 fit_train pseudo3D sources"):
        if row.get("status") == "error":
            continue
        scene = str(row.get("scene_name"))
        camera = str(row.get("camera_id"))
        target = output_root(config) / "generated_train_sources" / "fit_train" / scene / (camera + ".jsonl")
        if target.is_file() and not overwrite:
            output_rows.append({"scene_name": scene, "camera_id": camera, "path": str(target), "status": "skipped_existing"})
            continue
        detections = read_detections_csv(Path(str(row.get("detections_csv", ""))))
        calibration = load_scene_camera_calibration(dataset_root, "train", scene, camera)
        width = _calibration_int(calibration, "frame_width", "frameWidth")
        height = _calibration_int(calibration, "frame_height", "frameHeight")
        inputs = [
            Pseudo3DInput(
                scene_name=det.scene_name, camera_id=det.camera_id, frame_id=det.frame_id,
                class_id=det.class_id, class_name=det.class_name, bbox_xyxy=det.bbox_xyxy,
                confidence=det.confidence, image_width=width, image_height=height,
                calibration=calibration, subset="fit_train", split="train",
            )
            for det in detections
        ]
        outputs = estimator.estimate_batch(inputs)
        write_pseudo3d_predictions_jsonl(outputs, target)
        output_rows.append({
            "scene_name": scene, "camera_id": camera, "path": str(target), "status": "ok",
            "detections": len(detections), "predictions": len(outputs),
            "successful_predictions": sum(1 for item in outputs if item.center_3d is not None),
        })
    return output_rows


def _calibration_int(calibration: Any, snake: str, camel: str) -> int:
    value = calibration.get(snake, calibration.get(camel, 0)) if isinstance(calibration, dict) else getattr(calibration, snake, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _source_has_independent_prediction(path: Path) -> bool:
    """Reject files whose sampled rows all expose GT-derived geometry."""
    checked = 0
    for row in iter_jsonl(path):
        checked += 1
        explicit_gt = row.get("is_gt_derived") is True or str(row.get("is_gt_derived", "")).lower() in ("1", "true", "yes")
        matched_gt = row.get("matched_gt") is True or str(row.get("matched_gt", "")).lower() in ("1", "true", "yes")
        if not explicit_gt and not matched_gt and row.get("center_3d") is not None:
            return True
        if checked >= 200:
            break
    return False


def _fit_scene_coverage(config: Dict[str, Any], fit_scenes: List[str]) -> Tuple[Set[str], List[str]]:
    files = [item for item in discover_source_files(config) if item[0] == "fit_train" and _source_has_independent_prediction(item[4])]
    cameras_by_scene: Dict[str, Set[str]] = {}
    for _phase, _split, scene, camera, _path in files:
        cameras_by_scene.setdefault(scene, set()).add(camera)
    dataset_root = Path(str(config.get("paths", {}).get("dataset_root", "")))
    complete = set()
    for scene in fit_scenes:
        calibration_path = dataset_root / "train" / scene / "calibration.json"
        expected = set()
        if calibration_path.is_file():
            try:
                expected = set(str(key) for key in load_calibration_json(calibration_path).keys())
            except (OSError, ValueError, TypeError):
                expected = set()
        available = cameras_by_scene.get(scene, set())
        if available and (not expected or expected.issubset(available)):
            complete.add(scene)
    return complete, sorted(set(fit_scenes) - complete)
