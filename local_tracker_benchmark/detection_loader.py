"""Lazy per-camera loading and inventory of existing YOLO11m detections."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deep_oc_sort_3d.detection2d.yolo_detection_io import read_detections_csv
from deep_oc_sort_3d.observations.observation_io import read_observations_jsonl
from deep_oc_sort_3d.local_tracker_benchmark.tracker_input_types import BenchmarkDetection


def inventory_detection_files(
    pipeline_root: Path,
    scene_selection: Sequence[Tuple[str, str, str]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Inventory available per-camera detection CSV files."""
    rows = []
    warnings = []
    for subset, split, scene_name in scene_selection:
        scene_root = Path(pipeline_root) / "detections2d" / subset / scene_name
        files = sorted(scene_root.glob("*.csv")) if scene_root.exists() else []
        if not files:
            warnings.append("missing detections for %s/%s" % (subset, scene_name))
        for path in files:
            rows.append(
                {
                    "subset": subset,
                    "split": split,
                    "scene_name": scene_name,
                    "camera_id": path.stem,
                    "detections_path": str(path),
                    "observations_path": str(Path(pipeline_root) / "observations3d" / subset / scene_name / (path.stem + ".jsonl")),
                }
            )
    return rows, warnings


def load_camera_detections(
    inventory_row: Dict[str, Any],
    min_confidence: float = 0.001,
) -> List[BenchmarkDetection]:
    """Load one camera and attach GT identity only as diagnostic metadata."""
    detections = read_detections_csv(Path(str(inventory_row["detections_path"])))
    gt_lookup = {}
    if str(inventory_row.get("split", "")) in ("train", "val"):
        gt_lookup = _observation_gt_lookup(Path(str(inventory_row.get("observations_path", ""))))
    output = []
    for detection_id, det in enumerate(detections):
        if float(det.confidence) < float(min_confidence):
            continue
        object_id = gt_lookup.get((int(det.frame_id), int(detection_id)))
        output.append(
            BenchmarkDetection(
                scene_id=int(det.scene_id),
                scene_name=str(det.scene_name),
                subset=str(inventory_row.get("subset", "")),
                split=str(det.split),
                camera_id=str(det.camera_id),
                frame_id=int(det.frame_id),
                detection_id=int(detection_id),
                class_id=int(det.class_id),
                class_name=str(det.class_name),
                confidence=float(det.confidence),
                bbox_xyxy=tuple(float(value) for value in det.bbox_xyxy),
                matched_gt_object_id=object_id,
            )
        )
    return output


def group_detections_by_frame(detections: Sequence[BenchmarkDetection]) -> Dict[int, List[BenchmarkDetection]]:
    """Group camera detections without loading images."""
    grouped = {}  # type: Dict[int, List[BenchmarkDetection]]
    for detection in detections:
        grouped.setdefault(int(detection.frame_id), []).append(detection)
    return grouped


def _observation_gt_lookup(path: Path) -> Dict[Tuple[int, int], Optional[int]]:
    if not path.exists():
        return {}
    result = {}
    for observation in read_observations_jsonl(path):
        result[(int(observation.frame_id), int(observation.detection_id))] = observation.object_id
    return result
