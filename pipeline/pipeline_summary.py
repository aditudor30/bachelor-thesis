"""Summary helpers for batch detection-to-observation pipeline runs."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


INFERENCE_FIELDS = [
    "subset",
    "split",
    "scene_name",
    "camera_id",
    "num_frames_processed",
    "num_detections",
    "detections_csv",
    "mot_like_path",
    "status",
    "error_message",
]


OBSERVATION_FIELDS = [
    "subset",
    "split",
    "scene_name",
    "camera_id",
    "detections_csv",
    "observations_jsonl",
    "num_detections",
    "num_observations",
    "matched_gt",
    "unmatched",
    "mean_iou",
    "depth_valid",
    "center_3d_available",
    "per_class_counts_json",
    "status",
    "error_message",
]


PER_CLASS_FIELDS = [
    "subset",
    "class_name",
    "class_id",
    "detections",
    "observations",
    "matched_gt",
    "unmatched",
    "mean_iou",
    "depth_valid",
    "center_3d_available",
]


def write_inference_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write per-camera inference summary rows."""
    _write_csv(rows, path, INFERENCE_FIELDS)


def write_observation_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write per-camera observation summary rows."""
    _write_csv(rows, path, OBSERVATION_FIELDS)


def aggregate_per_class_from_observations(observation_jsonl_paths: List[Path]) -> Dict[str, Any]:
    """Aggregate per-class observation stats by streaming JSONL files."""
    summary = {}
    for path in observation_jsonl_paths:
        subset = _infer_subset(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obs = json.loads(line)
                key = (subset, str(obs.get("class_name")), int(obs.get("class_id", -1)))
                if key not in summary:
                    summary[key] = _empty_class_row(subset, str(obs.get("class_name")), int(obs.get("class_id", -1)))
                _accumulate_class_row(summary[key], obs)
    rows = []
    for row in summary.values():
        if row["_iou_count"] > 0:
            row["mean_iou"] = row["_iou_sum"] / float(row["_iou_count"])
        else:
            row["mean_iou"] = None
        row.pop("_iou_sum")
        row.pop("_iou_count")
        rows.append(row)
    return {"rows": sorted(rows, key=lambda item: (item["subset"], item["class_id"], item["class_name"]))}


def write_per_class_summary(summary: Dict[str, Any], path_csv: Path, path_json: Path) -> None:
    """Write per-class summary to CSV and JSON."""
    rows = list(summary.get("rows", []))
    _write_csv(rows, path_csv, PER_CLASS_FIELDS)
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def aggregate_per_scene_camera_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a compact per-scene/camera summary from observation rows."""
    output = []
    for row in rows:
        output.append(
            {
                "subset": row.get("subset"),
                "split": row.get("split"),
                "scene_name": row.get("scene_name"),
                "camera_id": row.get("camera_id"),
                "num_detections": int(row.get("num_detections", 0)),
                "num_observations": int(row.get("num_observations", 0)),
                "matched_gt": int(row.get("matched_gt", 0)),
                "unmatched": int(row.get("unmatched", 0)),
                "mean_iou": row.get("mean_iou"),
                "depth_valid": int(row.get("depth_valid", 0)),
                "center_3d_available": int(row.get("center_3d_available", 0)),
                "status": row.get("status"),
            }
        )
    return output


def write_per_scene_camera_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write per-scene/camera summary rows."""
    fields = [
        "subset",
        "split",
        "scene_name",
        "camera_id",
        "num_detections",
        "num_observations",
        "matched_gt",
        "unmatched",
        "mean_iou",
        "depth_valid",
        "center_3d_available",
        "status",
    ]
    _write_csv(rows, path, fields)


def print_pipeline_summary(
    inference_rows: Optional[List[Dict[str, Any]]] = None,
    observation_rows: Optional[List[Dict[str, Any]]] = None,
    per_class_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Print a compact pipeline summary."""
    if inference_rows is not None:
        print("Inference summary:")
        print("  cameras: %d" % len(inference_rows))
        print("  detections: %d" % sum(int(row.get("num_detections", 0)) for row in inference_rows))
        print("  errors: %d" % len([row for row in inference_rows if row.get("status") == "error"]))
    if observation_rows is not None:
        print("Observation summary:")
        print("  cameras: %d" % len(observation_rows))
        print("  observations: %d" % sum(int(row.get("num_observations", 0)) for row in observation_rows))
        print("  matched_gt: %d" % sum(int(row.get("matched_gt", 0)) for row in observation_rows))
        print("  errors: %d" % len([row for row in observation_rows if row.get("status") == "error"]))
    if per_class_summary is not None:
        print("Per-class summary:")
        for row in per_class_summary.get("rows", []):
            print(
                "  %s/%s: obs=%d matched=%d unmatched=%d"
                % (
                    row.get("subset"),
                    row.get("class_name"),
                    int(row.get("observations", 0)),
                    int(row.get("matched_gt", 0)),
                    int(row.get("unmatched", 0)),
                )
            )


def _write_csv(rows: List[Dict[str, Any]], path: Path, fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _empty_class_row(subset: str, class_name: str, class_id: int) -> Dict[str, Any]:
    return {
        "subset": subset,
        "class_name": class_name,
        "class_id": int(class_id),
        "detections": 0,
        "observations": 0,
        "matched_gt": 0,
        "unmatched": 0,
        "mean_iou": None,
        "depth_valid": 0,
        "center_3d_available": 0,
        "_iou_sum": 0.0,
        "_iou_count": 0,
    }


def _accumulate_class_row(row: Dict[str, Any], obs: Dict[str, Any]) -> None:
    row["detections"] += 1
    row["observations"] += 1
    if bool(obs.get("matched_gt")):
        row["matched_gt"] += 1
    else:
        row["unmatched"] += 1
    if obs.get("matched_iou") is not None:
        row["_iou_sum"] += float(obs.get("matched_iou"))
        row["_iou_count"] += 1
    if obs.get("depth_value") is not None:
        row["depth_valid"] += 1
    if obs.get("center_3d") is not None:
        row["center_3d_available"] += 1


def _infer_subset(path: Path) -> str:
    parts = list(Path(path).parts)
    if "observations3d" in parts:
        index = parts.index("observations3d")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"
