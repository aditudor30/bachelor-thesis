"""Evaluate pseudo-3D predictions with optional GT matching for val/holdout."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.audit3d.audit3d_io import iter_data_files, progress_iter, write_csv, write_json
from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_eval import build_eval_record, summarize_eval_records
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import read_pseudo3d_predictions_csv
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import pseudo3d_eval_record_to_dict


def run(args: Any) -> Dict[str, Any]:
    records = []
    gt_cache = {}
    for path in progress_iter(iter_data_files(args.predictions_root, [".csv"]), args.progress, "evaluate pseudo3D files", "file"):
        predictions = read_pseudo3d_predictions_csv(path)
        for prediction in predictions:
            gt = _match_gt(args.root, prediction, args.subsets, gt_cache)
            records.append(build_eval_record(prediction, gt))
    summary = summarize_eval_records(records)
    rows = [pseudo3d_eval_record_to_dict(record) for record in records]
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json({key: value for key, value in summary.items() if key != "records"}, args.output_root / "summary_eval.json")
    write_csv(rows, args.output_root / "summary_eval.csv")
    print("Evaluated predictions: %s" % summary.get("num_evaluated"))
    return summary


def _match_gt(
    root: Path,
    prediction: Dict[str, Any],
    allowed_subsets: List[str],
    gt_cache: Dict[Any, List[GroundTruthObject]],
) -> Optional[Dict[str, Any]]:
    subset = str(prediction.get("subset", ""))
    if subset not in allowed_subsets:
        return None
    split = str(prediction.get("split") or ("val" if subset == "official_val" else "train"))
    scene = str(prediction.get("scene_name", ""))
    camera_id = str(prediction.get("camera_id", ""))
    scene_paths = get_scene_paths(root, split, scene)
    cache_key = (split, scene)
    if cache_key not in gt_cache:
        if scene_paths.ground_truth_path is None or not scene_paths.ground_truth_path.exists():
            gt_cache[cache_key] = []
        else:
            gt_cache[cache_key] = load_ground_truth_json(scene_paths.ground_truth_path)
    gt_objects = gt_cache[cache_key]
    frame_id = int(float(prediction.get("frame_id", 0)))
    class_name = str(prediction.get("class_name", ""))
    bbox = _bbox(prediction)
    best = None
    best_iou = 0.0
    for obj in gt_objects:
        if obj.frame_id != frame_id:
            continue
        if class_name and obj.object_type and obj.object_type != class_name:
            continue
        gt_bbox = obj.visible_bboxes_2d.get(camera_id)
        if gt_bbox is None:
            continue
        iou = _iou(bbox, gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best = obj
    if best is None or best_iou < 0.1:
        return None
    return _gt_to_dict(best)


def _gt_to_dict(obj: GroundTruthObject) -> Dict[str, Any]:
    return {
        "center_3d": obj.location_3d.tolist(),
        "dimensions_3d": obj.bbox3d_scale.tolist(),
        "yaw": float(obj.bbox3d_rotation[2]) if obj.bbox3d_rotation.size >= 3 else None,
    }


def _bbox(row: Dict[str, Any]) -> Any:
    return (float(row.get("bbox_xyxy", 0.0) or row.get("x1", 0.0)), float(row.get("y1", 0.0)), float(row.get("x2", 0.0)), float(row.get("y2", 0.0)))


def _iou(a: Any, b: Any) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pseudo-3D predictions.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--predictions-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subsets", nargs="+", default=["official_val", "internal_holdout"])
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
