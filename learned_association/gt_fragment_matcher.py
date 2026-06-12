"""Conservative fragment-to-ground-truth identity matching."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject, load_ground_truth_json
from deep_oc_sort_3d.learned_association.pair_dataset_io import progress_iter, safe_float, safe_int


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU for two XYXY boxes."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def load_gt_by_scene(config: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[int, List[GroundTruthObject]]]:
    """Load train/val Person GT grouped by scene and frame."""
    dataset_root = Path(str(config.get("paths", {}).get("dataset_root", "")))
    result = {}  # type: Dict[Tuple[str, str], Dict[int, List[GroundTruthObject]]]
    for split_key in ("train", "val"):
        split_config = config.get("splits", {}).get(split_key, {})
        split_name = str(split_config.get("split_name", split_key))
        for scene_name in split_config.get("scenes", []):
            gt_path = dataset_root / split_name / str(scene_name) / "ground_truth.json"
            if not gt_path.is_file():
                result[(split_name, str(scene_name))] = {}
                continue
            objects = load_ground_truth_json(gt_path)
            by_frame = {}  # type: Dict[int, List[GroundTruthObject]]
            for obj in objects:
                if str(obj.object_type).lower() != "person":
                    continue
                by_frame.setdefault(int(obj.frame_id), []).append(obj)
            result[(split_name, str(scene_name))] = by_frame
    return result


def match_fragments_to_gt(
    fragments: Sequence[Dict[str, Any]],
    gt_lookup: Dict[Tuple[str, str], Dict[int, List[GroundTruthObject]]],
    config: Dict[str, Any],
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Match fragments in-place and return serializable match rows."""
    rows = []  # type: List[Dict[str, Any]]
    for fragment in progress_iter(fragments, "GT matching", progress, len(fragments)):
        key = (str(fragment.get("split") or ""), str(fragment.get("scene_name") or ""))
        result = match_fragment_to_gt(fragment, gt_lookup.get(key, {}), config)
        fragment.update(result)
        rows.append(
            {
                "fragment_id": fragment.get("fragment_id"),
                "split": fragment.get("split"),
                "scene_name": fragment.get("scene_name"),
                "camera_id": fragment.get("camera_id"),
                "gt_identity_id": result.get("gt_identity_id"),
                "gt_object_id": result.get("gt_object_id"),
                "gt_match_count": result.get("gt_match_count"),
                "gt_match_ratio": result.get("gt_match_ratio"),
                "gt_purity": result.get("gt_purity"),
                "gt_label_source": result.get("gt_label_source"),
                "valid_for_pairs": result.get("valid_for_pairs"),
                "invalid_reason": result.get("invalid_reason"),
            }
        )
    return rows


def match_fragments_to_gt_from_dataset(
    fragments: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Match fragments while loading only one scene GT file at a time."""
    dataset_root = Path(str(config.get("paths", {}).get("dataset_root", "")))
    grouped = defaultdict(list)  # type: Dict[Tuple[str, str], List[Dict[str, Any]]]
    for fragment in fragments:
        key = (str(fragment.get("split") or ""), str(fragment.get("scene_name") or ""))
        grouped[key].append(fragment)
    rows = []  # type: List[Dict[str, Any]]
    scene_items = sorted(grouped.items())
    for (split, scene_name), scene_fragments in progress_iter(
        scene_items, "GT matching by scene", progress, len(scene_items)
    ):
        gt_path = dataset_root / split / scene_name / "ground_truth.json"
        by_frame = {}  # type: Dict[int, List[GroundTruthObject]]
        if gt_path.is_file():
            for obj in load_ground_truth_json(gt_path):
                if str(obj.object_type).lower() == "person":
                    by_frame.setdefault(int(obj.frame_id), []).append(obj)
        for fragment in scene_fragments:
            result = match_fragment_to_gt(fragment, by_frame, config)
            fragment.update(result)
            rows.append(
                {
                    "fragment_id": fragment.get("fragment_id"),
                    "split": split,
                    "scene_name": scene_name,
                    "camera_id": fragment.get("camera_id"),
                    "gt_identity_id": result.get("gt_identity_id"),
                    "gt_object_id": result.get("gt_object_id"),
                    "gt_match_count": result.get("gt_match_count"),
                    "gt_match_ratio": result.get("gt_match_ratio"),
                    "gt_purity": result.get("gt_purity"),
                    "gt_label_source": result.get("gt_label_source"),
                    "valid_for_pairs": result.get("valid_for_pairs"),
                    "invalid_reason": result.get("invalid_reason"),
                }
            )
        del by_frame
    return rows


def match_fragment_to_gt(
    fragment: Dict[str, Any],
    gt_by_frame: Dict[int, List[GroundTruthObject]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Assign a dominant scene-local GT identity to one fragment."""
    settings = config.get("gt_matching", {})
    min_iou = float(settings.get("min_iou_for_gt_match", 0.3))
    min_ratio = float(settings.get("min_gt_match_ratio", 0.3))
    min_purity = float(settings.get("min_gt_purity", 0.6))
    min_length = int(config.get("fragment_source", {}).get("min_fragment_length", 3))
    min_confidence = float(config.get("fragment_source", {}).get("min_mean_confidence", 0.05))
    camera_id = str(fragment.get("camera_id") or "")
    observations = list(fragment.get("_observations") or [])
    matches = []  # type: List[int]
    comparable = 0
    for observation in observations:
        frame_id = safe_int(observation.get("frame_id"))
        bbox = observation.get("bbox_xyxy")
        if frame_id is None or bbox is None:
            continue
        comparable += 1
        best_object_id = None  # type: Optional[int]
        best_iou = min_iou
        for obj in gt_by_frame.get(frame_id, []):
            gt_bbox = obj.visible_bboxes_2d.get(camera_id)
            if gt_bbox is None:
                continue
            overlap = bbox_iou(bbox, gt_bbox)
            if overlap >= best_iou:
                best_iou = overlap
                best_object_id = int(obj.object_id)
        if best_object_id is not None:
            matches.append(best_object_id)

    label_source = "iou_2d"
    if matches:
        counts = Counter(matches)
        object_id, dominant_count = counts.most_common(1)[0]
        match_count = len(matches)
        match_ratio = match_count / float(max(1, comparable))
        purity = dominant_count / float(match_count)
    else:
        pipeline_object_id = safe_int(fragment.get("_pipeline_gt_object_id"))
        pipeline_purity = safe_float(fragment.get("_pipeline_gt_purity"))
        pipeline_match_count = safe_int(fragment.get("_pipeline_gt_match_count"), 0) or 0
        if pipeline_object_id is not None and pipeline_purity is not None:
            object_id = pipeline_object_id
            purity = pipeline_purity
            match_count = pipeline_match_count
            match_ratio = min(1.0, match_count / float(max(1, int(fragment.get("num_observations") or 0))))
            label_source = "pipeline_gt_summary"
        else:
            object_id = None
            purity = 0.0
            match_count = 0
            match_ratio = 0.0
            label_source = "unmatched"

    invalid_reasons = []  # type: List[str]
    if int(fragment.get("num_observations") or 0) < min_length:
        invalid_reasons.append("fragment_too_short")
    if float(fragment.get("mean_confidence") or 0.0) < min_confidence:
        invalid_reasons.append("confidence_too_low")
    if object_id is None:
        invalid_reasons.append("gt_identity_unknown")
    if match_ratio < min_ratio:
        invalid_reasons.append("gt_match_ratio_too_low")
    if purity < min_purity:
        invalid_reasons.append("gt_purity_too_low")
    valid = not invalid_reasons
    scene_name = str(fragment.get("scene_name") or "")
    return {
        "gt_identity_id": "%s_%d" % (scene_name, object_id) if object_id is not None else "unknown",
        "gt_object_id": object_id,
        "gt_match_count": match_count,
        "gt_match_ratio": match_ratio,
        "gt_purity": purity,
        "gt_label_source": label_source,
        "valid_for_pairs": valid,
        "invalid_reason": "ok" if valid else ";".join(invalid_reasons),
    }
