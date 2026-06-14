"""Local geometry matcher used to diagnose convention failures."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.official_failure_audit.box3d_utils import axis_aligned_iou3d, center_distance, yaw_error
from deep_oc_sort_3d.official_failure_audit.track1_parser import AuditTrack1Row


GroupKey = Tuple[int, int, int]


def evaluate_predictions(
    predictions: Sequence[AuditTrack1Row], ground_truth: Sequence[AuditTrack1Row],
    config: Dict[str, Any], hypothesis_name: str = "original",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rules = config.get("matching", {})
    thresholds = [float(value) for value in rules.get("distance_thresholds_m", [0.5, 1.0, 2.0, 5.0, 10.0])]
    max_distance = float(rules.get("max_center_distance_m", 10.0))
    reject_ambiguous = bool(rules.get("reject_ambiguous", True))
    ambiguity_margin = float(rules.get("ambiguity_distance_margin_m", 0.05))
    pred_groups = _group(predictions)
    gt_groups = _group(ground_truth)
    details: List[Dict[str, Any]] = []
    nearest_distances: List[float] = []
    matched_distances: List[float] = []
    width_ratios: List[float] = []
    length_ratios: List[float] = []
    height_ratios: List[float] = []
    yaw_errors: List[float] = []
    ious: List[float] = []
    matched_pred_ids = set()
    ambiguous_count = 0

    for key, pred_rows in pred_groups.items():
        gt_rows = gt_groups.get(key, [])
        ambiguous = set()
        pairs = []
        for pred_index, pred in enumerate(pred_rows):
            candidates = sorted(
                [(center_distance(pred, gt), gt_index) for gt_index, gt in enumerate(gt_rows)],
                key=lambda item: item[0],
            )
            if candidates:
                nearest_distances.append(float(candidates[0][0]))
                if reject_ambiguous and len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= ambiguity_margin:
                    ambiguous.add(pred_index)
                    ambiguous_count += 1
                for distance, gt_index in candidates:
                    pairs.append((distance, pred_index, gt_index))
        used_pred = set()
        used_gt = set()
        for distance, pred_index, gt_index in sorted(pairs, key=lambda item: item[0]):
            if distance > max_distance:
                break
            if pred_index in used_pred or gt_index in used_gt or pred_index in ambiguous:
                continue
            pred = pred_rows[pred_index]
            gt = gt_rows[gt_index]
            used_pred.add(pred_index)
            used_gt.add(gt_index)
            identity = id(pred)
            matched_pred_ids.add(identity)
            matched_distances.append(float(distance))
            ratios = _dimension_ratios(pred, gt)
            width_ratios.append(ratios[0])
            length_ratios.append(ratios[1])
            height_ratios.append(ratios[2])
            yaw_value = yaw_error(pred.yaw, gt.yaw)
            yaw_errors.append(yaw_value)
            iou = axis_aligned_iou3d(pred, gt)
            if iou is not None:
                ious.append(iou)
            details.append({
                "hypothesis": hypothesis_name, "matched": True, "scene_id": pred.scene_id,
                "frame_id": pred.frame_id, "class_id": pred.class_id, "pred_object_id": pred.object_id,
                "gt_object_id": gt.object_id, "center_error": float(distance),
                "nearest_gt_distance": float(distance), "width_ratio_pred_over_gt": ratios[0],
                "length_ratio_pred_over_gt": ratios[1], "height_ratio_pred_over_gt": ratios[2],
                "yaw_error": yaw_value, "iou3d_proxy": iou,
                "pred_xyz": [pred.x, pred.y, pred.z], "gt_xyz": [gt.x, gt.y, gt.z],
                "pred_dimensions": [pred.width, pred.length, pred.height],
                "gt_dimensions": [gt.width, gt.length, gt.height],
            })

    for pred in predictions:
        if id(pred) in matched_pred_ids:
            continue
        nearest = _nearest_for_row(pred, gt_groups.get((pred.scene_id, pred.frame_id, pred.class_id), []))
        details.append({
            "hypothesis": hypothesis_name, "matched": False, "scene_id": pred.scene_id,
            "frame_id": pred.frame_id, "class_id": pred.class_id, "pred_object_id": pred.object_id,
            "gt_object_id": None, "center_error": None, "nearest_gt_distance": nearest,
            "width_ratio_pred_over_gt": None, "length_ratio_pred_over_gt": None,
            "height_ratio_pred_over_gt": None, "yaw_error": None, "iou3d_proxy": None,
            "pred_xyz": [pred.x, pred.y, pred.z], "gt_xyz": None,
            "pred_dimensions": [pred.width, pred.length, pred.height], "gt_dimensions": None,
        })

    total_predictions = len(predictions)
    matched = len(matched_distances)
    summary: Dict[str, Any] = {
        "hypothesis": hypothesis_name, "num_predictions": total_predictions, "num_gt": len(ground_truth),
        "num_matches": matched, "match_rate": _rate(matched, total_predictions),
        "gt_recall_at_10m": _rate(matched, len(ground_truth)), "ambiguous_predictions_rejected": ambiguous_count,
        "predictions_without_same_scene_frame_class_gt": sum(1 for row in details if not row["matched"] and row["nearest_gt_distance"] is None),
        "nearest_distance_mean": _mean(nearest_distances), "nearest_distance_median": _percentile(nearest_distances, 50),
        "nearest_distance_p90": _percentile(nearest_distances, 90), "nearest_distance_p95": _percentile(nearest_distances, 95),
        "center_error_mean": _mean(matched_distances), "center_error_median": _percentile(matched_distances, 50),
        "center_error_p90": _percentile(matched_distances, 90), "center_error_p95": _percentile(matched_distances, 95),
        "dimension_ratio_width_median": _percentile(width_ratios, 50),
        "dimension_ratio_length_median": _percentile(length_ratios, 50),
        "dimension_ratio_height_median": _percentile(height_ratios, 50),
        "yaw_error_median": _percentile(yaw_errors, 50), "yaw_error_p90": _percentile(yaw_errors, 90),
        "iou3d_proxy_mean": _mean(ious), "iou3d_proxy_median": _percentile(ious, 50),
        "iou3d_proxy_match_rate_at_0_1": _rate(sum(1 for value in ious if value >= 0.1), total_predictions),
        "iou3d_proxy_match_rate_at_0_25": _rate(sum(1 for value in ious if value >= 0.25), total_predictions),
    }
    for threshold in thresholds:
        key = _threshold_key(threshold)
        summary["match_rate_at_%s" % key] = _rate(sum(1 for value in matched_distances if value <= threshold), total_predictions)
    summary["per_scene_match_rate"] = _group_rates(predictions, details, "scene_id")
    summary["per_class_match_rate"] = _group_rates(predictions, details, "class_id")
    summary["per_frame_match_rate"] = _frame_rates(predictions, details)
    return summary, details


def summary_row(summary: Dict[str, Any], category: str, operations: Dict[str, str]) -> Dict[str, Any]:
    row = dict(summary)
    row["category"] = category
    row["operations"] = dict(operations)
    return row


def _group(rows: Sequence[AuditTrack1Row]) -> Dict[GroupKey, List[AuditTrack1Row]]:
    output: Dict[GroupKey, List[AuditTrack1Row]] = defaultdict(list)
    for row in rows:
        output[(row.scene_id, row.frame_id, row.class_id)].append(row)
    return output


def _dimension_ratios(pred: AuditTrack1Row, gt: AuditTrack1Row) -> Tuple[float, float, float]:
    return (
        pred.width / gt.width if gt.width > 0.0 else float("nan"),
        pred.length / gt.length if gt.length > 0.0 else float("nan"),
        pred.height / gt.height if gt.height > 0.0 else float("nan"),
    )


def _nearest_for_row(pred: AuditTrack1Row, gt_rows: Sequence[AuditTrack1Row]) -> Any:
    values = [center_distance(pred, gt) for gt in gt_rows]
    return min(values) if values else None


def _group_rates(
    predictions: Sequence[AuditTrack1Row], details: Sequence[Dict[str, Any]], field: str,
) -> Dict[str, Any]:
    totals: Dict[str, int] = defaultdict(int)
    matches: Dict[str, int] = defaultdict(int)
    for row in predictions:
        totals[str(getattr(row, field))] += 1
    for row in details:
        if row.get("matched"):
            matches[str(row.get(field))] += 1
    return {key: _rate(matches.get(key, 0), total) for key, total in sorted(totals.items())}


def _frame_rates(
    predictions: Sequence[AuditTrack1Row], details: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    totals: Dict[str, int] = defaultdict(int)
    matches: Dict[str, int] = defaultdict(int)
    for row in predictions:
        totals["%d:%d" % (row.scene_id, row.frame_id)] += 1
    for row in details:
        if row.get("matched"):
            matches["%s:%s" % (row.get("scene_id"), row.get("frame_id"))] += 1
    return {key: _rate(matches.get(key, 0), total) for key, total in sorted(totals.items())}


def _threshold_key(value: float) -> str:
    return ("%g" % value).replace(".", "_") + ("m" if value >= 1.0 else "m")


def _rate(numerator: int, denominator: int) -> Any:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _mean(values: Sequence[float]) -> Any:
    finite = _finite(values)
    return float(np.mean(finite)) if finite.size else None


def _percentile(values: Sequence[float], percentile: float) -> Any:
    finite = _finite(values)
    return float(np.percentile(finite, percentile)) if finite.size else None


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]
