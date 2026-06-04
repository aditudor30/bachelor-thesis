"""Evaluation helpers for raw versus stabilized pseudo-3D predictions."""

from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.audit3d.audit3d_io import numeric_stats
from deep_oc_sort_3d.pseudo3d.pseudo3d_eval import evaluate_prediction_dicts
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import flatten_prediction_dict
from deep_oc_sort_3d.pseudo3d.pseudo3d_smoothness import audit_pseudo3d_smoothness
from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilization_io import outputs_to_dicts
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput, pseudo3d_output_to_dict


def compare_raw_and_stabilized(raw_outputs: List[Pseudo3DOutput], stabilized_outputs: List[Pseudo3DOutput]) -> Dict[str, Any]:
    """Compare raw and stabilized outputs record-by-record."""
    raw_lookup = {_output_key(output): output for output in raw_outputs}
    center_deltas = []
    depth_deltas = []
    confidence_deltas = []
    changed = 0
    source_counts = {}
    for output in stabilized_outputs:
        raw = raw_lookup.get(_output_key(output))
        if raw is None:
            continue
        center_delta = _center_delta(raw, output)
        depth_delta = _depth_delta(raw, output)
        confidence_delta = float(output.confidence_3d) - float(raw.confidence_3d)
        center_deltas.append(center_delta)
        depth_deltas.append(depth_delta)
        confidence_deltas.append(confidence_delta)
        if _is_changed(center_delta, depth_delta, confidence_delta, raw, output):
            changed += 1
        source_key = "%s->%s|%s->%s" % (raw.center_3d_source, output.center_3d_source, raw.depth_source, output.depth_source)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
    total = min(len(raw_outputs), len(stabilized_outputs))
    return {
        "num_records": total,
        "num_changed": changed,
        "changed_ratio": float(changed) / float(total) if total else None,
        "center_delta": numeric_stats(center_deltas),
        "depth_delta": numeric_stats(depth_deltas),
        "confidence_delta": numeric_stats(confidence_deltas),
        "source_change_counts": source_counts,
    }


def evaluate_stabilized_against_gt(stabilized_outputs: List[Pseudo3DOutput], gt_records: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate stabilized outputs against a GT lookup."""
    rows = [flatten_prediction_dict(pseudo3d_output_to_dict(output)) for output in stabilized_outputs]
    return evaluate_prediction_dicts(rows, gt_records)


def compute_stabilized_smoothness(stabilized_outputs: List[Pseudo3DOutput]) -> Dict[str, Any]:
    """Compute the standard pseudo-3D smoothness audit for stabilized outputs."""
    rows = [flatten_prediction_dict(row) for row in outputs_to_dicts(stabilized_outputs)]
    config = {
        "suspicious_step_m": 3.0,
        "invalid_step_m": 6.0,
        "suspicious_dimension_cv": 0.25,
        "invalid_dimension_cv": 0.50,
        "yaw_jump_threshold": 1.57,
    }
    return audit_pseudo3d_smoothness(rows, config)


def comparison_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten comparison stats to a compact CSV-friendly form."""
    rows = []
    for metric in ("center_delta", "depth_delta", "confidence_delta"):
        stats = summary.get(metric, {})
        if isinstance(stats, dict):
            row = {"metric": metric}
            row.update(stats)
            rows.append(row)
    return rows


def _output_key(output: Pseudo3DOutput) -> Any:
    return (
        output.subset,
        output.scene_name,
        output.camera_id,
        int(output.frame_id),
        output.local_track_id,
        output.global_track_id,
        output.candidate_id,
        int(output.class_id),
    )


def _center_delta(raw: Pseudo3DOutput, output: Pseudo3DOutput) -> Any:
    if raw.center_3d is None or output.center_3d is None:
        return None
    raw_center = np.asarray(raw.center_3d, dtype=float).reshape(-1)
    center = np.asarray(output.center_3d, dtype=float).reshape(-1)
    if raw_center.size < 3 or center.size < 3:
        return None
    return float(np.linalg.norm(center[:3] - raw_center[:3]))


def _depth_delta(raw: Pseudo3DOutput, output: Pseudo3DOutput) -> Any:
    if raw.depth is None or output.depth is None:
        return None
    return abs(float(output.depth) - float(raw.depth))


def _is_changed(center_delta: Any, depth_delta: Any, confidence_delta: float, raw: Pseudo3DOutput, output: Pseudo3DOutput) -> bool:
    if center_delta is not None and float(center_delta) > 1e-6:
        return True
    if depth_delta is not None and float(depth_delta) > 1e-6:
        return True
    if abs(float(confidence_delta)) > 1e-9:
        return True
    return raw.center_3d_source != output.center_3d_source or raw.depth_source != output.depth_source or raw.yaw_source != output.yaw_source
