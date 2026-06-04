"""Smoothness audit for pseudo-3D predictions."""

from typing import Any, Dict, List

from deep_oc_sort_3d.audit3d.smoothness_3d_audit import compute_smoothness_audit, find_worst_jumps


def pseudo3d_rows_for_smoothness(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert prediction dictionaries to Track1-like rows for smoothness helpers."""
    rows = []
    for row in predictions:
        scene_key = "%s/%s/%s" % (row.get("subset", ""), row.get("scene_name", ""), row.get("camera_id", ""))
        object_key = row.get("global_track_id") or row.get("local_track_id") or row.get("candidate_id")
        rows.append(
            {
                "scene_id": scene_key,
                "class_id": row.get("class_id"),
                "object_id": object_key,
                "frame_id": row.get("frame_id"),
                "x": row.get("center_x"),
                "y": row.get("center_y"),
                "z": row.get("center_z"),
                "width": row.get("width_3d"),
                "length": row.get("length_3d"),
                "height": row.get("height_3d"),
                "yaw": row.get("yaw"),
            }
        )
    return rows


def audit_pseudo3d_smoothness(predictions: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute pseudo-3D smoothness summary."""
    rows = pseudo3d_rows_for_smoothness(predictions)
    return compute_smoothness_audit(rows, config, show_progress=False)


def worst_pseudo3d_jumps(predictions: List[Dict[str, Any]], top_k: int = 100) -> List[Dict[str, Any]]:
    """Return worst pseudo-3D jumps."""
    return find_worst_jumps(pseudo3d_rows_for_smoothness(predictions), top_k=top_k)
