"""Rank observed pipeline bottlenecks without mixing incompatible units."""

from typing import Any, Dict, List


def rank_bottlenecks(stage_rows: List[Dict[str, Any]], gt_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rank consistent transitions first and diagnostic proxies second."""
    output = []
    for row in stage_rows:
        if row.get("variant_name") != "bytetrack_21c_best":
            continue
        retention = row.get("retention")
        if retention is None:
            continue
        comparison = str(row.get("unit_comparison_type", "diagnostic_only"))
        severity = max(0.0, 1.0 - float(retention))
        confidence = 1.0 if comparison == "consistent" else 0.35
        output.append(
            {
                "bottleneck": "%s -> %s" % (row.get("stage_from"), row.get("stage_to")),
                "stage_from": row.get("stage_from"),
                "stage_to": row.get("stage_to"),
                "retention": retention,
                "drop_ratio": row.get("drop_ratio"),
                "unit_comparison_type": comparison,
                "evidence_confidence": confidence,
                "ranking_score": severity * confidence,
            }
        )
    local_gt = _gt_retention(gt_result, "matched_in_bytetrack_21c_local")
    if local_gt is not None:
        output.append(
            {
                "bottleneck": "GT object-frame -> ByteTrack local export",
                "stage_from": "gt_object_frames",
                "stage_to": "local_records",
                "retention": local_gt,
                "drop_ratio": 1.0 - local_gt,
                "unit_comparison_type": "consistent",
                "evidence_confidence": 1.0,
                "ranking_score": 1.0 - local_gt,
            }
        )
    output.sort(key=lambda row: float(row.get("ranking_score", 0.0)), reverse=True)
    for index, row in enumerate(output):
        row["rank"] = index + 1
    return output


def _gt_retention(result: Dict[str, Any], stage: str) -> Any:
    for row in result.get("summary_rows", []):
        if row.get("stage") == stage:
            return row.get("gt_object_frame_retention")
    return None

