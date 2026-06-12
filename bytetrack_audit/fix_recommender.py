"""Rule-based Step 21D fix recommendation."""

from typing import Any, Dict, List, Optional


def recommend_fix(
    lifecycle: Dict[str, Any],
    stage_rows: List[Dict[str, Any]],
    gt_result: Dict[str, Any],
    motion_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Choose one honest next-step recommendation from audit evidence."""
    reasons = []
    export_evidence = _instrumented_export_evidence(lifecycle)
    motion_evidence = _motion_evidence(motion_result)
    candidate_retention = _transition(stage_rows, "bytetrack_21c_best", "tracklets", "candidates")
    motion_retention = _transition(stage_rows, "bytetrack_21c_best", "candidates", "motion_clean_candidates")
    global_retention = _transition(stage_rows, "bytetrack_21c_best", "motion_clean_candidates", "global_tracks")
    local_gt = _gt_retention(gt_result, "matched_in_bytetrack_21c_local")
    if export_evidence is True:
        reasons.append("instrumented associated-but-not-exported rate is material")
    if motion_evidence:
        reasons.append("motion rejection is concentrated by gap, jump or bbox-height change")
    if export_evidence is True and motion_evidence:
        verdict = "both_export_and_gap_aware_motion_fix_recommended"
    elif export_evidence is True:
        verdict = "bytetrack_export_policy_fix_recommended"
    elif motion_evidence:
        verdict = "gap_aware_motion_filter_recommended"
    elif candidate_retention is not None and candidate_retention < 0.80:
        verdict = "tracklet_candidate_builder_fix_recommended"
        reasons.append("tracklet-to-candidate retention is below 0.80")
    elif global_retention is not None and global_retention < 0.60:
        verdict = "global_association_retention_fix_recommended"
        reasons.append("motion-clean-to-global proxy is below 0.60")
    elif export_evidence is None and (local_gt is None or local_gt < 0.90):
        verdict = "audit_inconclusive_need_instrumented_rerun"
        reasons.append("artifact-only outputs cannot distinguish association from export-policy loss")
    elif motion_retention is not None and motion_retention < 0.75:
        verdict = "bytetrack_not_root_cause_downstream_filters_dominant"
        reasons.append("local GT retention is high while motion-clean candidate retention is low")
    else:
        verdict = "audit_inconclusive_need_instrumented_rerun"
        reasons.append("no single bottleneck meets a strong recommendation rule")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "evidence": {
            "instrumented_export_policy_evidence": export_evidence,
            "motion_gap_or_jump_evidence": motion_evidence,
            "local_gt_object_frame_retention": local_gt,
            "tracklet_to_candidate_retention": candidate_retention,
            "candidate_to_motion_clean_retention": motion_retention,
            "motion_clean_to_global_proxy": global_retention,
        },
        "recommended_step_21e": _next_step(verdict),
    }


def _instrumented_export_evidence(lifecycle: Dict[str, Any]) -> Optional[bool]:
    instrumented = lifecycle.get("instrumented", {})
    rows = instrumented.get("camera_rows", []) if isinstance(instrumented, dict) else []
    if not rows:
        return None
    associated = sum(int(row.get("detections_associated_to_any_track", 0) or 0) for row in rows)
    missing = sum(int(row.get("associated_but_not_exported_records", 0) or 0) for row in rows)
    return associated > 0 and float(missing) / float(associated) >= 0.05


def _motion_evidence(result: Dict[str, Any]) -> bool:
    rows = [row for row in result.get("gap_rows", []) if row.get("variant_name") == "bytetrack_21c_best"]
    rates = {str(row.get("gap_bucket")): row.get("rejection_rate") for row in rows}
    base = rates.get("gap_0_or_1")
    long_gap = max([float(value) for key, value in rates.items() if key != "gap_0_or_1" and value is not None] or [0.0])
    if base is not None and long_gap >= float(base) + 0.10:
        return True
    bbox_rows = [row for row in result.get("bbox_rows", []) if row.get("variant_name") == "bytetrack_21c_best"]
    high = [row for row in bbox_rows if row.get("bbox_height_delta_bucket") == "delta_ge_50"]
    low = [row for row in bbox_rows if row.get("bbox_height_delta_bucket") == "delta_lt_5"]
    return bool(high and low and float(high[0].get("rejection_rate") or 0.0) >= float(low[0].get("rejection_rate") or 0.0) + 0.10)


def _transition(rows: List[Dict[str, Any]], variant: str, left: str, right: str) -> Optional[float]:
    for row in rows:
        if row.get("variant_name") == variant and row.get("stage_from") == left and row.get("stage_to") == right:
            value = row.get("retention")
            return None if value is None else float(value)
    return None


def _gt_retention(result: Dict[str, Any], stage: str) -> Optional[float]:
    for row in result.get("summary_rows", []):
        if row.get("stage") == stage:
            value = row.get("gt_object_frame_retention")
            return None if value is None else float(value)
    return None


def _next_step(verdict: str) -> str:
    mapping = {
        "bytetrack_export_policy_fix_recommended": "Step 21E: instrumented local export-policy correction",
        "gap_aware_motion_filter_recommended": "Step 21E: gap-aware and class-aware motion filtering calibration",
        "both_export_and_gap_aware_motion_fix_recommended": "Step 21E: local export correction followed by gap-aware motion filtering",
        "tracklet_candidate_builder_fix_recommended": "Step 21E: tracklet/candidate builder retention correction",
        "global_association_retention_fix_recommended": "Step 21E: narrow global-association retention tuning",
        "bytetrack_not_root_cause_downstream_filters_dominant": "Step 21E: downstream filtering correction while retaining ByteTrack",
    }
    return mapping.get(verdict, "Step 21E: run the configured instrumented mini-rerun before changing the pipeline")

