"""Hard-gated selection for ByteTrack coverage tuning."""

from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.bytetrack_tuning.stage_retention_analyzer import compute_retention


def preliminary_variant_ranking(
    variants: Dict[str, Any],
    config: Dict[str, Any],
    baseline_v2: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Rank Phase A variants using local coverage and tracking quality only."""
    rows = []
    baseline = baseline_v2 or {}
    baseline_records = baseline.get("num_records")
    baseline_gt = baseline.get("gt_matched_records")
    for name, values in variants.items():
        local = values.get("local_tracking", {})
        record_retention = compute_retention(local.get("num_records"), baseline_records)
        gt_retention = compute_retention(local.get("gt_matched_records"), baseline_gt)
        score = 0.0
        score += 4.0 * float(record_retention or 0.0)
        score += 3.0 * float(gt_retention or 0.0)
        score += 0.01 * float(local.get("median_track_length") or 0.0)
        score -= 2.0 * float(local.get("short_track_ratio_le3") or 0.0)
        score -= 0.0001 * float(local.get("approx_fragmentation") or 0.0)
        rows.append((score, int(local.get("num_records", 0) or 0), str(name)))
    rows.sort(reverse=True)
    return [name for _score, _records, name in rows]


def select_tuned_variant(metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Select the best full candidate using hard retention gates before soft metrics."""
    baselines = metrics.get("baselines", {})
    baseline = baselines.get("baseline_v2_current", {})
    candidates = []
    all_rows = []
    for name, variant in sorted(metrics.get("variants", {}).items()):
        row = build_selection_row(name, variant, baseline, config)
        all_rows.append(row)
        if row.get("hard_criteria_met"):
            candidates.append(row)
    candidates.sort(key=lambda row: float(row.get("selection_score", -1e12)), reverse=True)
    selected = candidates[0] if candidates else None
    verdict = decide_tuning_verdict(selected, all_rows, baseline, config)
    return {
        "selected_variant": None if selected is None else selected.get("variant"),
        "selected_metrics": selected,
        "variant_selection_rows": all_rows,
        "verdict": verdict,
    }


def build_selection_row(
    name: str,
    variant: Dict[str, Any],
    baseline: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build hard criteria and soft score for one variant."""
    local = variant.get("local_tracking", {})
    global_metrics = variant.get("global_association", {})
    track1 = variant.get("track1", {})
    baseline_local = baseline.get("local_tracking", {})
    baseline_global = baseline.get("global_association", {})
    baseline_track1 = baseline.get("track1", {})
    selection = config.get("selection", {})
    retentions = {
        "local_records_retention": compute_retention(local.get("num_records"), baseline_local.get("num_records")),
        "gt_matched_retention": compute_retention(
            local.get("gt_matched_records"), baseline_local.get("gt_matched_records")
        ),
        "track1_rows_retention": compute_retention(track1.get("rows"), baseline_track1.get("rows")),
        "multi_camera_tracks_retention": compute_retention(
            global_metrics.get("multi_camera_tracks"), baseline_global.get("multi_camera_tracks")
        ),
        "person_records_retention": compute_retention(
            local.get("person_records"), baseline_local.get("person_records")
        ),
        "non_person_records_retention": compute_retention(
            local.get("non_person_records"), baseline_local.get("non_person_records")
        ),
    }
    failures = []
    if variant.get("status") != "ok":
        failures.append("variant_status_not_ok")
    _require_min(failures, retentions, "local_records_retention", selection.get("min_local_records_retention", 0.85))
    _require_min(failures, retentions, "gt_matched_retention", selection.get("min_gt_matched_retention", 0.90))
    _require_min(failures, retentions, "track1_rows_retention", selection.get("min_track1_rows_retention", 0.75))
    _require_min(
        failures,
        retentions,
        "multi_camera_tracks_retention",
        selection.get("min_multi_camera_tracks_retention", 0.60),
    )
    validation_errors = track1.get("validation_errors")
    if bool(selection.get("require_track1_errors_zero", True)) and validation_errors not in (0, 0.0, "0"):
        failures.append("track1_validation_errors")
    purity_delta = _delta(baseline_global.get("global_purity_mean"), global_metrics.get("global_purity_mean"))
    false_merge_delta = _delta(baseline_global.get("false_merge_rate"), global_metrics.get("false_merge_rate"))
    if purity_delta is not None and purity_delta < -float(selection.get("max_allowed_purity_drop", 0.01)):
        failures.append("global_purity_drop")
    if false_merge_delta is not None and false_merge_delta > float(selection.get("max_allowed_false_merge_rate_delta", 0.01)):
        failures.append("false_merge_rate_increase")
    score = _soft_score(local, global_metrics, baseline_local, baseline_global, retentions)
    row = {
        "variant": name,
        "status": variant.get("status"),
        "hard_criteria_met": len(failures) == 0,
        "hard_failures": failures,
        "selection_score": score,
        "runtime_seconds": variant.get("runtime_seconds"),
        "track1_validation_errors": validation_errors,
        "purity_delta": purity_delta,
        "false_merge_rate_delta": false_merge_delta,
    }
    row.update(retentions)
    row.update(
        {
            "local_records": local.get("num_records"),
            "gt_matched_records": local.get("gt_matched_records"),
            "median_track_length": local.get("median_track_length"),
            "short_track_ratio_le3": local.get("short_track_ratio_le3"),
            "local_fragmentation": local.get("approx_fragmentation"),
            "local_id_switches": local.get("approx_id_switches"),
            "global_tracks": global_metrics.get("global_tracks"),
            "multi_camera_tracks": global_metrics.get("multi_camera_tracks"),
            "global_purity_mean": global_metrics.get("global_purity_mean"),
            "false_merge_rate": global_metrics.get("false_merge_rate"),
            "global_fragmentation": global_metrics.get("fragmentation_approx"),
            "person_fragmentation": global_metrics.get("person_fragmentation"),
            "non_person_fragmentation": global_metrics.get("non_person_fragmentation"),
            "track1_rows": track1.get("rows"),
        }
    )
    return row


def decide_tuning_verdict(
    selected: Optional[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return one honest Step 21C verdict."""
    completed = [row for row in rows if row.get("track1_rows") not in (None, 0, "0")]
    if not completed:
        return {"label": "bytetrack_tuning_invalid_fix_required", "reasons": ["no_completed_full_candidate"]}
    if selected is None:
        false_merge_only = any("false_merge_rate_increase" in row.get("hard_failures", []) for row in completed)
        if false_merge_only:
            return {
                "label": "bytetrack_tuned_valid_but_false_merges_too_high",
                "reasons": ["no_variant_passed_false_merge_and_coverage_gates"],
            }
        return {
            "label": "bytetrack_tuned_valid_but_still_low_coverage",
            "reasons": ["no_variant_passed_all_hard_retention_gates"],
        }
    baseline_local = baseline.get("local_tracking", {})
    local_gain = (
        float(selected.get("median_track_length") or 0.0) > float(baseline_local.get("median_track_length") or 0.0)
        and float(selected.get("short_track_ratio_le3") or 1.0)
        < float(baseline_local.get("short_track_ratio_le3") or 1.0)
    )
    global_fragmentation = selected.get("global_fragmentation")
    baseline_fragmentation = baseline.get("global_association", {}).get("fragmentation_approx")
    if local_gain and _less(global_fragmentation, baseline_fragmentation):
        return {
            "label": "bytetrack_tuned_ready_for_full_submission_candidate",
            "reasons": ["hard_coverage_gates_passed", "local_and_global_tracking_improved"],
        }
    if local_gain:
        return {
            "label": "bytetrack_tuned_valid_coverage_recovered_needs_global_tuning",
            "reasons": ["hard_coverage_gates_passed", "local_gain_needs_global_propagation"],
        }
    return {
        "label": "bytetrack_tuned_no_better_than_v2_current",
        "reasons": ["coverage_passed_without_clear_tracking_gain"],
    }


def _soft_score(
    local: Dict[str, Any],
    global_metrics: Dict[str, Any],
    baseline_local: Dict[str, Any],
    baseline_global: Dict[str, Any],
    retentions: Dict[str, Any],
) -> float:
    score = 0.0
    score += 5.0 * float(retentions.get("track1_rows_retention") or 0.0)
    score += 4.0 * float(retentions.get("multi_camera_tracks_retention") or 0.0)
    score += 3.0 * float(retentions.get("local_records_retention") or 0.0)
    score += 2.0 * float(retentions.get("gt_matched_retention") or 0.0)
    score += 1.0 * float(retentions.get("non_person_records_retention") or 0.0)
    score += 1.0 * float(retentions.get("person_records_retention") or 0.0)
    score += 0.02 * max(
        0.0,
        float(local.get("median_track_length") or 0.0) - float(baseline_local.get("median_track_length") or 0.0),
    )
    score -= 5.0 * max(
        0.0,
        float(local.get("short_track_ratio_le3") or 0.0) - float(baseline_local.get("short_track_ratio_le3") or 0.0),
    )
    score += 2.0 * float(global_metrics.get("global_purity_mean") or 0.0)
    score -= 2.0 * float(global_metrics.get("false_merge_rate") or 0.0)
    return score


def _require_min(failures: List[str], values: Dict[str, Any], key: str, minimum: Any) -> None:
    value = values.get(key)
    if value is None or float(value) < float(minimum):
        failures.append("%s_below_minimum" % key)


def _delta(left: Any, right: Any) -> Optional[float]:
    try:
        return float(right) - float(left)
    except (TypeError, ValueError):
        return None


def _less(left: Any, right: Any) -> bool:
    try:
        return float(left) < float(right)
    except (TypeError, ValueError):
        return False
