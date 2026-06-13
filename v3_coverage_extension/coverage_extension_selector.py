"""Select a V3.1 candidate only when conservative safety gates pass."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import output_root
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import write_json


def select_final_variant(config: Dict[str, Any], comparison: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    """Evaluate the balanced variant; no other variant may become final."""
    summaries = {str(row.get("variant")): row for row in comparison.get("variants", [])}
    candidate = summaries.get("v3_balanced_coverage_extension")
    rules = config.get("selection", {})
    expected_scenes = sorted(int(value) for value in config.get("official_track1", {}).get("valid_scene_ids", []))
    failures = []
    if candidate is None:
        verdict = "v3_official_remains_best_identity_quality_candidate"
        failures.append("balanced_variant_missing")
    else:
        if int(candidate.get("validation_errors", -1)) != 0:
            failures.append("validation_errors")
        for key in ["duplicate_keys", "nan_inf", "non_positive_dimensions", "rounding_issues"]:
            if int(candidate.get(key, -1)) != 0:
                failures.append(key)
        if sorted(int(value) for value in candidate.get("scene_ids", [])) != expected_scenes:
            failures.append("missing_official_scenes")
        if int(candidate.get("row_gain_vs_v3", 0)) < int(rules.get("min_row_gain_vs_v3", 20000)):
            failures.append("row_gain_too_small")
        multiplier = candidate.get("unique_track_multiplier_vs_v3")
        if multiplier is None or float(multiplier) > float(rules.get("max_unique_track_multiplier_vs_v3", 3.0)):
            failures.append("too_many_unique_tracks")
        balanced_rules = config.get("recovery_rules", {}).get("balanced", {})
        scene_share = candidate.get("target_scene_added_share")
        class_share = candidate.get("target_class_added_share")
        if scene_share is not None and float(scene_share) < float(balanced_rules.get("target_scene_share_min", 0.65)):
            failures.append("target_scene_recovery_share_too_low")
        if class_share is not None and float(class_share) < float(balanced_rules.get("target_class_share_min", 0.75)):
            failures.append("target_class_recovery_share_too_low")
        max_zip_mb = float(config.get("packaging", {}).get("max_zip_size_mb", 50.0))
        track_path = output_root(config) / "variants" / "v3_balanced_coverage_extension" / "track1_official.txt"
        if track_path.exists() and float(track_path.stat().st_size) / (1024.0 * 1024.0) > max_zip_mb:
            failures.append("uncompressed_track1_exceeds_zip_limit")
        if not failures:
            verdict = "v3_coverage_extension_ready_for_upload"
        elif "validation_errors" in failures or any(key in failures for key in ["duplicate_keys", "nan_inf", "non_positive_dimensions", "rounding_issues", "missing_official_scenes"]):
            verdict = "v3_coverage_extension_invalid_fix_required"
        elif "row_gain_too_small" in failures:
            verdict = "v3_coverage_extension_valid_but_small_gain"
        elif "too_many_unique_tracks" in failures:
            verdict = "v3_coverage_extension_valid_but_too_many_tracks"
        else:
            verdict = "v3_coverage_extension_valid_but_quality_risk"
    selected = "v3_balanced_coverage_extension" if candidate is not None and not failures else None
    selected_payload = {"selected_variant": selected, "selection_failures": failures, "candidate_metrics": candidate}
    verdict_payload = {
        "label": verdict, "selected_variant": selected, "reasons": failures,
        "recommendation": "Package V3.1 only when selected_variant is non-null; otherwise retain V3 official as the identity-quality candidate.",
    }
    root = output_root(config) / "comparison"
    write_json(root / "selected_variant.json", selected_payload)
    write_json(root / "verdict.json", verdict_payload)
    return selected, verdict_payload
