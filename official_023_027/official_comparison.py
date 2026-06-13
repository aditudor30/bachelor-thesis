"""Compare official V2/V3 candidates and build upload readiness."""

from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.official_023_027.official_config import (
    VARIANT_KEYS,
    frozen_output_root,
    frozen_variant_root,
    output_root,
    scene_ids,
)
from deep_oc_sort_3d.official_023_027.official_track1_io import read_json, read_track1_rows, write_csv, write_json


OFFICIAL_CLASS_NAMES = {0: "Person", 1: "Forklift", 2: "NovaCarter", 3: "Transporter", 4: "FourierGR1T2", 5: "AgilityDigit", 6: "PalletTruck"}


def compare_official_candidates(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Write official comparison, readiness and verdict artifacts."""
    summaries = {}
    per_track_rows = []
    for variant in VARIANT_KEYS:
        summary, tracks = _summarize_candidate(config, variant, progress)
        summaries[variant] = summary
        per_track_rows.extend(tracks)
    comparison_root = output_root(config) / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    metrics = _metric_rows(summaries.get("v2_current", {}), summaries.get("v3_gap_aware_soft", {}))
    readiness = _upload_readiness(config, summaries)
    verdict = _verdict(config, readiness)
    result = {"candidates": summaries, "metrics": metrics, "upload_readiness": readiness, "verdict": verdict}
    write_json(comparison_root / "v2_vs_v3_official_023_027_summary.json", result)
    write_csv(comparison_root / "v2_vs_v3_official_023_027_summary.csv", metrics, ["metric", "v2_current", "v3_gap_aware_soft", "delta_v3_minus_v2", "ratio_v3_over_v2"])
    write_csv(comparison_root / "per_scene_comparison.csv", _breakdown(summaries, "per_scene_rows", "scene_id"), ["scene_id", "v2_current", "v3_gap_aware_soft", "delta_v3_minus_v2", "ratio_v3_over_v2"])
    class_rows = _breakdown(summaries, "per_class_rows", "class_id")
    for row in class_rows:
        row["class_name"] = OFFICIAL_CLASS_NAMES.get(int(row.get("class_id", -1)), "unknown")
    write_csv(comparison_root / "per_class_comparison.csv", class_rows, ["class_id", "class_name", "v2_current", "v3_gap_aware_soft", "delta_v3_minus_v2", "ratio_v3_over_v2"])
    write_csv(comparison_root / "per_track_statistics.csv", per_track_rows, ["variant", "scene_id", "class_id", "class_name", "object_id", "num_rows"])
    write_json(comparison_root / "official_upload_readiness.json", readiness)
    write_json(comparison_root / "verdict.json", verdict)
    frozen_comparison = frozen_output_root(config) / "comparison"
    write_json(frozen_comparison / "upload_readiness.json", readiness)
    write_json(frozen_comparison / "verdict.json", verdict)
    return result


def _summarize_candidate(config: Dict[str, Any], variant: str, progress: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    root = frozen_variant_root(config, variant)
    path = root / "track1.txt"
    rows = read_track1_rows(path, progress=progress)
    validation = read_json(root / "validation_summary.json")
    manifest = read_json(root / "manifest.json")
    tracks = defaultdict(int)
    per_scene = defaultdict(int)
    per_class = defaultdict(int)
    for row in rows:
        tracks[(row.scene_id, row.class_id, row.object_id)] += 1
        per_scene[str(row.scene_id)] += 1
        per_class[str(row.class_id)] += 1
    lengths = sorted(tracks.values())
    summary = {
        "variant": variant,
        "track1_path": str(path),
        "track1_rows": len(rows),
        "unique_tracks": len(tracks),
        "rows_per_track": _numeric_summary(lengths),
        "per_scene_rows": dict(sorted(per_scene.items(), key=lambda item: int(item[0]))),
        "per_class_rows": dict(sorted(per_class.items(), key=lambda item: int(item[0]))),
        "scene_ids": validation.get("scene_ids", []),
        "validation_errors": validation.get("num_errors"),
        "validation_status": validation.get("status"),
        "duplicate_keys": validation.get("checks", {}).get("duplicate_key_count"),
        "nan_inf_count": validation.get("checks", {}).get("nan_or_inf_values"),
        "non_positive_dimensions": validation.get("checks", {}).get("non_positive_dimensions"),
        "rounding_issues": validation.get("checks", {}).get("rounding_issues"),
        "sha256": manifest.get("sha256"),
        "mode": manifest.get("mode"),
    }
    track_rows = [
        {"variant": variant, "scene_id": key[0], "class_id": key[1], "class_name": OFFICIAL_CLASS_NAMES.get(key[1], "unknown"), "object_id": key[2], "num_rows": count}
        for key, count in sorted(tracks.items())
    ]
    return summary, track_rows


def _metric_rows(v2: Dict[str, Any], v3: Dict[str, Any]) -> List[Dict[str, Any]]:
    metrics = {
        "track1_rows": (["track1_rows"]),
        "unique_tracks": (["unique_tracks"]),
        "rows_per_track_mean": (["rows_per_track", "mean"]),
        "rows_per_track_median": (["rows_per_track", "median"]),
        "validation_errors": (["validation_errors"]),
        "duplicate_keys": (["duplicate_keys"]),
        "nan_inf_count": (["nan_inf_count"]),
        "non_positive_dimensions": (["non_positive_dimensions"]),
        "rounding_issues": (["rounding_issues"]),
    }
    output = []
    for metric, keys in metrics.items():
        left = _nested(v2, keys)
        right = _nested(v3, keys)
        output.append({"metric": metric, "v2_current": left, "v3_gap_aware_soft": right, "delta_v3_minus_v2": _delta(left, right), "ratio_v3_over_v2": _ratio(right, left)})
    return output


def _upload_readiness(config: Dict[str, Any], summaries: Dict[str, Any]) -> Dict[str, Any]:
    package_manifest = read_json(frozen_output_root(config) / "packages" / "package_manifest.json")
    package_rows = {str(row.get("variant")): row for row in package_manifest.get("packages", []) if isinstance(row, dict)}
    output = {}
    for variant in VARIANT_KEYS:
        summary = summaries.get(variant, {})
        package = package_rows.get(variant, {})
        name = "v2_current_official" if variant == "v2_current" else "v3_gap_aware_soft_official"
        ready = (
            summary.get("validation_status") == "ok"
            and summary.get("scene_ids") == scene_ids(config)
            and package.get("status") == "ok"
            and package.get("within_size_limit") is True
        )
        output[name] = {
            "ready": ready,
            "track1_path": summary.get("track1_path"),
            "zip_path": package.get("zip_path"),
            "validation_errors": summary.get("validation_errors"),
            "scene_ids": summary.get("scene_ids"),
            "class_mapping": "official",
            "float_rounding_decimals": int(config.get("official_track1", {}).get("round_float_decimals", 2)),
            "zip_size_mb": package.get("zip_size_mb"),
            "zip_within_limit": package.get("within_size_limit"),
            "track1_sha256": summary.get("sha256"),
            "sha256": summary.get("sha256"),
            "zip_sha256": package.get("zip_sha256"),
            "zip_verification": package.get("verification"),
        }
    output["recommendation"] = "Upload V2 official first, then V3 official as a separate submission."
    return output


def _verdict(config: Dict[str, Any], readiness: Dict[str, Any]) -> Dict[str, Any]:
    v2 = readiness.get("v2_current_official", {})
    v3 = readiness.get("v3_gap_aware_soft_official", {})
    scene_audit = read_json(output_root(config) / "audit" / "test_scene_audit.json")
    mapping_audit = read_json(output_root(config) / "audit" / "class_mapping_audit.json")
    compliance = read_json(output_root(config) / "audit" / "compliance_audit.json")
    missing_scenes = scene_audit.get("missing_scenes", [])
    compliance_candidates = compliance.get("candidates", {})
    has_missing_outputs = any(
        value.get("status") == "missing_new_test_scene_outputs" or value.get("all_scene_ids_present") is False
        for value in compliance_candidates.values()
        if isinstance(value, dict)
    )
    if mapping_audit.get("status") != "ok":
        label = "class_mapping_audit_failed"
    elif scene_audit.get("status") != "ok" or missing_scenes or has_missing_outputs:
        label = "missing_new_test_scene_outputs"
    elif v2.get("zip_within_limit") is False or v3.get("zip_within_limit") is False:
        label = "package_size_exceeds_limit"
    elif v2.get("ready") and v3.get("ready"):
        label = "both_official_candidates_ready_for_upload"
    elif v2.get("ready"):
        label = "v2_official_ready_v3_official_not_ready"
    elif v3.get("ready"):
        label = "v3_official_ready_v2_official_not_ready"
    else:
        label = "official_candidates_invalid_fix_required"
    return {
        "label": label,
        "official_winner": None,
        "recommended_action": "Upload V2 official first, then V3 official as a separate submission.",
        "official_evaluation_required": True,
        "test_scene_audit_status": scene_audit.get("status"),
        "class_mapping_audit_status": mapping_audit.get("status"),
        "compliance_audit_status": compliance.get("status"),
    }


def _breakdown(summaries: Dict[str, Any], key: str, output_key: str) -> List[Dict[str, Any]]:
    left = summaries.get("v2_current", {}).get(key, {})
    right = summaries.get("v3_gap_aware_soft", {}).get(key, {})
    keys = sorted(set(list(left.keys()) + list(right.keys())), key=lambda value: int(value))
    return [{output_key: value, "v2_current": int(left.get(value, 0)), "v3_gap_aware_soft": int(right.get(value, 0)), "delta_v3_minus_v2": int(right.get(value, 0)) - int(left.get(value, 0)), "ratio_v3_over_v2": _ratio(right.get(value, 0), left.get(value, 0))} for value in keys]


def _numeric_summary(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "p90": None}
    return {"count": len(values), "mean": mean(values), "median": median(values), "p25": _percentile(values, 0.25), "p75": _percentile(values, 0.75), "p90": _percentile(values, 0.90)}


def _percentile(values: Sequence[int], fraction: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = fraction * float(len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return float(values[low]) * (1.0 - weight) + float(values[high]) * weight


def _nested(data: Dict[str, Any], keys: List[str]) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _delta(left: Any, right: Any) -> Any:
    try:
        return float(right) - float(left)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> Any:
    try:
        denominator_value = float(denominator)
        return None if denominator_value == 0.0 else float(numerator) / denominator_value
    except (TypeError, ValueError):
        return None
