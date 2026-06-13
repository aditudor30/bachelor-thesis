"""Metrics and comparison tables for V2, V3 and V3.1 variants."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import zlib

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row, read_track1_rows
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import VARIANT_NAMES, output_root, variant_root
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import read_json, write_csv, write_json


OPTIONAL_METRICS = [
    "fragmentation_approx", "fragmentation_delta_vs_v3", "fragmentation_delta_vs_v2", "purity",
    "purity_delta_vs_v3", "false_merge_rate", "false_merge_delta_vs_v3", "multi_camera_tracks",
    "multi_camera_delta_vs_v3", "motion_good", "motion_suspicious", "motion_invalid", "step_p95", "step_p99", "jump_ratio",
]


def compute_variant_metrics(
    variant: str,
    rows: Sequence[OfficialTrack1Row],
    v3_rows: Sequence[OfficialTrack1Row],
    v2_rows: Sequence[OfficialTrack1Row],
    validation: Dict[str, Any],
    added: Dict[str, Any],
    zip_size_mb: Any = "not_available",
    config: Any = None,
    manifest: Any = None,
) -> Dict[str, Any]:
    """Compute mandatory Step 22B metrics for one variant."""
    lengths = _track_lengths(rows)
    v3_tracks = len(_track_lengths(v3_rows))
    target_scenes = set(str(value) for value in (config or {}).get("targeting", {}).get("target_scenes", []))
    target_classes = set(str(value) for value in (config or {}).get("targeting", {}).get("target_official_classes", []))
    added_total = max(0, int(added.get("added_rows", 0)))
    target_scene_rows = sum(int(value) for key, value in added.get("added_rows_by_scene", {}).items() if str(key) in target_scenes)
    target_class_rows = sum(int(value) for key, value in added.get("added_rows_by_class", {}).items() if str(key) in target_classes)
    metrics = {
        "variant": variant, "status": validation.get("status", "not_validated"),
        "track1_rows": len(rows), "row_gain_vs_v3": len(rows) - len(v3_rows),
        "row_gain_vs_v2": len(rows) - len(v2_rows),
        "row_ratio_vs_v3": _ratio(len(rows), len(v3_rows)), "row_ratio_vs_v2": _ratio(len(rows), len(v2_rows)),
        "unique_tracks": len(lengths), "unique_track_delta_vs_v3": len(lengths) - v3_tracks,
        "unique_track_multiplier_vs_v3": _ratio(len(lengths), v3_tracks),
        "rows_per_track_mean": float(np.mean(list(lengths.values()))) if lengths else None,
        "rows_per_track_median": _pct(lengths, 50), "rows_per_track_p25": _pct(lengths, 25),
        "rows_per_track_p75": _pct(lengths, 75), "rows_per_track_p90": _pct(lengths, 90),
        "scene_distribution": _distribution(rows, "scene_id"), "class_distribution": _distribution(rows, "class_id"),
        "added_rows_by_scene": added.get("added_rows_by_scene", {}), "added_rows_by_class": added.get("added_rows_by_class", {}),
        "added_rows_by_scene_class": added.get("added_rows_by_scene_class", {}),
        "target_scene_added_share": _ratio(target_scene_rows, added_total),
        "target_class_added_share": _ratio(target_class_rows, added_total),
        "validation_errors": int(validation.get("num_errors", -1)),
        "duplicate_keys": int(validation.get("checks", {}).get("duplicate_key_count", -1)),
        "nan_inf": int(validation.get("checks", {}).get("nan_or_inf_values", -1)),
        "non_positive_dimensions": int(validation.get("checks", {}).get("non_positive_dimensions", -1)),
        "rounding_issues": int(validation.get("checks", {}).get("rounding_issues", -1)),
        "scene_ids": validation.get("scene_ids", []), "zip_size_mb": zip_size_mb,
    }
    for key in OPTIONAL_METRICS:
        metrics[key] = "not_available"
    metrics.update(_motion_metrics((manifest or {}).get("track_manifest", [])))
    return metrics


def build_comparison_outputs(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Collect all existing variants and write comparison JSON/CSV tables."""
    root = output_root(config)
    v2_rows = read_track1_rows(Path(str(config.get("paths", {}).get("v2_official_track1", ""))), progress=progress)
    v3_rows = read_track1_rows(Path(str(config.get("paths", {}).get("v3_official_track1", ""))), progress=progress)
    summaries = []
    variant_rows = {}
    for variant in VARIANT_NAMES:
        vroot = variant_root(config, variant)
        path = vroot / "track1_official.txt"
        if not path.exists():
            continue
        rows = read_track1_rows(path, progress=progress)
        variant_rows[variant] = rows
        validation = read_json(vroot / "validation_summary.json")
        added = read_json(vroot / "added_rows_summary.json")
        manifest = read_json(vroot / "manifest.json")
        summaries.append(compute_variant_metrics(variant, rows, v3_rows, v2_rows, validation, added, zip_size_mb=_estimated_zip_size_mb(path), config=config, manifest=manifest))
    comparison = {"v2_rows": len(v2_rows), "v3_rows": len(v3_rows), "variants": summaries}
    write_json(root / "comparison" / "v3_coverage_extension_summary.json", comparison)
    write_csv(root / "comparison" / "v3_coverage_extension_summary.csv", [_flat_summary(row) for row in summaries])
    write_csv(root / "comparison" / "per_scene_comparison.csv", _comparison_rows(v2_rows, v3_rows, variant_rows, "scene_id"))
    write_csv(root / "comparison" / "per_class_comparison.csv", _comparison_rows(v2_rows, v3_rows, variant_rows, "class_id"))
    write_csv(root / "comparison" / "metric_deltas_vs_v3_official.csv", _delta_rows(summaries, "v3"))
    write_csv(root / "comparison" / "metric_deltas_vs_v2_official.csv", _delta_rows(summaries, "v2"))
    return comparison


def _track_lengths(rows: Sequence[OfficialTrack1Row]) -> Dict[Any, int]:
    values = defaultdict(int)
    for row in rows:
        values[(row.scene_id, row.class_id, row.object_id)] += 1
    return values


def _distribution(rows: Sequence[OfficialTrack1Row], field: str) -> Dict[str, int]:
    values = defaultdict(int)
    for row in rows:
        values[str(getattr(row, field))] += 1
    return dict(sorted(values.items(), key=lambda item: int(item[0])))


def _pct(lengths: Dict[Any, int], value: float) -> Any:
    return float(np.percentile(list(lengths.values()), value)) if lengths else None


def _ratio(numerator: int, denominator: int) -> Any:
    return float(numerator) / float(denominator) if denominator else None


def _estimated_zip_size_mb(path: Path) -> Any:
    """Estimate single-file ZIP size using equivalent DEFLATE compression."""
    if not path.exists():
        return "not_available"
    compressed = zlib.compress(path.read_bytes(), level=9)
    return float(len(compressed)) / (1024.0 * 1024.0)


def _motion_metrics(track_manifest: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    values = [float(row.get("p95_step_distance")) for row in track_manifest if row.get("p95_step_distance") is not None]
    jumps = [float(row.get("jump_ratio")) for row in track_manifest if row.get("jump_ratio") is not None]
    if not values:
        return {}
    return {
        "motion_good": sum(1 for value in values if value <= 12.0),
        "motion_suspicious": sum(1 for value in values if 12.0 < value <= 18.0),
        "motion_invalid": sum(1 for value in values if value > 18.0),
        "step_p95": float(np.percentile(values, 95)),
        "step_p99": float(np.percentile(values, 99)),
        "jump_ratio": float(np.mean(jumps)) if jumps else "not_available",
    }


def _flat_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if not isinstance(value, dict)}


def _comparison_rows(v2: Sequence[OfficialTrack1Row], v3: Sequence[OfficialTrack1Row], variants: Dict[str, Sequence[OfficialTrack1Row]], field: str) -> List[Dict[str, Any]]:
    datasets = {"v2_official": v2, "v3_official": v3}
    datasets.update(variants)
    distributions = {name: _distribution(rows, field) for name, rows in datasets.items()}
    keys = sorted(set(key for values in distributions.values() for key in values.keys()), key=int)
    output = []
    for key in keys:
        row = {field: int(key)}
        for name, values in distributions.items():
            row[name + "_rows"] = values.get(key, 0)
        output.append(row)
    return output


def _delta_rows(summaries: Sequence[Dict[str, Any]], baseline: str) -> List[Dict[str, Any]]:
    output = []
    for row in summaries:
        output.append({
            "variant": row.get("variant"), "baseline": baseline,
            "row_delta": row.get("row_gain_vs_v3") if baseline == "v3" else row.get("row_gain_vs_v2"),
            "row_ratio": row.get("row_ratio_vs_v3") if baseline == "v3" else row.get("row_ratio_vs_v2"),
            "unique_track_delta_vs_v3": row.get("unique_track_delta_vs_v3") if baseline == "v3" else "not_available",
        })
    return output
