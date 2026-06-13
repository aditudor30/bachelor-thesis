"""Audit frozen V3.1 geometry before refinement."""

from typing import Any, Dict

from deep_oc_sort_3d.official_023_027.official_track1_validator import validate_official_track1
from deep_oc_sort_3d.v4_geometry_refinement.geometry_metrics import compact_metrics, compute_geometry_metrics
from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import output_root, v31_track1_path
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import read_geometry_rows, write_csv, write_json, write_yaml


def audit_v31_geometry(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Write V3.1 baseline geometry, suspect-track and suspect-point audits."""
    root = output_root(config)
    rows = read_geometry_rows(v31_track1_path(config), progress=progress)
    if not rows:
        raise RuntimeError("V3.1 Track1 input is missing or empty: %s" % v31_track1_path(config))
    metrics = compute_geometry_metrics(rows, config)
    compact = compact_metrics(metrics)
    validation = validate_official_track1(v31_track1_path(config), config, progress=False)
    checks = validation.get("checks", {})
    compact.update({
        "validation_errors": validation.get("num_errors"),
        "duplicate_keys": checks.get("duplicate_key_count"),
        "nan_inf": checks.get("nan_or_inf_values"),
        "non_positive_dimensions": checks.get("non_positive_dimensions"),
        "rounding_issues": checks.get("rounding_issues"),
    })
    write_yaml(root / "configs" / "resolved_config.yaml", {key: value for key, value in config.items() if not str(key).startswith("_")})
    write_json(root / "audit" / "v31_geometry_audit.json", compact)
    write_json(root / "audit" / "v31_track_smoothness_summary.json", _section(compact, ["step_", "suspect_", "jump_", "z_outlier", "track_count", "mean_track", "median_track"]))
    write_json(root / "audit" / "v31_dimension_consistency_summary.json", _section(compact, ["dimension_"]))
    write_json(root / "audit" / "v31_yaw_consistency_summary.json", _section(compact, ["yaw_"]))
    write_csv(root / "audit" / "suspect_tracks.csv", metrics.get("suspect_tracks", []))
    write_csv(root / "audit" / "suspect_points.csv", metrics.get("suspect_points", []))
    return compact


def _section(values: Dict[str, Any], prefixes: Any) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if any(str(key).startswith(prefix) for prefix in prefixes)}
