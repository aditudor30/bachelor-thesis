"""Generate V5 calibrated Track1 variants from immutable V4/V3.1 input."""

from typing import Any, Dict, Sequence

from deep_oc_sort_3d.v5_geometry_calibration.correction_selector import correction_sets
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import input_track1_path, output_root, variant_root
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import read_geometry_rows, read_json, write_json, write_variant_rows
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_metrics import summarize_test_changes
from deep_oc_sort_3d.v5_geometry_calibration.test_track1_calibration_applier import apply_corrections_to_track1


def run_calibration_variants(config: Dict[str, Any], variants: Sequence[str], progress: bool = True) -> Dict[str, Any]:
    """Apply each correction set without changing rows, tracks or identities."""
    baseline = read_geometry_rows(input_track1_path(config), progress=progress)
    if not baseline:
        raise RuntimeError("V5 input Track1 is missing or empty: %s" % input_track1_path(config))
    selected = read_json(output_root(config) / "learned_corrections" / "selected_corrections.json")
    sets = correction_sets(selected)
    output: Dict[str, Any] = {}
    for variant in variants:
        refined, applied = apply_corrections_to_track1(baseline, sets[variant])
        root = variant_root(config, variant)
        write_variant_rows(root / "track1.txt", refined, config)
        official_rows = read_geometry_rows(root / "track1.txt", progress=False)
        metrics = summarize_test_changes(baseline, official_rows)
        metrics["variant"] = variant
        metrics["input_track1"] = str(input_track1_path(config))
        write_json(root / "geometry_summary.json", metrics)
        write_json(root / "applied_corrections_summary.json", applied)
        output[variant] = {"geometry_summary": metrics, "applied_corrections_summary": applied}
    return output
