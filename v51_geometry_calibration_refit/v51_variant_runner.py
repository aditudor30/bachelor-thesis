"""Generate V5.1 variants from immutable V4/V3.1 geometry."""

from typing import Any, Dict, Sequence

from deep_oc_sort_3d.v51_geometry_calibration_refit.correction_selector import correction_sets
from deep_oc_sort_3d.v51_geometry_calibration_refit.test_track1_applier import apply_corrections_to_track1
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import input_track1_path, output_root, variant_root
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import read_geometry_rows, read_json, write_json, write_variant_rows
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_metrics import summarize_test_changes


def run_v51_variants(
    config: Dict[str, Any], variants: Sequence[str], progress: bool = True,
) -> Dict[str, Any]:
    baseline = read_geometry_rows(input_track1_path(config), progress=progress)
    if not baseline:
        raise RuntimeError("V5.1 input Track1 is missing or empty: %s" % input_track1_path(config))
    selected = read_json(output_root(config) / "learned_corrections" / "selected_corrections.json")
    sets = correction_sets(selected)
    output: Dict[str, Any] = {}
    for variant in variants:
        refined, applied = apply_corrections_to_track1(baseline, sets[variant])
        directory = variant_root(config, variant)
        write_variant_rows(directory / "track1.txt", refined, config)
        official_rows = read_geometry_rows(directory / "track1.txt", progress=False)
        metrics = summarize_test_changes(baseline, official_rows)
        metrics.update({"variant": variant, "input_track1": str(input_track1_path(config))})
        write_json(directory / "geometry_summary.json", metrics)
        write_json(directory / "applied_corrections_summary.json", applied)
        output[variant] = {"geometry_summary": metrics, "applied_corrections_summary": applied}
    return output
