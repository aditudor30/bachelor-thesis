"""Official validation plus exact V4/V3.1 input identity preservation."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.official_023_027.official_track1_validator import validate_official_track1
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_config import input_track1_path
from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import (
    identity_key_set,
    read_geometry_rows,
    unique_track_count,
    write_json,
)


def validate_v5_track1(path: Path, output_path: Path, config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Validate official format and exact immutable input key preservation."""
    report = validate_official_track1(path, config, progress=progress)
    baseline = read_geometry_rows(input_track1_path(config), progress=False)
    candidate = read_geometry_rows(path, progress=False)
    baseline_keys = identity_key_set(baseline)
    candidate_keys = identity_key_set(candidate)
    identity = {
        "same_row_count_as_input": len(candidate) == len(baseline),
        "same_unique_track_count_as_input": unique_track_count(candidate) == unique_track_count(baseline),
        "identical_row_keys_as_input": candidate_keys == baseline_keys,
        "missing_keys": len(baseline_keys - candidate_keys), "extra_keys": len(candidate_keys - baseline_keys),
        "input_rows": len(baseline), "candidate_rows": len(candidate),
        "input_unique_tracks": unique_track_count(baseline), "candidate_unique_tracks": unique_track_count(candidate),
    }
    failed = [key for key in ["same_row_count_as_input", "same_unique_track_count_as_input", "identical_row_keys_as_input"] if not identity[key]]
    report["identity_preservation"] = identity
    if failed:
        report["status"] = "error"
        report["errors"] = list(report.get("errors", [])) + ["identity_preservation:%s" % key for key in failed]
        report["num_errors"] = len(report["errors"])
    write_json(output_path, report)
    return report
