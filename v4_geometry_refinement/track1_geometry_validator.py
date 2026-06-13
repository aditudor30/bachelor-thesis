"""Official validation plus strict V3.1 identity preservation checks."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.official_023_027.official_track1_validator import validate_official_track1
from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import v31_track1_path
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import identity_key_set, read_geometry_rows, unique_track_count, write_json


def validate_v4_track1(path: Path, output_path: Path, config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Validate official format and exact immutable key preservation."""
    report = validate_official_track1(path, config, progress=progress)
    baseline = read_geometry_rows(v31_track1_path(config), progress=False)
    candidate = read_geometry_rows(path, progress=False)
    baseline_keys = identity_key_set(baseline)
    candidate_keys = identity_key_set(candidate)
    identity_checks = {
        "same_row_count_as_v31": len(candidate) == len(baseline),
        "same_unique_track_count_as_v31": unique_track_count(candidate) == unique_track_count(baseline),
        "identical_row_keys_as_v31": candidate_keys == baseline_keys,
        "missing_keys": len(baseline_keys - candidate_keys),
        "extra_keys": len(candidate_keys - baseline_keys),
        "v31_rows": len(baseline), "candidate_rows": len(candidate),
        "v31_unique_tracks": unique_track_count(baseline), "candidate_unique_tracks": unique_track_count(candidate),
    }
    failures = [key for key in ["same_row_count_as_v31", "same_unique_track_count_as_v31", "identical_row_keys_as_v31"] if not identity_checks[key]]
    report["identity_preservation"] = identity_checks
    if failures:
        report["status"] = "error"
        report["errors"] = list(report.get("errors", [])) + ["identity_preservation:%s" % key for key in failures]
        report["num_errors"] = len(report["errors"])
    write_json(output_path, report)
    return report

