"""Track1 validation wrapper for frozen upload candidates."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.final_export.track1_final_checks import read_track1_txt, validate_track1_rows
from deep_oc_sort_3d.freeze_upload.freeze_io import write_json


def validate_frozen_track1(
    path: Path,
    config: Dict[str, Any],
    progress: bool = True,
) -> Dict[str, Any]:
    """Validate one frozen Track1 file using the confirmed official schema."""
    section = config.get("track1_validation", {})
    if not path.exists():
        return _missing_report(path)
    rows = read_track1_txt(path)
    report = validate_track1_rows(
        rows,
        expected_scene_ids=[int(value) for value in section.get("valid_scene_ids", [23, 24, 25])],
        valid_class_ids=[int(value) for value in section.get("valid_class_ids", [0, 1, 2, 3, 4, 5, 6])],
        show_progress=progress,
    )
    report["track1_path"] = str(path)
    report["expected_num_columns"] = int(section.get("expected_num_columns", 11))
    return report


def validate_and_write(
    path: Path,
    output_path: Path,
    config: Dict[str, Any],
    progress: bool = True,
) -> Dict[str, Any]:
    """Validate and write the requested validation summary."""
    report = validate_frozen_track1(path, config, progress=progress)
    write_json(output_path, report)
    return report


def _missing_report(path: Path) -> Dict[str, Any]:
    return {
        "status": "error",
        "num_errors": 1,
        "num_warnings": 0,
        "errors": ["missing_file"],
        "warnings": [],
        "checks": {"empty_file": 1},
        "total_rows": 0,
        "distribution": {},
        "track1_path": str(path),
    }

