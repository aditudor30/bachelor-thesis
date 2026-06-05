"""Validation helpers for baseline_v2_pseudo3d_fullcam Track1 exports."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.final_export.track1_final_checks import read_track1_txt, validate_track1_rows


def validate_fullcam_track1(
    track1_path: Path,
    output_root: Optional[Path] = None,
    expected_scene_ids: Optional[List[int]] = None,
    valid_class_ids: Optional[List[int]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Validate a baseline_v2_fullcam Track1 text file."""
    scene_ids = expected_scene_ids if expected_scene_ids is not None else [23, 24, 25]
    class_ids = valid_class_ids if valid_class_ids is not None else [0, 1, 2, 3, 4, 5, 6]
    rows = read_track1_txt(track1_path)
    report = validate_track1_rows(rows, expected_scene_ids=scene_ids, valid_class_ids=class_ids, show_progress=show_progress)
    report["track1_path"] = str(track1_path)
    if output_root is not None:
        write_fullcam_validation_report(report, output_root)
    return report


def write_fullcam_validation_report(report: Dict[str, Any], output_root: Path) -> None:
    """Write validation report aliases expected by Step 15H."""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "track1_validation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "final_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
