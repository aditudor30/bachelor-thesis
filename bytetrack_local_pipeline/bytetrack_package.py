"""Package the isolated ByteTrack-local Track1 submission."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_export.track1_packaging import package_track1_submission


def package_bytetrack_submission(
    config: Dict[str, Any],
    overwrite: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Create package without modifying previous submission artifacts."""
    paths = config.get("paths", {})
    track1_path = Path(str(paths.get("output_track1_root"))) / "track1.txt"
    package_root = Path(str(paths.get("output_package_root")))
    reports = _existing(
        [
            Path(str(paths.get("output_comparison_root"))) / "BASELINE_V2_BYTETRACK_LOCAL_REPORT.md",
            Path(str(paths.get("output_track1_root"))) / "validation" / "track1_validation_summary.json",
        ]
    )
    config_paths = _existing([Path(str(config.get("_config_path", "")))])
    return package_track1_submission(
        track1_path=track1_path,
        output_package_root=package_root,
        config_paths=config_paths,
        reports=reports,
        baseline_name="baseline_v2_pseudo3d_fullcam_bytetrack_local",
        overwrite=overwrite,
        make_zip=False,
        show_progress=progress,
    )


def _existing(values: List[Path]) -> List[Path]:
    return [path for path in values if path.exists()]
