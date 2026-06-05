"""Packaging helpers for baseline_v2_pseudo3d_fullcam."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_export.track1_packaging import package_track1_submission


def package_fullcam_submission(config: Dict[str, Any], overwrite: bool = False, show_progress: bool = True) -> Dict[str, Any]:
    """Create a separate submission package for baseline_v2_pseudo3d_fullcam."""
    paths = config.get("paths", {})
    track1_path = Path(paths.get("output_track1_root", "output/track1_submission/baseline_v2_pseudo3d_fullcam")) / "track1.txt"
    output_root = Path(paths.get("output_package_root", "output/submission_packages/baseline_v2_pseudo3d_fullcam"))
    config_paths = _existing_paths(
        [
            "deep_oc_sort_3d/configs/baseline_v2_fullcam_full_pipeline.yaml",
            "deep_oc_sort_3d/configs/baseline_v2_fullcam_observations.yaml",
            "deep_oc_sort_3d/configs/baseline_v2_fullcam_local_tracking.yaml",
            "deep_oc_sort_3d/configs/baseline_v2_fullcam_global_association.yaml",
            "deep_oc_sort_3d/configs/baseline_v2_fullcam_track1_export.yaml",
        ]
    )
    reports = _existing_paths(
        [
            Path(paths.get("output_comparison_root", "output/baseline_v2_pseudo3d_fullcam_comparison"))
            / "BASELINE_V1_VS_V2_PSEUDO3D_FULLCAM_REPORT.md",
            Path(paths.get("output_track1_root", "output/track1_submission/baseline_v2_pseudo3d_fullcam"))
            / "validation"
            / "track1_validation_summary.json",
        ]
    )
    return package_track1_submission(
        track1_path=track1_path,
        output_package_root=output_root,
        config_paths=config_paths,
        reports=reports,
        baseline_name="baseline_v2_pseudo3d_fullcam",
        overwrite=overwrite,
        make_zip=False,
        show_progress=show_progress,
    )


def _existing_paths(values: List[Any]) -> List[Path]:
    paths = []
    for value in values:
        path = Path(value)
        if path.exists():
            paths.append(path)
    return paths
