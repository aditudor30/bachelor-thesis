"""Full pipeline runner for baseline_v2_pseudo3d_fullcam."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_stage_runner import run_fullcam_stages


def run_fullcam_pipeline(
    config_path: Path,
    overwrite: bool = False,
    progress: bool = True,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Run the complete baseline_v2_pseudo3d_fullcam pipeline."""
    return run_fullcam_stages(
        config_path=config_path,
        stages=None,
        overwrite=overwrite,
        progress=progress,
        dry_run=dry_run,
        skip_existing=skip_existing,
    )
