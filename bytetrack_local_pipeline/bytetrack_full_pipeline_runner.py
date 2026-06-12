"""Full Step 21B pipeline entry point."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_pipeline_config import selected_stages
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_stage_runner import run_bytetrack_stages


def run_bytetrack_full_pipeline(
    config_path: Path,
    progress: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run precheck, then the isolated full rerun if the gate passes."""
    return run_bytetrack_stages(
        config_path=config_path,
        stages=selected_stages(None),
        progress=progress,
        overwrite=overwrite,
        dry_run=dry_run,
    )
