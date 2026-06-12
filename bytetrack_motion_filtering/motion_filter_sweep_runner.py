"""Orchestrate all Step 21E motion-filter variants."""

from typing import Any, Dict, Optional

from deep_oc_sort_3d.bytetrack_motion_filtering.downstream_pipeline_runner import run_downstream_pipeline
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_config import variant_names, write_resolved_config
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_io import progress_iter
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_variant_runner import run_motion_filter_variant
from deep_oc_sort_3d.bytetrack_motion_filtering.velocity_prior_estimator import load_or_estimate_velocity_priors


def run_motion_filter_sweep(
    config: Dict[str, Any],
    variant: Optional[str] = None,
    progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Run filtering and unchanged downstream stages for configured variants."""
    write_resolved_config(config)
    load_or_estimate_velocity_priors(config, progress=progress, overwrite=overwrite)
    names = [variant] if variant else variant_names(config)
    rows = []
    for name in progress_iter(names, progress, "Step 21E variants"):
        filter_summary = run_motion_filter_variant(
            config,
            str(name),
            progress=progress,
            overwrite=overwrite,
            skip_existing=skip_existing,
        )
        downstream = run_downstream_pipeline(
            config,
            str(name),
            progress=progress,
            overwrite=overwrite,
            skip_existing=skip_existing,
        )
        rows.append({"variant_name": str(name), "motion_filter": filter_summary, "downstream": downstream})
    return {"variants": rows}

