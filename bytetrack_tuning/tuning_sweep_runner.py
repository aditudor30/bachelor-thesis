"""Two-phase coverage-oriented ByteTrack sweep orchestration."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.bytetrack_tuning.tuning_config import tuning_output_root, write_resolved_configs
from deep_oc_sort_3d.bytetrack_tuning.tuning_io import progress_iter, write_json
from deep_oc_sort_3d.bytetrack_tuning.tuning_metrics import collect_all_tuning_metrics
from deep_oc_sort_3d.bytetrack_tuning.tuning_selector import preliminary_variant_ranking
from deep_oc_sort_3d.bytetrack_tuning.tuning_stage_runner import run_tuning_variant
from deep_oc_sort_3d.bytetrack_tuning.variant_grid import list_variants, validate_variant_grid


def run_tuning_sweep(
    config: Dict[str, Any],
    variant: Optional[str],
    full_all_variants: bool,
    top_k: Optional[int],
    progress: bool,
    overwrite: bool,
    skip_existing: bool,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run local sweep for all variants, then full pipeline for selected variants."""
    errors = validate_variant_grid(config)
    if errors:
        raise ValueError("Invalid ByteTrack tuning grid: %s" % "; ".join(errors))
    write_resolved_configs(config)
    names = [variant] if variant else list_variants(config)
    local_rows = []
    for name in progress_iter(names, progress, "ByteTrack coverage local sweep"):
        local_rows.append(
            run_tuning_variant(
                config,
                name,
                full_pipeline=False,
                progress=progress,
                overwrite=overwrite,
                skip_existing=skip_existing,
                dry_run=dry_run,
            )
        )
    if dry_run:
        return _write_runbook(config, local_rows, [], [])

    phase_a_metrics = collect_all_tuning_metrics(
        config,
        names=names,
        include_baselines=True,
        progress=progress,
    )
    configured_top_k = int(config.get("sweep", {}).get("run_full_for_top_k", 2))
    selected_count = configured_top_k if top_k is None else int(top_k)
    phase_a_baseline = (
        phase_a_metrics.get("baselines", {})
        .get("baseline_v2_current", {})
        .get("phase_a_local_tracking", {})
    )
    ranked = preliminary_variant_ranking(
        phase_a_metrics.get("variants", {}),
        config,
        baseline_v2=phase_a_baseline,
    )
    selected = names if full_all_variants else ranked[: max(1, selected_count)]
    full_rows = []
    for name in progress_iter(selected, progress, "ByteTrack coverage full candidates"):
        full_rows.append(
            run_tuning_variant(
                config,
                name,
                full_pipeline=True,
                progress=progress,
                overwrite=overwrite,
                skip_existing=skip_existing,
                dry_run=False,
            )
        )
    return _write_runbook(config, local_rows, full_rows, selected)


def _write_runbook(
    config: Dict[str, Any],
    local_rows: List[Dict[str, Any]],
    full_rows: List[Dict[str, Any]],
    selected: List[str],
) -> Dict[str, Any]:
    result = {"phase_a": local_rows, "phase_b": full_rows, "selected_for_full": selected}
    write_json(tuning_output_root(config) / "comparison" / "sweep_runbook.json", result)
    return result
