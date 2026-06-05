"""Runner for controlled global association tuning sweeps."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.global_tuning.tuned_global_export import (
    materialize_configs_for_run,
    maybe_run_compact_export_stage,
    run_final_export_stage,
    run_global_association_stage,
    run_track1_export_stage,
    validate_track1_stage,
)
from deep_oc_sort_3d.global_tuning.tuning_comparison import compare_global_tuning_runs
from deep_oc_sort_3d.global_tuning.tuning_config import (
    build_run_spec_from_config,
    build_run_specs_from_sweep,
    load_sweep_config,
)
from deep_oc_sort_3d.global_tuning.tuning_io import progress_iter, write_json
from deep_oc_sort_3d.global_tuning.tuning_metrics import collect_run_metrics, write_run_metrics
from deep_oc_sort_3d.global_tuning.tuning_report import write_global_tuning_report


def run_single_global_tuning(
    run_name: str,
    config_path: Path,
    output_root: Path,
    overwrite: bool = False,
    skip_existing: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run one tuning preset end-to-end from motion-clean candidates onward."""
    spec = build_run_spec_from_config(run_name, config_path, output_root=output_root)
    return run_resolved_spec(spec, overwrite=overwrite, skip_existing=skip_existing, progress=progress)


def run_resolved_spec(
    spec: Any,
    overwrite: bool = False,
    skip_existing: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run one already-resolved spec."""
    metrics_path = spec.summaries_root / "run_metrics.json"
    if skip_existing and metrics_path.exists():
        return {"run_name": spec.name, "status": "skipped_existing", "metrics_path": str(metrics_path)}
    spec.output_root.mkdir(parents=True, exist_ok=True)
    runtime_configs = materialize_configs_for_run(spec, progress=progress)
    status = {"run_name": spec.name, "status": "ok", "error_message": ""}
    try:
        print("[global_association] %s" % spec.name)
        run_global_association_stage(spec, runtime_configs, overwrite=overwrite, progress=progress)
        print("[final_export] %s" % spec.name)
        run_final_export_stage(spec, runtime_configs, overwrite=overwrite, progress=progress)
        print("[compact_export] %s" % spec.name)
        generic_root = maybe_run_compact_export_stage(spec, progress=progress)
        print("[track1_export] %s" % spec.name)
        submission_path = run_track1_export_stage(spec, generic_root=generic_root, progress=progress)
        print("[track1_validation] %s" % spec.name)
        validate_track1_stage(spec, submission_path=submission_path, progress=progress)
        metrics = collect_run_metrics(spec.name, spec.output_root)
        write_run_metrics(metrics, metrics_path)
        status["metrics_path"] = str(metrics_path)
    except Exception as exc:
        status["status"] = "error"
        status["error_message"] = str(exc)
    write_json(status, spec.summaries_root / "run_status.json")
    return status


def run_global_tuning_sweep(
    config_path: Path,
    run_names: Optional[List[str]] = None,
    overwrite: bool = False,
    skip_existing: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run all selected tuning presets and write comparison outputs."""
    sweep = load_sweep_config(config_path)
    output_root = Path(sweep.get("paths", {}).get("output_root", "output/global_tuning/debug"))
    specs = build_run_specs_from_sweep(config_path, run_names=run_names)
    _copy_run_configs(config_path, specs, output_root)
    statuses = []
    for spec in progress_iter(specs, progress, "global tuning runs", "run"):
        if not spec.enabled:
            statuses.append({"run_name": spec.name, "status": "disabled", "error_message": ""})
            continue
        status = run_resolved_spec(spec, overwrite=overwrite, skip_existing=skip_existing, progress=progress)
        statuses.append(status)
        write_json({"runs": statuses}, output_root / "comparison" / "incremental_run_status.json")
    comparison = compare_global_tuning_runs(config_path, output_root=output_root, progress=progress)
    write_global_tuning_report(comparison, output_root / "comparison" / "GLOBAL_ASSOCIATION_TUNING_REPORT.md")
    return {"statuses": statuses, "comparison": comparison}


def _copy_run_configs(config_path: Path, specs: List[Any], output_root: Path) -> None:
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        (config_root / "sweep.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    for spec in specs:
        if spec.config_path is not None and spec.config_path.exists():
            (config_root / ("%s.yaml" % spec.name)).write_text(
                spec.config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
