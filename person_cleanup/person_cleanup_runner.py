"""Runner for Person cleanup audit and experiments."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.final_export.track1_export_types import load_track1_schema_yaml
from deep_oc_sort_3d.final_export.track1_validator import validate_track1_export, write_track1_validation_report
from deep_oc_sort_3d.person_cleanup.person_cleanup_comparison import compare_person_cleanup_runs
from deep_oc_sort_3d.person_cleanup.person_cleanup_io import load_yaml, progress_iter, write_json, write_yaml
from deep_oc_sort_3d.person_cleanup.person_cleanup_metrics import collect_person_cleanup_metrics, write_metrics
from deep_oc_sort_3d.person_cleanup.person_cleanup_report import write_person_cleanup_report
from deep_oc_sort_3d.person_cleanup.person_export_policy import apply_person_cleanup_export_policy
from deep_oc_sort_3d.person_cleanup.person_fragmentation_audit import audit_person_fragmentation
from deep_oc_sort_3d.scripts.export_track1_submission import export_track1_submission


class Namespace:
    """Small argparse-like namespace."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def run_person_cleanup_experiment(
    run_name: str,
    config_path: Path,
    output_root: Path,
    overwrite: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run one Person cleanup experiment."""
    config = load_run_config(config_path)
    paths = config.get("paths", {})
    source_final = Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam")))
    schema_yaml = Path(str(paths.get("schema_yaml", "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml")))
    output_root.mkdir(parents=True, exist_ok=True)
    write_json({"run_name": run_name, "config_path": str(config_path)}, output_root / "summaries" / "run_input.json")
    try:
        apply_person_cleanup_export_policy(
            source_final_export_root=source_final,
            output_final_export_root=output_root / "final_export",
            config=config,
            show_progress=progress,
        )
        track1_path = _run_track1_export(output_root, schema_yaml, progress)
        _validate_track1(track1_path, schema_yaml, output_root, progress)
        metrics = collect_person_cleanup_metrics(
            run_name,
            output_root / "final_export",
            output_root / "track1_submission",
            Path(str(paths.get("v2_global_root", "output/global_mtmc_transition/baseline_v2_pseudo3d_fullcam"))),
        )
        write_metrics(metrics, output_root / "summaries" / "run_metrics.json")
        status = {"run_name": run_name, "status": "ok", "error_message": ""}
    except Exception as exc:
        status = {"run_name": run_name, "status": "error", "error_message": str(exc)}
    write_json(status, output_root / "summaries" / "run_status.json")
    return status


def run_person_cleanup_sweep(
    config_path: Path,
    run_names: Optional[List[str]] = None,
    overwrite: bool = False,
    skip_existing: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run audit, selected cleanup experiments, comparison, and report."""
    config = load_yaml(config_path)
    root = Path(config.get("person_cleanup", {}).get("output_root", config.get("paths", {}).get("output_root", "output/person_cleanup/baseline_v2_pseudo3d_fullcam")))
    _copy_configs(config_path, config, root)
    audit_person_fragmentation(
        final_export_root=Path(str(config.get("paths", {}).get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam"))),
        output_root=root / "audit",
        config=config.get("person_cleanup", {}),
        progress=progress,
    )
    requested = None if run_names is None else set([str(name) for name in run_names])
    statuses = []
    for run_item in progress_iter(config.get("runs", []), progress, "person cleanup runs", "run"):
        if not isinstance(run_item, dict):
            continue
        name = str(run_item.get("name", ""))
        if requested is not None and name not in requested:
            continue
        run_root = root / "runs" / name
        status_path = run_root / "summaries" / "run_status.json"
        if skip_existing and status_path.exists():
            statuses.append({"run_name": name, "status": "skipped_existing", "error_message": ""})
            continue
        status = run_person_cleanup_experiment(
            name,
            Path(str(run_item.get("config", ""))),
            run_root,
            overwrite=overwrite,
            progress=progress,
        )
        statuses.append(status)
        write_json({"runs": statuses}, root / "comparison" / "incremental_run_status.json")
    comparison = compare_person_cleanup_runs(config_path, output_root=root, progress=progress)
    write_person_cleanup_report(comparison, root / "comparison" / "PERSON_CLEANUP_REPORT.md")
    return {"statuses": statuses, "comparison": comparison}


def load_run_config(config_path: Path) -> Dict[str, Any]:
    """Load one cleanup run config and normalize section."""
    data = load_yaml(config_path)
    section = data.get("person_cleanup_run", data)
    return section if isinstance(section, dict) else {}


def _run_track1_export(output_root: Path, schema_yaml: Path, progress: bool) -> Path:
    track1_config_path = output_root / "configs" / "track1_export.yaml"
    track1_config = {
        "track1_export": {
            "generic_export_root": str(output_root / "final_export" / "generic_tracking_export" / "test"),
            "output_root": str(output_root / "track1_submission"),
            "schema_yaml": str(schema_yaml),
            "force_unconfirmed_preview": False,
            "subsets": ["test"],
            "scenes": ["Warehouse_023", "Warehouse_024", "Warehouse_025"],
            "progress": bool(progress),
        }
    }
    write_yaml(track1_config, track1_config_path)
    args = Namespace(
        config=track1_config_path,
        generic_export_root=None,
        schema_yaml=None,
        schema_report=None,
        output_root=None,
        subsets=None,
        scenes=None,
        force_unconfirmed_preview=None,
        progress=progress,
    )
    export_track1_submission(args)
    return output_root / "track1_submission" / "track1.txt"


def _validate_track1(track1_path: Path, schema_yaml: Path, output_root: Path, progress: bool) -> Dict[str, Any]:
    schema = load_track1_schema_yaml(schema_yaml)
    report = validate_track1_export(track1_path, schema, show_progress=progress)
    write_track1_validation_report(report, output_root / "track1_submission" / "track1_validation_report.json")
    write_track1_validation_report(report, output_root / "validation" / "track1_validation_report.json")
    return report


def _copy_configs(config_path: Path, config: Dict[str, Any], root: Path) -> None:
    config_root = root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        (config_root / "person_cleanup_sweep.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    for item in config.get("runs", []):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("config", "")))
        if path.exists():
            (config_root / ("%s.yaml" % item.get("name", path.stem))).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

