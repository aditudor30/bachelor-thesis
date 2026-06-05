"""Run Person-aware association experiments."""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.final_export.track1_export_types import load_track1_schema_yaml
from deep_oc_sort_3d.final_export.track1_validator import validate_track1_export, write_track1_validation_report
from deep_oc_sort_3d.person_association.person_association_comparison import compare_person_association_runs
from deep_oc_sort_3d.person_association.person_association_io import (
    frame_record_csv_files,
    generic_csv_files,
    infer_subset_from_path,
    load_yaml,
    optional_list,
    progress_iter,
    read_csv_rows,
    write_csv_rows,
    write_json,
    write_yaml,
)
from deep_oc_sort_3d.person_association.person_association_metrics import collect_person_association_metrics, write_metrics
from deep_oc_sort_3d.person_association.person_association_report import write_person_association_report
from deep_oc_sort_3d.person_association.person_merge_policy import (
    apply_person_merge_mapping,
    build_person_merge_mapping,
    mapping_rows,
    summarize_merge_audit,
)
from deep_oc_sort_3d.person_association.person_pair_mining import (
    load_person_fragments_from_final_export,
    mine_person_candidate_pairs_with_summary,
    write_candidate_pairs,
)
from deep_oc_sort_3d.person_association.person_pair_scoring import score_person_pairs, summarize_pair_scores, write_score_rows
from deep_oc_sort_3d.person_cleanup.person_export_policy import apply_person_cleanup_export_policy
from deep_oc_sort_3d.scripts.export_track1_submission import export_track1_submission


class Namespace:
    """Small argparse-like namespace."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def run_person_association_experiment(
    run_name: str,
    config_path: Path,
    output_root: Path,
    overwrite: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run one Person-aware association experiment."""
    config = load_run_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_run_input(run_name, config_path, output_root)
    try:
        paths = config.get("paths", {})
        source_final = Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam")))
        schema_yaml = Path(str(paths.get("schema_yaml", "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml")))
        fragments = load_person_fragments_from_final_export(source_final, _fragment_config(config), show_progress=progress)
        write_json(_fragment_summary(fragments), output_root / "audit" / "person_fragment_summary.json")
        candidate_rows, candidate_summary = mine_person_candidate_pairs_with_summary(
            fragments,
            config.get("pair_mining", {}),
            show_progress=progress,
        )
        write_candidate_pairs(
            candidate_rows,
            output_root / "candidate_pairs" / "person_candidate_pairs.csv",
            output_root / "candidate_pairs" / "person_candidate_pairs_summary.json",
            summary=candidate_summary,
        )
        scored_rows = score_person_pairs(candidate_rows, config.get("scoring", {}))
        write_score_rows(scored_rows, output_root / "scores" / "person_pair_scores.csv")
        write_json(
            summarize_pair_scores(scored_rows, max_pair_score=config.get("merge_policy", {}).get("max_pair_score")),
            output_root / "scores" / "person_pair_scores_summary.json",
        )
        merge_policy = dict(config.get("merge_policy", {}))
        if bool(config.get("diagnostic_only", False)):
            merge_policy["apply_merges"] = False
        all_frame_rows = []
        if bool(merge_policy.get("apply_merges", True)):
            all_frame_rows = _load_all_frame_rows(source_final)
        mapping, merge_audit_rows = build_person_merge_mapping(scored_rows, all_frame_rows, merge_policy)
        merge_summary = summarize_merge_audit(merge_audit_rows, mapping)
        write_csv_rows(merge_audit_rows, output_root / "diagnostics" / "person_merge_audit.csv")
        write_csv_rows(mapping_rows(mapping), output_root / "diagnostics" / "person_merge_mapping.csv")
        write_json(merge_summary, output_root / "diagnostics" / "person_merge_summary.json")
        merged_root = output_root / "merged_final_export" if bool(config.get("export_compact", {}).get("enabled", False)) else output_root / "final_export"
        _write_mapped_final_export(source_final, merged_root, mapping, config, progress)
        if bool(config.get("export_compact", {}).get("enabled", False)):
            compact_config = _compact_config(config)
            apply_person_cleanup_export_policy(
                source_final_export_root=merged_root,
                output_final_export_root=output_root / "final_export",
                config=compact_config,
                show_progress=progress,
            )
        track1_path = _run_track1_export(output_root, schema_yaml, progress)
        _validate_track1(track1_path, schema_yaml, output_root, progress)
        metrics = collect_person_association_metrics(
            run_name,
            output_root / "final_export",
            output_root / "track1_submission",
            output_root / "diagnostics" / "person_merge_summary.json",
        )
        write_metrics(metrics, output_root / "summaries" / "run_metrics.json")
        status = {"run_name": run_name, "status": "ok", "error_message": ""}
    except Exception as exc:
        status = {"run_name": run_name, "status": "error", "error_message": str(exc)}
    write_json(status, output_root / "summaries" / "run_status.json")
    return status


def run_person_association_sweep(
    config_path: Path,
    run_names: Optional[List[str]] = None,
    overwrite: bool = False,
    skip_existing: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run the configured Person-aware association sweep."""
    config = load_yaml(config_path)
    root = Path(config.get("person_association", {}).get("output_root", config.get("paths", {}).get("output_root", "output/person_association/baseline_v2_pseudo3d_fullcam")))
    _copy_configs(config_path, config, root)
    requested = None if run_names is None else set([str(name) for name in run_names])
    statuses = []
    for run_item in progress_iter(config.get("runs", []), progress, "person association runs", "run"):
        if not isinstance(run_item, dict):
            continue
        name = str(run_item.get("name", ""))
        if requested is not None and name not in requested:
            continue
        run_root = root / "experiments" / name
        status_path = run_root / "summaries" / "run_status.json"
        if skip_existing and status_path.exists():
            statuses.append({"run_name": name, "status": "skipped_existing", "error_message": ""})
            continue
        status = run_person_association_experiment(
            name,
            Path(str(run_item.get("config", ""))),
            run_root,
            overwrite=overwrite,
            progress=progress,
        )
        statuses.append(status)
        gc.collect()
        write_json({"runs": statuses}, root / "comparison" / "incremental_run_status.json")
    comparison = compare_person_association_runs(config_path, output_root=root, progress=progress)
    write_person_association_report(comparison, root / "report" / "PERSON_AWARE_ASSOCIATION_REPORT.md")
    return {"statuses": statuses, "comparison": comparison}


def load_run_config(config_path: Path) -> Dict[str, Any]:
    """Load one association run config."""
    data = load_yaml(config_path)
    section = data.get("person_association_run", data)
    return section if isinstance(section, dict) else {}


def mine_pairs_from_config(config_path: Path, output_root: Optional[Path] = None, progress: bool = True) -> Dict[str, Any]:
    """Mine candidate pairs for diagnostic use without running a merge."""
    config = load_run_config(config_path)
    paths = config.get("paths", {})
    source_final = Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam")))
    root = output_root if output_root is not None else Path(str(config.get("output_root", "output/person_association/baseline_v2_pseudo3d_fullcam")))
    fragments = load_person_fragments_from_final_export(source_final, _fragment_config(config), show_progress=progress)
    rows, candidate_summary = mine_person_candidate_pairs_with_summary(
        fragments,
        config.get("pair_mining", {}),
        show_progress=progress,
    )
    write_candidate_pairs(
        rows,
        root / "candidate_pairs" / "person_candidate_pairs.csv",
        root / "candidate_pairs" / "person_candidate_pairs_summary.json",
        summary=candidate_summary,
    )
    write_json(_fragment_summary(fragments), root / "audit" / "person_fragment_summary.json")
    return {"fragments": len(fragments), "candidate_pairs": len(rows), "output_root": str(root)}


def score_pairs_from_config(
    config_path: Path,
    input_csv: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Score previously mined candidate pairs."""
    config = load_run_config(config_path)
    root = output_root if output_root is not None else Path(str(config.get("output_root", "output/person_association/baseline_v2_pseudo3d_fullcam")))
    source = input_csv if input_csv is not None else root / "candidate_pairs" / "person_candidate_pairs.csv"
    rows, _fields = read_csv_rows(source)
    scored = score_person_pairs(rows, config.get("scoring", {}))
    write_score_rows(scored, root / "scores" / "person_pair_scores.csv")
    summary = summarize_pair_scores(scored, max_pair_score=config.get("merge_policy", {}).get("max_pair_score"))
    write_json(summary, root / "scores" / "person_pair_scores_summary.json")
    return summary


def _write_mapped_final_export(
    source_final: Path,
    output_final: Path,
    mapping: Dict[Tuple[str, str, str, str], str],
    config: Dict[str, Any],
    progress: bool,
) -> Dict[str, Any]:
    subsets = optional_list(config.get("apply_to_subsets"))
    scenes = optional_list(config.get("apply_to_scenes"))
    generic_report = _write_mapped_tree(
        source_final / "generic_tracking_export",
        output_final / "generic_tracking_export",
        generic_csv_files(source_final / "generic_tracking_export", subsets=subsets, scenes=scenes),
        mapping,
        progress,
        "person association generic files",
    )
    frame_report = _write_mapped_tree(
        source_final / "frame_global_records",
        output_final / "frame_global_records",
        frame_record_csv_files(source_final / "frame_global_records", subsets=subsets, scenes=scenes),
        mapping,
        progress,
        "person association frame files",
    )
    summary = {"generic_report": generic_report, "frame_report": frame_report, "mapping_size": len(mapping)}
    write_json(summary, output_final.parent / "summaries" / "mapped_final_export_summary.json")
    return summary


def _write_mapped_tree(
    source_root: Path,
    output_root: Path,
    files: List[Path],
    mapping: Dict[Tuple[str, str, str, str], str],
    progress: bool,
    desc: str,
) -> Dict[str, Any]:
    rows_written = 0
    for path in progress_iter(files, progress, desc, "file"):
        rows, fields = read_csv_rows(path)
        subset = infer_subset_from_path(path)
        working = []
        for row in rows:
            copied = dict(row)
            copied.setdefault("subset", subset)
            working.append(copied)
        mapped = apply_person_merge_mapping(working, mapping)
        relative = path.relative_to(source_root)
        write_csv_rows([{field: row.get(field, "") for field in fields} for row in mapped], output_root / relative, fields)
        rows_written += len(mapped)
    return {"files": len(files), "rows_written": rows_written}


def _load_all_frame_rows(final_export_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in frame_record_csv_files(final_export_root / "frame_global_records"):
        file_rows, _fields = read_csv_rows(path)
        rows.extend(file_rows)
    return rows


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


def _compact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(config.get("export_compact", {}))
    return {
        "apply_to_subsets": compact.get("apply_to_subsets", config.get("apply_to_subsets", ["official_val", "internal_holdout", "test"])),
        "apply_to_scenes": config.get("apply_to_scenes"),
        "classification": compact.get(
            "classification",
            {"short_rows_threshold": 5, "low_mean_confidence_threshold": 0.02, "low_max_confidence_threshold": 0.06},
        ),
        "pruning": compact.get(
            "pruning",
            {
                "enabled": True,
                "class_id": 0,
                "mode": "compact",
                "max_rows_per_track": 5,
                "mean_confidence_threshold": 0.02,
            },
        ),
        "selective_merge_safe": {"enabled": False},
    }


def _fragment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class_id": config.get("class_id", 0),
        "subsets": config.get("apply_to_subsets"),
        "scenes": config.get("apply_to_scenes"),
    }


def _fragment_summary(fragments: List[Any]) -> Dict[str, Any]:
    per_subset: Dict[str, int] = {}
    per_scene: Dict[str, int] = {}
    for fragment in fragments:
        per_subset[fragment.subset] = per_subset.get(fragment.subset, 0) + 1
        per_scene[fragment.scene_name] = per_scene.get(fragment.scene_name, 0) + 1
    return {"fragments": len(fragments), "per_subset": per_subset, "per_scene": per_scene}


def _write_run_input(run_name: str, config_path: Path, output_root: Path) -> None:
    write_json({"run_name": run_name, "config_path": str(config_path)}, output_root / "summaries" / "run_input.json")


def _copy_configs(config_path: Path, config: Dict[str, Any], root: Path) -> None:
    config_root = root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        (config_root / "person_association_sweep.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    for item in config.get("runs", []):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("config", "")))
        if path.exists():
            (config_root / ("%s.yaml" % item.get("name", path.stem))).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
