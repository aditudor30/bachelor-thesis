"""Run conservative ReID-guided Person association experiments."""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.person_association.person_association_io import (
    TrackKey,
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
from deep_oc_sort_3d.person_association.person_merge_policy import apply_person_merge_mapping, mapping_rows
from deep_oc_sort_3d.person_association.person_merge_runner import _run_track1_export, _validate_track1
from deep_oc_sort_3d.person_cleanup.person_export_policy import apply_person_cleanup_export_policy
from deep_oc_sort_3d.person_reid_association.reid_association_comparison import compare_reid_person_association_runs
from deep_oc_sort_3d.person_reid_association.reid_association_metrics import collect_reid_association_metrics, write_metrics
from deep_oc_sort_3d.person_reid_association.reid_association_report import write_reid_association_report
from deep_oc_sort_3d.person_reid_association.reid_merge_policy import build_reid_person_merge_mapping
from deep_oc_sort_3d.person_reid_association.reid_pair_mining import (
    mine_reid_person_pairs_from_config,
    write_reid_candidate_pairs,
)
from deep_oc_sort_3d.person_reid_association.reid_pair_scoring import (
    score_reid_person_pairs,
    summarize_reid_pair_scores,
    write_reid_score_rows,
)


def run_reid_person_association_experiment(
    run_name: str,
    config_path: Path,
    output_root: Path,
    overwrite: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run one conservative ReID-guided Person association experiment."""
    _unused_overwrite = overwrite
    config = load_run_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_run_input(run_name, config_path, output_root)
    try:
        paths = config.get("paths", {})
        source_final = Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam")))
        schema_yaml = Path(str(paths.get("schema_yaml", "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml")))
        candidate_rows, candidate_summary = mine_reid_person_pairs_from_config(config, show_progress=progress)
        write_reid_candidate_pairs(
            candidate_rows,
            output_root / "candidate_pairs" / "reid_person_candidate_pairs.csv",
            output_root / "candidate_pairs" / "reid_person_candidate_pairs_summary.json",
            candidate_summary,
        )
        scored_rows = score_reid_person_pairs(candidate_rows, config.get("scoring", {}))
        write_reid_score_rows(scored_rows, output_root / "scores" / "reid_person_pair_scores.csv")
        score_summary = summarize_reid_pair_scores(
            scored_rows,
            threshold=config.get("merge_policy", {}).get("reid_similarity_threshold"),
        )
        write_json(score_summary, output_root / "scores" / "reid_person_pair_scores_summary.json")
        merge_policy = dict(config.get("merge_policy", {}))
        if bool(config.get("diagnostic_only", False)):
            merge_policy["apply_merges"] = False
        all_frame_rows = []
        if bool(merge_policy.get("apply_merges", True)):
            all_frame_rows = _load_all_frame_rows(source_final)
        mapping, merge_audit_rows, merge_summary = build_reid_person_merge_mapping(scored_rows, all_frame_rows, merge_policy)
        write_csv_rows(merge_audit_rows, output_root / "diagnostics" / "reid_merge_audit.csv")
        write_csv_rows(mapping_rows(mapping), output_root / "diagnostics" / "reid_merge_mapping.csv")
        write_json(merge_summary, output_root / "diagnostics" / "reid_merge_summary.json")
        merged_root = output_root / "merged_final_export" if bool(config.get("export_compact", {}).get("enabled", False)) else output_root / "final_export"
        _write_mapped_final_export(source_final, merged_root, mapping, config, progress)
        if bool(config.get("export_compact", {}).get("enabled", False)):
            apply_person_cleanup_export_policy(
                source_final_export_root=merged_root,
                output_final_export_root=output_root / "final_export",
                config=_compact_config(config),
                show_progress=progress,
            )
        track1_path = _run_track1_export(output_root, schema_yaml, progress)
        _validate_track1(track1_path, schema_yaml, output_root, progress)
        metrics = collect_reid_association_metrics(
            run_name,
            output_root / "final_export",
            output_root / "track1_submission",
            output_root / "diagnostics",
        )
        write_metrics(metrics, output_root / "summaries" / "run_metrics.json")
        status = {"run_name": run_name, "status": "ok", "error_message": ""}
    except Exception as exc:
        status = {"run_name": run_name, "status": "error", "error_message": str(exc)}
    write_json(status, output_root / "summaries" / "run_status.json")
    return status


def run_reid_person_association_sweep(
    config_path: Path,
    run_names: Optional[List[str]] = None,
    overwrite: bool = False,
    skip_existing: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run the configured ReID-guided Person association sweep."""
    config = load_yaml(config_path)
    root = _sweep_output_root(config)
    _copy_configs(config_path, config, root)
    requested = None if run_names is None else set([str(name) for name in run_names])
    statuses = []
    for run_item in progress_iter(config.get("runs", []), progress, "ReID Person association runs", "run"):
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
        status = run_reid_person_association_experiment(
            name,
            Path(str(run_item.get("config", ""))),
            run_root,
            overwrite=overwrite,
            progress=progress,
        )
        statuses.append(status)
        gc.collect()
        write_json({"runs": statuses}, root / "comparison" / "incremental_run_status.json")
    comparison = compare_reid_person_association_runs(config_path, output_root=root, progress=progress)
    write_reid_association_report(comparison, root / "report" / "PERSON_REID_GUIDED_ASSOCIATION_REPORT.md")
    return {"statuses": statuses, "comparison": comparison}


def load_run_config(config_path: Path) -> Dict[str, Any]:
    """Load one ReID association run config."""
    data = load_yaml(config_path)
    section = data.get("person_reid_association_run", data)
    return section if isinstance(section, dict) else {}


def mine_pairs_from_config(config_path: Path, output_root: Optional[Path] = None, progress: bool = True) -> Dict[str, Any]:
    """Mine ReID-covered Person candidate pairs without applying merges."""
    config = load_run_config(config_path)
    root = output_root if output_root is not None else Path(str(config.get("output_root", "output/person_reid_association/baseline_v2_pseudo3d_fullcam")))
    rows, summary = mine_reid_person_pairs_from_config(config, show_progress=progress)
    write_reid_candidate_pairs(
        rows,
        root / "candidate_pairs" / "reid_person_candidate_pairs.csv",
        root / "candidate_pairs" / "reid_person_candidate_pairs_summary.json",
        summary,
    )
    return {"candidate_pairs": len(rows), "pairs_with_both_reid": summary.get("pairs_with_both_reid"), "output_root": str(root)}


def score_pairs_from_config(
    config_path: Path,
    input_csv: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Score previously mined ReID Person candidate pairs."""
    config = load_run_config(config_path)
    root = output_root if output_root is not None else Path(str(config.get("output_root", "output/person_reid_association/baseline_v2_pseudo3d_fullcam")))
    source = input_csv if input_csv is not None else root / "candidate_pairs" / "reid_person_candidate_pairs.csv"
    rows, _fields = read_csv_rows(source)
    scored = score_reid_person_pairs(rows, config.get("scoring", {}))
    write_reid_score_rows(scored, root / "scores" / "reid_person_pair_scores.csv")
    summary = summarize_reid_pair_scores(scored, threshold=config.get("merge_policy", {}).get("reid_similarity_threshold"))
    write_json(summary, root / "scores" / "reid_person_pair_scores_summary.json")
    return summary


def _write_mapped_final_export(
    source_final: Path,
    output_final: Path,
    mapping: Dict[TrackKey, str],
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
        "ReID Person generic files",
    )
    frame_report = _write_mapped_tree(
        source_final / "frame_global_records",
        output_final / "frame_global_records",
        frame_record_csv_files(source_final / "frame_global_records", subsets=subsets, scenes=scenes),
        mapping,
        progress,
        "ReID Person frame files",
    )
    summary = {"generic_report": generic_report, "frame_report": frame_report, "mapping_size": len(mapping)}
    write_json(summary, output_final.parent / "summaries" / "mapped_final_export_summary.json")
    return summary


def _write_mapped_tree(
    source_root: Path,
    output_root: Path,
    files: List[Path],
    mapping: Dict[TrackKey, str],
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


def _sweep_output_root(config: Dict[str, Any]) -> Path:
    section = config.get("person_reid_association", {})
    paths = config.get("paths", {})
    return Path(str(section.get("output_root", paths.get("output_root", "output/person_reid_association/baseline_v2_pseudo3d_fullcam"))))


def _write_run_input(run_name: str, config_path: Path, output_root: Path) -> None:
    write_json({"run_name": run_name, "config_path": str(config_path)}, output_root / "summaries" / "run_input.json")


def _copy_configs(config_path: Path, config: Dict[str, Any], root: Path) -> None:
    config_root = root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        (config_root / "person_reid_association_sweep.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    for item in config.get("runs", []):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("config", "")))
        if path.exists():
            (config_root / ("%s.yaml" % item.get("name", path.stem))).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
