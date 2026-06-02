"""Run final MVP export pipeline for propagated global IDs."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.final_export.export_eval import (
    evaluate_global_frame_records,
    save_final_eval_csv,
    save_final_eval_json,
)
from deep_oc_sort_3d.final_export.export_summary import (
    print_final_export_summary,
    summarize_generic_exports,
    summarize_propagation_rows,
    write_summary_csv,
    write_summary_json,
)
from deep_oc_sort_3d.final_export.export_validation import (
    validate_generic_tracking_export,
    validate_global_frame_record_file,
    write_validation_report,
)
from deep_oc_sort_3d.final_export.generic_export import (
    export_generic_tracking_scene_csv,
    read_global_frame_records_file,
)
from deep_oc_sort_3d.final_export.global_id_propagation import propagate_for_camera_file


def run_final_mvp_export(args: Any) -> None:
    """Run final MVP export pipeline."""
    config = _load_config(args.config)
    options = _resolve_options(config, args)
    scenes = _scenes_from_config(options)
    propagation_rows = []
    for subset, split, scene_name in _progress_iter(scenes, options["progress"], "final propagation scenes", "scene"):
        _unused_split = split
        propagation_rows.extend(_propagate_scene(subset, scene_name, options))
    propagation_summary = summarize_propagation_rows(propagation_rows)
    summary_root = Path(options["output_root"]) / "summaries"
    write_summary_json(propagation_summary, summary_root / "propagation_summary.json")
    write_summary_csv(propagation_summary, summary_root / "propagation_summary.csv")

    export_rows = _export_generic_scenes(scenes, options)
    export_summary = summarize_generic_exports(export_rows)
    write_summary_json(export_summary, summary_root / "export_summary.json")
    write_summary_csv(export_summary, summary_root / "export_summary.csv")

    _validate_outputs(options)
    _evaluate_outputs(scenes, options)
    print_final_export_summary(propagation_summary)
    print("generic rows written: %s" % export_summary.get("rows_written"))
    print("Run root: %s" % options["output_root"])


def _propagate_scene(subset: str, scene_name: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    local_scene_root = Path(options["local_tracks_root"]) / subset / scene_name
    global_scene_root = Path(options["global_mtmc_root"]) / subset / scene_name
    candidates_path = global_scene_root / "candidates_with_global_ids.jsonl"
    if not candidates_path.exists():
        candidates_path = global_scene_root / "candidates_with_global_ids.csv"
    output_scene_root = Path(options["output_root"]) / "frame_global_records" / subset / scene_name
    rows = []
    local_files = sorted(local_scene_root.glob("*.csv"))
    camera_ids = options.get("camera_ids")
    if camera_ids is not None:
        camera_set = set([str(item) for item in camera_ids])
        local_files = [path for path in local_files if path.stem in camera_set]
    for local_file in _progress_iter(local_files, options["progress"], "final propagation cameras", "camera"):
        camera_id = local_file.stem
        output_csv = output_scene_root / ("%s_global_records.csv" % camera_id)
        output_jsonl = output_scene_root / ("%s_global_records.jsonl" % camera_id)
        if output_csv.exists() and not options["overwrite"]:
            rows.append(_skipped_row(subset, scene_name, camera_id, output_csv))
            continue
        row = propagate_for_camera_file(
            local_tracks_csv=local_file,
            candidates_with_global_ids_path=candidates_path,
            subset=subset,
            output_csv=output_csv,
            output_jsonl=output_jsonl,
            include_unassigned=options["include_unassigned"],
            show_progress=options["progress"],
            namespace_global_ids=options["namespace_global_ids"],
            global_id_stride=options["global_id_stride"],
            drop_invalid_bbox=options["drop_invalid_bbox"],
        )
        rows.append(row)
    return rows


def _export_generic_scenes(scenes: List[Tuple[str, str, str]], options: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for subset, _split, scene_name in _progress_iter(scenes, options["progress"], "final generic exports", "scene"):
        frame_scene_root = Path(options["output_root"]) / "frame_global_records" / subset / scene_name
        files = sorted(frame_scene_root.glob("*_global_records.csv"))
        output_path = Path(options["output_root"]) / "generic_tracking_export" / subset / ("%s.csv" % scene_name)
        row = export_generic_tracking_scene_csv(
            files,
            output_path,
            drop_unassigned=options["drop_unassigned_for_generic_export"],
            drop_invalid_bbox=options["drop_invalid_bbox_for_generic_export"],
        )
        row["subset"] = subset
        rows.append(row)
    return rows


def _validate_outputs(options: Dict[str, Any]) -> None:
    output_root = Path(options["output_root"])
    validation_root = output_root / "validation"
    rows = []
    for path in sorted((output_root / "frame_global_records").rglob("*_global_records.csv")):
        report = validate_global_frame_record_file(path)
        write_validation_report(
            report,
            validation_root
            / "frame_records"
            / _relative_validation_path(output_root / "frame_global_records", path),
        )
        rows.append(_validation_row(path, "frame_records", report))
    for path in sorted((output_root / "generic_tracking_export").rglob("*.csv")):
        report = validate_generic_tracking_export(path)
        write_validation_report(
            report,
            validation_root
            / "generic_export"
            / _relative_validation_path(output_root / "generic_tracking_export", path),
        )
        rows.append(_validation_row(path, "generic_export", report))
    summary = {
        "files": len(rows),
        "num_errors": sum([int(row.get("num_errors", 0)) for row in rows]),
        "num_warnings": sum([int(row.get("num_warnings", 0)) for row in rows]),
        "rows": rows,
    }
    write_summary_json(summary, validation_root / "global_validation_summary.json")
    write_summary_csv(summary, validation_root / "global_validation_summary.csv")
    for subset, subset_rows in _group_rows_by_subset(rows).items():
        write_summary_json(
            {
                "files": len(subset_rows),
                "num_errors": sum([int(row.get("num_errors", 0)) for row in subset_rows]),
                "num_warnings": sum([int(row.get("num_warnings", 0)) for row in subset_rows]),
                "rows": subset_rows,
            },
            validation_root / ("%s_validation.json" % subset),
        )


def _evaluate_outputs(scenes: List[Tuple[str, str, str]], options: Dict[str, Any]) -> None:
    output_root = Path(options["output_root"])
    eval_root = output_root / "eval"
    subsets_with_gt = [item for item in sorted(set([subset for subset, _split, _scene in scenes])) if item != "test"]
    all_records = []
    for subset in subsets_with_gt:
        records = []
        for path in sorted((output_root / "frame_global_records" / subset).rglob("*_global_records.csv")):
            records.extend(read_global_frame_records_file(path))
        all_records.extend(records)
        metrics = evaluate_global_frame_records(records)
        save_final_eval_json(metrics, eval_root / ("%s_eval.json" % subset))
        save_final_eval_csv(metrics, eval_root / ("%s_eval.csv" % subset))
    metrics = evaluate_global_frame_records(all_records)
    save_final_eval_json(metrics, eval_root / "global_eval.json")
    save_final_eval_csv(metrics, eval_root / "global_eval.csv")


def _scenes_from_config(options: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    subset_config = options.get("subsets", {})
    selected_subsets = options.get("selected_subsets")
    selected_scenes = options.get("selected_scenes")
    subset_filter = None if selected_subsets is None else set(selected_subsets)
    scene_filter = None if selected_scenes is None else set(selected_scenes)
    scenes = []
    for subset, data in subset_config.items():
        if subset_filter is not None and subset not in subset_filter:
            continue
        split = str(data.get("split", ""))
        for scene_name in data.get("scenes", []):
            if scene_filter is not None and scene_name not in scene_filter:
                continue
            scenes.append((str(subset), split, str(scene_name)))
    return scenes


def _load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data.get("final_export", data)


def _resolve_options(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    return {
        "root": _value(None, config.get("root"), ""),
        "local_tracks_root": _value(None, config.get("local_tracks_root"), "output/local_tracks/yolo11m_medium_conf001"),
        "global_mtmc_root": _value(None, config.get("global_mtmc_root"), "output/global_mtmc_transition/debug"),
        "output_root": _value(None, config.get("output_root"), "output/final_mvp_exports/debug_transition"),
        "include_unassigned": bool(_value(args.include_unassigned, config.get("include_unassigned"), True)),
        "namespace_global_ids": bool(_value(args.namespace_global_ids, config.get("namespace_global_ids"), True)),
        "global_id_stride": int(_value(args.global_id_stride, config.get("global_id_stride"), 100000)),
        "drop_unassigned_for_generic_export": bool(
            _value(args.drop_unassigned_for_generic_export, config.get("drop_unassigned_for_generic_export"), True)
        ),
        "drop_invalid_bbox": bool(_value(args.drop_invalid_bbox, config.get("drop_invalid_bbox"), True)),
        "drop_invalid_bbox_for_generic_export": bool(
            _value(args.drop_invalid_bbox_for_generic_export, config.get("drop_invalid_bbox_for_generic_export"), True)
        ),
        "progress": bool(_value(args.progress, config.get("progress"), True)),
        "overwrite": bool(args.overwrite),
        "subsets": config.get("subsets", {}),
        "selected_subsets": args.subsets,
        "selected_scenes": args.scenes,
        "camera_ids": args.camera_ids,
    }


def _value(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _skipped_row(subset: str, scene_name: str, camera_id: str, output_csv: Path) -> Dict[str, Any]:
    return {
        "subset": subset,
        "scene_name": scene_name,
        "camera_id": camera_id,
        "input_records": 0,
        "output_records": 0,
        "assigned_records": 0,
        "unassigned_records": 0,
        "unique_local_tracks": 0,
        "unique_global_tracks": 0,
        "output_csv": str(output_csv),
        "status": "skipped_existing",
        "error_message": "",
    }


def _validation_row(path: Path, kind: str, report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(path),
        "kind": kind,
        "subset": _infer_subset(path, kind),
        "num_errors": int(report.get("num_errors", 0)),
        "num_warnings": int(report.get("num_warnings", 0)),
        "status": report.get("status", ""),
    }


def _group_rows_by_subset(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = {}
    for row in rows:
        subset = str(row.get("subset", "unknown"))
        groups.setdefault(subset, []).append(row)
    return groups


def _infer_subset(path: Path, kind: str) -> str:
    parts = list(path.parts)
    marker = "frame_global_records" if kind == "frame_records" else "generic_tracking_export"
    if marker in parts:
        index = parts.index(marker)
        if index + 1 < len(parts):
            return str(parts[index + 1])
    return "unknown"


def _relative_validation_path(root: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    stem = relative.with_suffix("")
    return Path(str(stem) + "_validation.json")


def _progress_iter(values: List[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: %d/%d %s" % (desc, index + 1, total, value))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run final MVP export pipeline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--camera-ids", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    assign_group = parser.add_mutually_exclusive_group()
    assign_group.add_argument("--include-unassigned", dest="include_unassigned", action="store_true", default=None)
    assign_group.add_argument("--drop-unassigned", dest="include_unassigned", action="store_false")
    namespace_group = parser.add_mutually_exclusive_group()
    namespace_group.add_argument("--namespace-global-ids", dest="namespace_global_ids", action="store_true", default=None)
    namespace_group.add_argument("--keep-local-global-ids", dest="namespace_global_ids", action="store_false")
    parser.add_argument("--global-id-stride", type=int, default=None)
    bbox_frame_group = parser.add_mutually_exclusive_group()
    bbox_frame_group.add_argument("--drop-invalid-bbox", dest="drop_invalid_bbox", action="store_true", default=None)
    bbox_frame_group.add_argument("--keep-invalid-bbox", dest="drop_invalid_bbox", action="store_false")
    generic_group = parser.add_mutually_exclusive_group()
    generic_group.add_argument("--drop-unassigned-generic", dest="drop_unassigned_for_generic_export", action="store_true", default=None)
    generic_group.add_argument("--include-unassigned-generic", dest="drop_unassigned_for_generic_export", action="store_false")
    bbox_group = parser.add_mutually_exclusive_group()
    bbox_group.add_argument("--drop-invalid-bbox-generic", dest="drop_invalid_bbox_for_generic_export", action="store_true", default=None)
    bbox_group.add_argument("--keep-invalid-bbox-generic", dest="drop_invalid_bbox_for_generic_export", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=None)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_final_mvp_export(args)


if __name__ == "__main__":
    main()
