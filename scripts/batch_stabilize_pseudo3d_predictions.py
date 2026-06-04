"""Batch-stabilize isolated pseudo-3D prediction files."""

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List

import yaml

from deep_oc_sort_3d.audit3d.audit3d_io import iter_data_files, progress_iter, write_csv, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilization_io import (
    read_pseudo3d_outputs,
    write_smoothing_report_csv,
    write_smoothing_report_json,
    write_stabilized_outputs_csv,
    write_stabilized_outputs_jsonl,
)
from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilizer import Pseudo3DStabilizer


def run(args: Any) -> Dict[str, Any]:
    """Run batch stabilization from a YAML config."""
    cfg = _load_yaml(args.config)
    batch_cfg = cfg.get("pseudo3d_stabilization", cfg)
    input_root = Path(batch_cfg.get("input_root", "output/pseudo3d/baseline_v2_pseudo3d_isolated/predictions"))
    output_root = Path(batch_cfg.get("output_root", "output/pseudo3d/baseline_v2_pseudo3d_stabilized"))
    stabilizer_config = _load_stabilizer_config(batch_cfg, args.config)
    subsets = [str(item) for item in batch_cfg.get("subsets", [])]
    progress = bool(args.progress if args.progress is not None else cfg.get("progress", True))
    files = _prediction_files(input_root, subsets)
    summaries = []
    for path in progress_iter(files, progress, "stabilize pseudo3D files", "file"):
        summaries.append(_run_one_file(path, input_root, output_root, stabilizer_config, args.overwrite))
    summary = _summarize_batch(summaries)
    write_json(summary, output_root / "smoothing_reports" / "summary_smoothing_report.json")
    write_csv(summaries, output_root / "smoothing_reports" / "summary_smoothing_report.csv")
    write_json(summary.get("source_metadata_completeness", {}), output_root / "summaries" / "source_metadata_completeness.json")
    write_csv(_dict_counts_to_rows(summary.get("per_class", {}), "class_id"), output_root / "summaries" / "per_class_summary.csv")
    write_csv(_dict_counts_to_rows(summary.get("per_subset", {}), "subset"), output_root / "summaries" / "per_subset_summary.csv")
    write_json({"raw_root": str(input_root), "files": [str(path) for path in files]}, output_root / "predictions_raw_reference" / "raw_prediction_manifest.json")
    print("Stabilized files: %s" % len(files))
    print("Camera errors: %s" % summary.get("camera_errors"))
    return summary


def _run_one_file(path: Path, input_root: Path, output_root: Path, config: Dict[str, Any], overwrite: bool) -> Dict[str, Any]:
    relative = path.relative_to(input_root)
    stem = path.stem.replace("_pseudo3d_predictions", "_pseudo3d_stabilized")
    pred_dir = output_root / "predictions_stabilized" / relative.parent
    report_dir = output_root / "smoothing_reports" / relative.parent
    output_jsonl = pred_dir / ("%s.jsonl" % stem)
    output_csv = pred_dir / ("%s.csv" % stem)
    report_json = report_dir / ("%s_smoothing_report.json" % _camera_stem(path.stem))
    report_csv = report_dir / ("%s_smoothing_report.csv" % _camera_stem(path.stem))
    if output_jsonl.exists() and not overwrite:
        return {"status": "skipped", "input": str(path), "output_jsonl": str(output_jsonl)}
    try:
        outputs = read_pseudo3d_outputs(path)
        stabilizer = Pseudo3DStabilizer(config)
        stabilized, report = stabilizer.stabilize_batch(outputs)
        write_stabilized_outputs_jsonl(stabilized, output_jsonl)
        write_stabilized_outputs_csv(stabilized, output_csv)
        write_smoothing_report_json(_compact_report(report), report_json)
        write_smoothing_report_csv(list(report.get("track_reports", [])), report_csv)
        row = _summary_row(report)
        row.update({"status": "ok", "input": str(path), "output_jsonl": str(output_jsonl), "output_csv": str(output_csv), "report_json": str(report_json)})
        return row
    except Exception as exc:
        return {"status": "error", "input": str(path), "error": str(exc), "traceback": traceback.format_exc()}


def _load_stabilizer_config(batch_cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    nested_path = batch_cfg.get("config")
    if nested_path:
        return _load_yaml(Path(nested_path))
    return _load_yaml(config_path)


def _prediction_files(input_root: Path, subsets: List[str]) -> List[Path]:
    files = []
    for path in iter_data_files(input_root, [".jsonl"]):
        if "stabilized" in path.name:
            continue
        if subsets and _subset_from_relative(input_root, path) not in subsets:
            continue
        files.append(path)
    return files


def _subset_from_relative(input_root: Path, path: Path) -> str:
    try:
        return path.relative_to(input_root).parts[0]
    except (ValueError, IndexError):
        return ""


def _camera_stem(stem: str) -> str:
    return stem.replace("_pseudo3d_predictions", "").replace("_pseudo3d_stabilized", "")


def _summary_row(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_predictions": report.get("num_predictions"),
        "num_success": report.get("num_success"),
        "num_failed": report.get("num_failed"),
        "success_rate": report.get("success_rate"),
        "num_tracks": report.get("num_tracks"),
        "num_center_smoothed": report.get("num_center_smoothed"),
        "num_depth_smoothed": report.get("num_depth_smoothed"),
        "num_jump_corrected": report.get("num_jump_corrected"),
        "num_small_bbox_guarded": report.get("num_small_bbox_guarded"),
        "source_metadata_completeness": report.get("source_metadata_completeness", {}),
        "per_class": report.get("per_class", {}),
        "per_subset": report.get("per_subset", {}),
    }


def _compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(report)
    compact["track_report_count"] = len(compact.get("track_reports", []))
    return compact


def _summarize_batch(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    total = sum(int(row.get("num_predictions", 0) or 0) for row in ok_rows)
    failed = sum(int(row.get("num_failed", 0) or 0) for row in ok_rows)
    return {
        "camera_count": len(rows),
        "camera_errors": sum(1 for row in rows if row.get("status") == "error"),
        "camera_skipped": sum(1 for row in rows if row.get("status") == "skipped"),
        "num_predictions": total,
        "num_failed": failed,
        "success_rate": float(total - failed) / float(total) if total else None,
        "num_tracks": sum(int(row.get("num_tracks", 0) or 0) for row in ok_rows),
        "num_center_smoothed": sum(int(row.get("num_center_smoothed", 0) or 0) for row in ok_rows),
        "num_depth_smoothed": sum(int(row.get("num_depth_smoothed", 0) or 0) for row in ok_rows),
        "num_jump_corrected": sum(int(row.get("num_jump_corrected", 0) or 0) for row in ok_rows),
        "num_small_bbox_guarded": sum(int(row.get("num_small_bbox_guarded", 0) or 0) for row in ok_rows),
        "source_metadata_completeness": _aggregate_metadata(ok_rows, total),
        "per_class": _merge_count_dicts([row.get("per_class", {}) for row in ok_rows]),
        "per_subset": _merge_count_dicts([row.get("per_subset", {}) for row in ok_rows]),
    }


def _merge_count_dicts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
    return counts


def _dict_counts_to_rows(counts: Dict[str, int], key_name: str) -> List[Dict[str, Any]]:
    return [{key_name: key, "count": value} for key, value in sorted(counts.items())]


def _aggregate_metadata(rows: List[Dict[str, Any]], total_predictions: int) -> Dict[str, Any]:
    totals = {}
    for row in rows:
        metadata = row.get("source_metadata_completeness", {})
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            if key.endswith("_complete") or key == "is_estimated_for_test_set":
                totals[key] = totals.get(key, 0) + int(value or 0)
    for key, value in list(totals.items()):
        totals["%s_rate" % key] = float(value) / float(total_predictions) if total_predictions else None
    totals["total"] = total_predictions
    return totals


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-stabilize pseudo-3D prediction files.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
