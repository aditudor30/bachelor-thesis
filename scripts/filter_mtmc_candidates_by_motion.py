"""Filter MTMC candidates by motion quality."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.mtmc.candidate_motion_filtering import split_candidates_and_metrics
from deep_oc_sort_3d.mtmc.candidate_motion_io import (
    write_candidates_with_motion_csv,
    write_candidates_with_motion_jsonl,
    write_motion_metrics_csv,
)
from deep_oc_sort_3d.mtmc.candidate_motion_quality import (
    CandidateMotionMetrics,
    merge_motion_quality_config,
)
from deep_oc_sort_3d.scripts.audit_candidate_motion_quality import (
    summarize_motion_metrics,
    write_worst_outliers_csv,
)


def filter_mtmc_candidates_by_motion(args: Any) -> None:
    """Filter candidate files by motion quality."""
    config = _resolve_config(args)
    files = _find_candidate_files(
        args.candidate_root,
        args.subsets,
        args.scenes,
        args.camera_ids,
    )
    rows = []
    all_metrics = []
    for path in _progress_iter(files, args.progress, "filter candidate motion files"):
        row, metrics = _process_one_file(path, args, config)
        rows.append(row)
        all_metrics.extend(metrics)
    summary = summarize_motion_metrics(all_metrics)
    summary["files"] = rows
    summary_root = args.output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    (summary_root / "motion_quality_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(summary, summary_root / "motion_quality_summary.csv")
    write_worst_outliers_csv(all_metrics, summary_root / "worst_motion_outliers.csv", top_k=200)
    print("files: %d" % len(files))
    print("errors: %d" % len([row for row in rows if row.get("status") == "error"]))
    print("clean: %s" % summary.get("clean_count"))
    print("Wrote summary: %s" % (summary_root / "motion_quality_summary.json"))


def _process_one_file(path: Path, args: Any, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[CandidateMotionMetrics]]:
    subset, scene_name, camera_id = _parse_candidate_path(args.candidate_root, path)
    output_dir = args.output_root / subset / scene_name
    stem = camera_id
    expected = output_dir / ("%s_clean_candidates.jsonl" % stem)
    if expected.exists() and not args.overwrite:
        return _row(path, output_dir, 0, 0, 0, 0, 0, "skipped_existing", ""), []
    try:
        candidates = read_candidates_file(path)
        buckets, metrics = split_candidates_and_metrics(candidates, config, show_progress=args.progress)
        metrics_by_id = {item.candidate_id: item for item in metrics}
        _write_bucket(output_dir, stem, "clean", buckets["clean"], metrics_by_id)
        _write_bucket(output_dir, stem, "suspicious", buckets["suspicious"], metrics_by_id)
        _write_bucket(output_dir, stem, "invalid", buckets["invalid"], metrics_by_id)
        if buckets["unknown"]:
            _write_bucket(output_dir, stem, "unknown", buckets["unknown"], metrics_by_id)
        write_motion_metrics_csv(metrics, output_dir / ("%s_motion_metrics.csv" % stem))
        return (
            _row(
                path,
                output_dir,
                len(candidates),
                len(buckets["clean"]),
                len(buckets["suspicious"]),
                len(buckets["invalid"]),
                len(buckets["unknown"]),
                "ok",
                "",
            ),
            metrics,
        )
    except Exception as exc:
        return _row(path, output_dir, 0, 0, 0, 0, 0, "error", str(exc)), []


def _write_bucket(output_dir: Path, stem: str, name: str, candidates: List[Any], metrics_by_id: Dict[str, CandidateMotionMetrics]) -> None:
    write_candidates_with_motion_jsonl(
        candidates,
        metrics_by_id,
        output_dir / ("%s_%s_candidates.jsonl" % (stem, name)),
    )
    write_candidates_with_motion_csv(
        candidates,
        metrics_by_id,
        output_dir / ("%s_%s_candidates.csv" % (stem, name)),
    )


def _find_candidate_files(
    root: Path,
    subsets: Optional[List[str]],
    scenes: Optional[List[str]],
    camera_ids: Optional[List[str]],
) -> List[Path]:
    files = sorted(root.rglob("*_candidates.jsonl"))
    if not files:
        files = sorted(root.rglob("*_candidates.csv"))
    subset_set = None if subsets is None else set(subsets)
    scene_set = None if scenes is None else set(scenes)
    camera_set = None if camera_ids is None else set(camera_ids)
    output = []
    for path in files:
        if "summaries" in path.parts:
            continue
        subset, scene_name, camera_id = _parse_candidate_path(root, path)
        if subset_set is not None and subset not in subset_set:
            continue
        if scene_set is not None and scene_name not in scene_set:
            continue
        if camera_set is not None and camera_id not in camera_set:
            continue
        output.append(path)
    return output


def _parse_candidate_path(root: Path, path: Path) -> Tuple[str, str, str]:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if len(parts) < 3:
        return "unknown", "unknown", path.stem.replace("_candidates", "")
    camera = Path(parts[2]).stem.replace("_candidates", "")
    return parts[0], parts[1], camera


def _row(
    input_path: Path,
    output_dir: Path,
    num_candidates: int,
    clean_count: int,
    suspicious_count: int,
    invalid_count: int,
    unknown_count: int,
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "num_candidates": int(num_candidates),
        "clean_count": int(clean_count),
        "suspicious_count": int(suspicious_count),
        "invalid_count": int(invalid_count),
        "unknown_count": int(unknown_count),
        "status": status,
        "error_message": error_message,
    }


def _write_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def _resolve_config(args: Any) -> Dict[str, Any]:
    data = _load_config(args.config)
    if args.require_3d_motion is not None:
        data["require_3d_motion"] = bool(args.require_3d_motion)
    if args.allow_suspicious_as_clean:
        data["allow_suspicious_as_clean"] = True
    return merge_motion_quality_config(data)


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    section = data.get("motion_quality", data)
    return section if isinstance(section, dict) else {}


def _progress_iter(values: List[Any], show_progress: bool, desc: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit="file")


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: file %d/%d %s" % (desc, index + 1, total, value))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Filter MTMC candidates by motion quality.")
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--camera-ids", nargs="+", default=None)
    parser.add_argument("--allow-suspicious-as-clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    req_group = parser.add_mutually_exclusive_group()
    req_group.add_argument("--require-3d-motion", dest="require_3d_motion", action="store_true", default=None)
    req_group.add_argument("--no-require-3d-motion", dest="require_3d_motion", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    filter_mtmc_candidates_by_motion(args)


if __name__ == "__main__":
    main()
