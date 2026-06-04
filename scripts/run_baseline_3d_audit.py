"""Run the full baseline_v1 geometry-only 3D audit."""

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List

import yaml

from deep_oc_sort_3d.audit3d.audit3d_io import ensure_clean_output_dir, write_csv, write_json, write_markdown
from deep_oc_sort_3d.audit3d.audit3d_report import build_3d_audit_report, build_3d_audit_summary_json
from deep_oc_sort_3d.audit3d.class_3d_priors import (
    class_priors_to_rows,
    compare_class_priors_between_subsets,
    comparison_to_rows,
    compute_class_dimension_priors,
    split_rows_by_subset,
)
from deep_oc_sort_3d.audit3d.generic_3d_audit import (
    compare_generic_vs_track1,
    compute_generic_3d_stats,
    compute_generic_per_class_stats,
    compute_generic_per_scene_stats,
    read_generic_export_rows,
    stats_dict_to_rows as generic_stats_to_rows,
)
from deep_oc_sort_3d.audit3d.projection_3d_audit import audit_projection_batch, projection_failures_to_rows
from deep_oc_sort_3d.audit3d.smoothness_3d_audit import (
    compute_smoothness_audit,
    find_worst_dimension_variation,
    find_worst_jumps,
)
from deep_oc_sort_3d.audit3d.source_3d_audit import audit_3d_sources, write_missing_source_metadata_report
from deep_oc_sort_3d.audit3d.track1_3d_audit import (
    compute_3d_field_stats,
    compute_per_class_3d_stats,
    compute_per_scene_3d_stats,
    detect_extreme_3d_values,
    read_track1_rows,
    stats_dict_to_rows as track1_stats_to_rows,
)


def run_baseline_3d_audit(args: Any) -> Dict[str, Any]:
    """Run all configured 3D audits and write the requested output tree."""
    config = _load_config(args.config)
    output_root = Path(config.get("audit3d", {}).get("output_root", "output/3d_audit/baseline_v1_geometry_only"))
    progress = bool(args.progress if args.progress is not None else config.get("audit3d", {}).get("progress", True))
    ensure_clean_output_dir(output_root, overwrite=args.overwrite)

    paths = config.get("paths", {})
    thresholds = config.get("thresholds", {})
    results = {}

    track1_rows = []
    generic_rows = []

    results["track1"] = _run_step(
        "track1 audit",
        lambda: _run_track1_audit(output_root, paths, thresholds, track1_rows, progress),
    )
    if track1_rows:
        print("Track1 rows loaded: %d" % len(track1_rows))

    results["generic"] = _run_step(
        "generic export audit",
        lambda: _run_generic_audit(output_root, paths, track1_rows, generic_rows, progress),
    )
    if generic_rows:
        print("Generic rows loaded: %d" % len(generic_rows))

    results["smoothness"] = _run_step(
        "smoothness audit",
        lambda: _run_smoothness_audit(output_root, track1_rows, thresholds, progress),
    )

    results["source"] = _run_step(
        "source audit",
        lambda: _run_source_audit(output_root, paths, progress),
    )

    results["class_priors"] = _run_step(
        "class priors",
        lambda: _run_class_priors(output_root, generic_rows, config.get("class_priors", {})),
    )

    results["projection"] = _run_step(
        "projection audit",
        lambda: _run_projection_audit(output_root, paths, config.get("subsets", {}), config.get("projection_audit", {}), progress),
    )
    _write_failed_step_placeholders(output_root, results)

    report_dir = output_root / "report"
    track1_summary = _safe_result(results, "track1")
    generic_summary = _safe_result(results, "generic")
    smoothness_summary = _safe_result(results, "smoothness")
    source_summary = _safe_result(results, "source")
    class_priors_summary = _safe_result(results, "class_priors")
    projection_summary = _safe_result(results, "projection")
    report = build_3d_audit_report(
        track1_summary,
        generic_summary,
        smoothness_summary,
        source_summary,
        class_priors_summary,
        projection_summary,
    )
    summary = build_3d_audit_summary_json(
        track1_summary,
        generic_summary,
        smoothness_summary,
        source_summary,
        class_priors_summary,
        projection_summary,
    )
    summary["step_status"] = {key: value.get("status") for key, value in results.items()}
    write_markdown(report, report_dir / "TRACK1_3D_AUDIT_REPORT.md")
    write_json(summary, report_dir / "TRACK1_3D_AUDIT_SUMMARY.json")
    write_json(results, report_dir / "audit_step_status.json")
    print("Wrote final 3D audit report to %s" % report_dir)
    return summary


def _run_track1_audit(
    output_root: Path,
    paths: Dict[str, Any],
    thresholds: Dict[str, Any],
    row_sink: List[Dict[str, Any]],
    progress: bool,
) -> Dict[str, Any]:
    rows = read_track1_rows(paths.get("track1_path", ""), show_progress=progress)
    row_sink.extend(rows)
    summary = compute_3d_field_stats(rows)
    per_class = compute_per_class_3d_stats(rows)
    per_scene = compute_per_scene_3d_stats(rows)
    extremes = detect_extreme_3d_values(rows, thresholds)
    out = output_root / "track1_audit"
    write_json(summary, out / "track1_3d_field_summary.json")
    write_csv(track1_stats_to_rows(summary), out / "track1_3d_field_summary.csv")
    write_csv(per_class, out / "track1_per_class_summary.csv")
    write_csv(per_scene, out / "track1_per_scene_summary.csv")
    write_csv(extremes, out / "track1_extreme_values.csv")
    return summary


def _run_generic_audit(
    output_root: Path,
    paths: Dict[str, Any],
    track1_rows: List[Dict[str, Any]],
    row_sink: List[Dict[str, Any]],
    progress: bool,
) -> Dict[str, Any]:
    rows = read_generic_export_rows(paths.get("generic_export_root", ""), show_progress=progress)
    row_sink.extend(rows)
    summary = compute_generic_3d_stats(rows)
    comparison = compare_generic_vs_track1(rows, track1_rows)
    summary["generic_vs_track1"] = comparison
    out = output_root / "generic_export_audit"
    write_json(summary, out / "generic_3d_field_summary.json")
    write_csv(generic_stats_to_rows(summary), out / "generic_3d_field_summary.csv")
    write_csv(compute_generic_per_class_stats(rows), out / "generic_per_class_summary.csv")
    write_csv(compute_generic_per_scene_stats(rows), out / "generic_per_scene_summary.csv")
    write_json(comparison, out / "generic_vs_track1_comparison.json")
    return summary


def _run_smoothness_audit(
    output_root: Path,
    track1_rows: List[Dict[str, Any]],
    thresholds: Dict[str, Any],
    progress: bool,
) -> Dict[str, Any]:
    audit = compute_smoothness_audit(track1_rows, thresholds, show_progress=progress)
    per_object = list(audit.get("per_object", []))
    summary = dict(audit)
    summary.pop("per_object", None)
    out = output_root / "smoothness_audit"
    write_json(summary, out / "track_smoothness_summary.json")
    write_csv(per_object, out / "track_smoothness_per_object.csv")
    write_csv(find_worst_jumps(track1_rows), out / "worst_3d_jumps.csv")
    write_csv(find_worst_dimension_variation(track1_rows), out / "worst_dimension_variation.csv")
    return summary


def _run_source_audit(output_root: Path, paths: Dict[str, Any], progress: bool) -> Dict[str, Any]:
    summary = audit_3d_sources(
        paths.get("frame_records_root", ""),
        observations_root=paths.get("observations_root"),
        candidates_root=paths.get("candidates_root"),
        show_progress=progress,
    )
    out = output_root / "source_audit"
    write_json(summary, out / "source_3d_summary.json")
    write_csv(list(summary.get("per_subset", [])), out / "source_3d_per_subset.csv")
    write_missing_source_metadata_report(summary, out / "missing_source_metadata_report.md")
    return summary


def _run_class_priors(output_root: Path, generic_rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    use_subsets = config.get("use_subsets", ["official_val", "internal_holdout"])
    prior_rows = split_rows_by_subset(generic_rows, use_subsets)
    test_rows = split_rows_by_subset(generic_rows, ["test"])
    priors = compute_class_dimension_priors(prior_rows)
    test_priors = compute_class_dimension_priors(test_rows)
    comparison = compare_class_priors_between_subsets(priors, test_priors)
    out = output_root / "class_priors"
    write_json(priors, out / "class_dimension_priors.json")
    write_csv(class_priors_to_rows(priors), out / "class_dimension_priors.csv")
    write_csv(comparison_to_rows(comparison), out / "class_dimension_comparison_test_vs_val.csv")
    return priors


def _run_projection_audit(
    output_root: Path,
    paths: Dict[str, Any],
    subsets: Dict[str, Any],
    projection_config: Dict[str, Any],
    progress: bool,
) -> Dict[str, Any]:
    enabled = bool(projection_config.get("enabled", True))
    out = output_root / "projection_audit"
    if not enabled:
        summary = {"enabled": False, "total_records_checked": 0, "projection_success": 0, "projection_failed": 0}
    else:
        dataset_root_text = str(paths.get("dataset_root", ""))
        dataset_root = Path(dataset_root_text) if dataset_root_text else None
        if dataset_root is None or not dataset_root.exists():
            summary = {
                "enabled": True,
                "error": "missing_dataset_root",
                "dataset_root": dataset_root_text,
                "total_records_checked": 0,
                "projection_success": 0,
                "projection_failed": 0,
                "failure_reasons": {"missing_dataset_root": 1},
            }
        else:
            summary = audit_projection_batch(
                dataset_root,
                paths.get("generic_export_root", ""),
                subsets=subsets,
                projection_config=projection_config,
                show_progress=progress,
            )
    write_json(summary, out / "projection_audit_summary.json")
    write_csv(projection_failures_to_rows(summary), out / "projection_failures.csv")
    (out / "optional_example_images").mkdir(parents=True, exist_ok=True)
    return summary


def _run_step(name: str, func: Any) -> Dict[str, Any]:
    print("Running %s..." % name)
    try:
        result = func()
        print("Finished %s" % name)
        return {"status": "ok", "result": result}
    except Exception as exc:
        print("Failed %s: %s" % (name, exc))
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc(), "result": {}}


def _safe_result(results: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = results.get(key, {})
    result = value.get("result", {})
    return result if isinstance(result, dict) else {}


def _write_failed_step_placeholders(output_root: Path, results: Dict[str, Any]) -> None:
    mapping = {
        "track1": ("track1_audit", "track1_3d_field_summary.json"),
        "generic": ("generic_export_audit", "generic_3d_field_summary.json"),
        "smoothness": ("smoothness_audit", "track_smoothness_summary.json"),
        "source": ("source_audit", "source_3d_summary.json"),
        "class_priors": ("class_priors", "class_dimension_priors.json"),
        "projection": ("projection_audit", "projection_audit_summary.json"),
    }
    for key, value in results.items():
        if value.get("status") != "error" or key not in mapping:
            continue
        folder, filename = mapping[key]
        write_json(
            {
                "status": "error",
                "error": value.get("error", ""),
                "traceback": value.get("traceback", ""),
            },
            output_root / folder / filename,
        )


def _load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full baseline 3D audit.")
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
    run_baseline_3d_audit(args)


if __name__ == "__main__":
    main()
