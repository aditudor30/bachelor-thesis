"""Sanity checks for the current SmartSpaces 3D MTMC MVP outputs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


REQUIRED_PATH_KEYS = [
    "pipeline_run_root",
    "local_tracks_root",
    "tracklets_root",
    "mtmc_candidates_root",
    "motion_clean_candidates_root",
    "global_mtmc_root",
    "final_export_root",
]


def load_mvp_baseline_config(path: Path) -> Dict[str, Any]:
    """Load an MVP baseline config from YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("MVP config must be a YAML mapping.")
    config = data.get("mvp", data)
    if not isinstance(config, dict):
        raise ValueError("MVP config section must be a mapping.")
    return config


def validate_mvp_baseline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the baseline config shape without checking filesystem outputs."""
    errors = []
    warnings = []
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        errors.append("missing_paths_section")
        paths = {}
    for key in REQUIRED_PATH_KEYS:
        if not paths.get(key):
            errors.append("missing_path_key:%s" % key)
    if not config.get("subsets"):
        errors.append("missing_subsets")
    if not config.get("classes"):
        warnings.append("missing_classes")
    final_export = config.get("final_export", {})
    if not isinstance(final_export, dict):
        errors.append("missing_final_export_section")
    elif final_export.get("official_track1_export") != "todo_until_schema_confirmed":
        warnings.append("track1_export_schema_should_remain_explicit")
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def check_mvp_outputs(config: Dict[str, Any], show_progress: bool = True) -> Dict[str, Any]:
    """Check that the MVP output folders and summary metrics are internally sane."""
    report = {
        "status": "ok",
        "errors": [],
        "warnings": [],
        "checks": [],
        "summary": {},
    }
    _merge_config_validation(report, validate_mvp_baseline_config(config))
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}

    for key in REQUIRED_PATH_KEYS:
        _check_path(report, key, paths.get(key), "dir", required=True)
    _check_path(report, "yolo_model_path", paths.get("yolo_model_path"), "file", required=False)

    final_root = Path(str(paths.get("final_export_root", "")))
    summary_files = {
        "propagation": final_root / "summaries" / "propagation_summary.json",
        "generic_export": final_root / "summaries" / "export_summary.json",
        "validation": final_root / "validation" / "global_validation_summary.json",
        "eval": final_root / "eval" / "global_eval.json",
    }
    summaries = {}
    for label, path in summary_files.items():
        summaries[label] = _read_json_summary(report, label, path)
    report["summary"]["summaries"] = summaries

    _check_summary_metrics(report, config, summaries)
    generic_files = _expected_generic_export_files(config, final_root)
    generic_scan = _scan_generic_exports(report, generic_files, show_progress=show_progress)
    report["summary"]["generic_export_scan"] = generic_scan

    if report["errors"]:
        report["status"] = "error"
    return report


def print_mvp_output_report(report: Dict[str, Any]) -> None:
    """Print a compact human-readable MVP output report."""
    print("MVP output check status: %s" % report.get("status"))
    print("errors: %d" % len(report.get("errors", [])))
    for item in report.get("errors", []):
        print("  error: %s" % item)
    print("warnings: %d" % len(report.get("warnings", [])))
    for item in report.get("warnings", []):
        print("  warning: %s" % item)
    scan = report.get("summary", {}).get("generic_export_scan", {})
    if scan:
        print("generic files checked: %s" % scan.get("files_checked"))
        print("generic rows counted: %s" % scan.get("rows"))
        print("generic invalid bbox rows: %s" % scan.get("invalid_bbox_rows"))
        print("generic missing global id rows: %s" % scan.get("missing_global_track_id_rows"))
    summaries = report.get("summary", {}).get("summaries", {})
    _print_summary_value(summaries, "propagation", "assignment_ratio")
    _print_summary_value(summaries, "generic_export", "rows_written")
    _print_summary_value(summaries, "validation", "num_errors")
    _print_summary_value(summaries, "eval", "global_id_purity_mean")


def _merge_config_validation(report: Dict[str, Any], validation: Dict[str, Any]) -> None:
    for item in validation.get("errors", []):
        _add_error(report, "config:%s" % item)
    for item in validation.get("warnings", []):
        _add_warning(report, "config:%s" % item)


def _check_path(
    report: Dict[str, Any],
    name: str,
    value: Any,
    expected_kind: str,
    required: bool,
) -> None:
    if value in (None, ""):
        if required:
            _add_error(report, "missing path config value for %s" % name)
        else:
            _add_warning(report, "missing optional path config value for %s" % name)
        return
    path = Path(str(value))
    exists = path.exists()
    kind_ok = (path.is_dir() if expected_kind == "dir" else path.is_file()) if exists else False
    report["checks"].append(
        {
            "name": name,
            "path": str(path),
            "expected_kind": expected_kind,
            "exists": exists,
            "kind_ok": kind_ok,
            "required": required,
        }
    )
    if required and not kind_ok:
        _add_error(report, "%s missing or not a %s: %s" % (name, expected_kind, path))
    elif not required and not kind_ok:
        _add_warning(report, "%s missing or not a %s: %s" % (name, expected_kind, path))


def _read_json_summary(report: Dict[str, Any], label: str, path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        _add_error(report, "missing %s summary: %s" % (label, path))
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _add_error(report, "could not read %s summary %s: %s" % (label, path, exc))
        return None
    if not isinstance(data, dict):
        _add_error(report, "%s summary is not a JSON object: %s" % (label, path))
        return None
    return data


def _check_summary_metrics(
    report: Dict[str, Any],
    config: Dict[str, Any],
    summaries: Dict[str, Optional[Dict[str, Any]]],
) -> None:
    sanity = config.get("sanity_checks", {})
    if not isinstance(sanity, dict):
        sanity = {}
    propagation = summaries.get("propagation") or {}
    validation = summaries.get("validation") or {}
    generic = summaries.get("generic_export") or {}
    eval_summary = summaries.get("eval") or {}

    if propagation:
        assignment_ratio = _optional_float(propagation.get("assignment_ratio"))
        min_assignment_ratio = float(sanity.get("min_assignment_ratio", 0.0))
        if assignment_ratio is None:
            _add_warning(report, "propagation assignment_ratio is missing")
        elif assignment_ratio < min_assignment_ratio:
            _add_warning(report, "assignment_ratio %.6f below expected %.6f" % (assignment_ratio, min_assignment_ratio))
        if int(propagation.get("assigned_records", 0)) <= 0:
            _add_error(report, "propagation assigned_records is zero")

    if generic:
        rows_written = int(generic.get("rows_written", 0))
        if rows_written <= 0:
            _add_error(report, "generic export rows_written is zero")
        expected_files = sanity.get("expected_generic_export_files")
        if expected_files is not None and int(generic.get("files", 0)) != int(expected_files):
            _add_warning(
                report,
                "generic export file count %s differs from expected %s"
                % (generic.get("files"), expected_files),
            )

    if validation:
        num_errors = int(validation.get("num_errors", 0))
        if bool(sanity.get("require_validation_errors_zero", True)) and num_errors != 0:
            _add_error(report, "validation num_errors is %d" % num_errors)
        num_warnings = int(validation.get("num_warnings", 0))
        if num_warnings > 0:
            _add_warning(report, "validation warnings present: %d" % num_warnings)

    if eval_summary:
        purity = _optional_float(eval_summary.get("global_id_purity_mean"))
        min_purity = float(sanity.get("min_global_purity", 0.0))
        if purity is None:
            _add_warning(report, "global eval purity is missing")
        elif purity < min_purity:
            _add_warning(report, "global eval purity %.6f below expected %.6f" % (purity, min_purity))


def _expected_generic_export_files(config: Dict[str, Any], final_root: Path) -> List[Path]:
    subsets = config.get("subsets", {})
    if not isinstance(subsets, dict):
        return []
    files = []
    for subset, subset_config in subsets.items():
        if not isinstance(subset_config, dict):
            continue
        for scene_name in subset_config.get("scenes", []):
            files.append(final_root / "generic_tracking_export" / str(subset) / ("%s.csv" % scene_name))
    return files


def _scan_generic_exports(
    report: Dict[str, Any],
    files: List[Path],
    show_progress: bool,
) -> Dict[str, Any]:
    total_rows = 0
    invalid_bbox_rows = 0
    missing_global_track_id_rows = 0
    missing_files = []
    for path in _progress_iter(files, show_progress, "check generic exports", "file"):
        if not path.exists():
            missing_files.append(str(path))
            continue
        row_count, invalid_bbox, missing_global = _scan_one_generic_csv(path)
        total_rows += row_count
        invalid_bbox_rows += invalid_bbox
        missing_global_track_id_rows += missing_global
    for path in missing_files:
        _add_error(report, "missing generic export file: %s" % path)
    if invalid_bbox_rows > 0:
        _add_error(report, "generic export contains invalid bbox rows: %d" % invalid_bbox_rows)
    if missing_global_track_id_rows > 0:
        _add_error(report, "generic export contains missing global_track_id rows: %d" % missing_global_track_id_rows)
    return {
        "expected_files": len(files),
        "files_checked": len(files) - len(missing_files),
        "missing_files": missing_files,
        "rows": total_rows,
        "invalid_bbox_rows": invalid_bbox_rows,
        "missing_global_track_id_rows": missing_global_track_id_rows,
    }


def _scan_one_generic_csv(path: Path) -> Tuple[int, int, int]:
    rows = 0
    invalid_bbox_rows = 0
    missing_global_track_id_rows = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            if row.get("global_track_id") in (None, ""):
                missing_global_track_id_rows += 1
            if not _valid_bbox(row):
                invalid_bbox_rows += 1
    return rows, invalid_bbox_rows, missing_global_track_id_rows


def _valid_bbox(row: Dict[str, Any]) -> bool:
    try:
        x1 = float(row.get("x1", "nan"))
        y1 = float(row.get("y1", "nan"))
        x2 = float(row.get("x2", "nan"))
        y2 = float(row.get("y2", "nan"))
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


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


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_error(report: Dict[str, Any], message: str) -> None:
    report["errors"].append(message)


def _add_warning(report: Dict[str, Any], message: str) -> None:
    report["warnings"].append(message)


def _print_summary_value(summaries: Dict[str, Any], summary_name: str, key: str) -> None:
    summary = summaries.get(summary_name)
    if isinstance(summary, dict) and key in summary:
        print("%s.%s: %s" % (summary_name, key, summary.get(key)))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Check MVP output folders and summary metrics.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-json", default=None, type=Path)
    parser.add_argument("--fail-on-errors", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_mvp_baseline_config(args.config)
    report = check_mvp_outputs(config, show_progress=args.progress)
    print_mvp_output_report(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print("Wrote %s" % args.output_json)
    if args.fail_on_errors and report.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
