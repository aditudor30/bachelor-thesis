"""Validate final MVP export outputs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from deep_oc_sort_3d.final_export.export_validation import (
    validate_generic_tracking_export,
    validate_global_frame_record_file,
    write_validation_report,
)


def validate_final_export(args: Any) -> None:
    """Validate frame_global_records and generic_tracking_export folders."""
    frame_files = sorted((args.export_root / "frame_global_records").rglob("*_global_records.csv"))
    generic_files = sorted((args.export_root / "generic_tracking_export").rglob("*.csv"))
    rows = []
    for path in _progress_iter(frame_files, args.progress, "validate frame records", "file"):
        report = validate_global_frame_record_file(path)
        report_path = args.output_root / "frame_records" / _relative_validation_path(
            args.export_root / "frame_global_records",
            path,
        )
        write_validation_report(report, report_path)
        rows.append(_row(path, "frame_records", report))
    for path in _progress_iter(generic_files, args.progress, "validate generic exports", "file"):
        report = validate_generic_tracking_export(path)
        report_path = args.output_root / "generic_export" / _relative_validation_path(
            args.export_root / "generic_tracking_export",
            path,
        )
        write_validation_report(report, report_path)
        rows.append(_row(path, "generic_export", report))
    summary = _summary(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "global_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_rows_csv(rows, args.output_root / "global_validation_summary.csv")
    for subset, subset_rows in _group_rows_by_subset(rows).items():
        subset_summary = _summary(subset_rows)
        (args.output_root / ("%s_validation.json" % subset)).write_text(
            json.dumps(subset_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print("files: %d" % len(rows))
    print("errors: %s" % summary.get("num_errors"))
    print("warnings: %s" % summary.get("num_warnings"))
    if args.fail_on_errors and int(summary.get("num_errors", 0)) > 0:
        raise SystemExit(1)


def _row(path: Path, kind: str, report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(path),
        "kind": kind,
        "subset": _infer_subset(path, kind),
        "num_errors": int(report.get("num_errors", 0)),
        "num_warnings": int(report.get("num_warnings", 0)),
        "status": report.get("status", ""),
    }


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "files": len(rows),
        "num_errors": sum([int(row.get("num_errors", 0)) for row in rows]),
        "num_warnings": sum([int(row.get("num_warnings", 0)) for row in rows]),
        "rows": rows,
    }


def _write_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = ["path", "kind", "subset", "num_errors", "num_warnings", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    parser = argparse.ArgumentParser(description="Validate final MVP export.")
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--fail-on-errors", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_final_export(args)


if __name__ == "__main__":
    main()
