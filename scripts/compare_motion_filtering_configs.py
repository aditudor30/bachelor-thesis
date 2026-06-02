"""Compare motion filtering summary JSON files."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def compare_motion_filtering_configs(args: Any) -> None:
    """Compare motion filtering configs from summary JSON files."""
    names = args.names or [Path(path).stem for path in args.summaries]
    if len(names) != len(args.summaries):
        raise ValueError("--names must match --summaries length")
    rows = []
    for name, path in zip(names, args.summaries):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_row(name, path, summary))
    _write_rows(rows, args.output)
    print("summaries: %d" % len(rows))
    print("Wrote %s" % args.output)


def _row(name: str, path: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    max_step = summary.get("max_step_stats", {})
    jumps = summary.get("jump_count_stats", {})
    return {
        "name": name,
        "summary": str(path),
        "total_candidates": summary.get("total_candidates"),
        "clean_count": summary.get("clean_count"),
        "motion_good": summary.get("motion_good"),
        "motion_suspicious": summary.get("motion_suspicious"),
        "motion_invalid": summary.get("motion_invalid"),
        "motion_unknown": summary.get("motion_unknown"),
        "max_step_mean": max_step.get("mean"),
        "max_step_p95": max_step.get("p95"),
        "max_step_max": max_step.get("max"),
        "jump_count_mean": jumps.get("mean"),
        "jump_count_max": jumps.get("max"),
        "per_class_flags_json": json.dumps(summary.get("per_class_flag_counts", {}), sort_keys=True),
    }


def _write_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "summary",
        "total_candidates",
        "clean_count",
        "motion_good",
        "motion_suspicious",
        "motion_invalid",
        "motion_unknown",
        "max_step_mean",
        "max_step_p95",
        "max_step_max",
        "jump_count_mean",
        "jump_count_max",
        "per_class_flags_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare motion filtering configs.")
    parser.add_argument("--summaries", required=True, nargs="+", type=Path)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_motion_filtering_configs(args)


if __name__ == "__main__":
    main()
