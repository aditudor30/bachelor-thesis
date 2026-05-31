"""Compare two or more YOLO curriculum summary JSON files."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def compare_yolo_curricula(args: Any) -> None:
    """Write a CSV comparing curriculum summaries."""
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    names = list(args.names) if args.names is not None else _default_names(args.summaries, summaries)
    if len(names) != len(summaries):
        raise ValueError("--names must have the same length as --summaries.")
    rows = []
    for metric in ["total_images", "total_labels", "total_objects", "person_only_frames", "rare_class_frames"]:
        rows.append(_wide_row("total", metric, summaries, names, metric))
    for section in [
        "per_class_counts",
        "per_class_images",
        "per_scene_counts",
        "per_camera_counts",
        "per_difficulty_counts",
    ]:
        for key in _all_keys(summaries, section):
            rows.append(_wide_row(section, key, summaries, names, section))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["section", "key"] + names
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("Wrote %s" % args.output)


def _wide_row(section: str, key: str, summaries: List[Dict[str, Any]], names: List[str], source: str) -> Dict[str, Any]:
    row = {"section": section, "key": key}
    for name, summary in zip(names, summaries):
        if section == "total":
            row[name] = summary.get(key, 0)
        else:
            row[name] = summary.get(source, {}).get(key, 0)
    return row


def _all_keys(summaries: List[Dict[str, Any]], section: str) -> List[str]:
    keys = set()
    for summary in summaries:
        keys.update(summary.get(section, {}).keys())
    return sorted(keys)


def _default_names(paths: List[Path], summaries: List[Dict[str, Any]]) -> List[str]:
    names = []
    for path, summary in zip(paths, summaries):
        curriculum = summary.get("curriculum")
        names.append(str(curriculum) if curriculum else path.parent.name)
    return names


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare YOLO curriculum summary JSON files.")
    parser.add_argument("--summaries", nargs="+", required=True, type=Path)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_yolo_curricula(args)


if __name__ == "__main__":
    main()
