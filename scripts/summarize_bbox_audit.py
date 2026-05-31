"""Summarize a bbox audit CSV into readable text."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.detection2d.yolo_bbox_audit import load_bbox_audit_csv
from deep_oc_sort_3d.detection2d.yolo_dataset_exporter import DEFAULT_CLASS_MAPPING


def summarize_bbox_audit_csv(args: Any) -> None:
    """Read audit CSV and write recommendations text."""
    rows = load_bbox_audit_csv(args.csv)
    text = _summary_text(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    print("Wrote %s" % args.output)


def _summary_text(rows: List[Dict[str, Any]]) -> str:
    lines = []
    class_counts = _count(rows, "class_name")
    missing = [name for name in DEFAULT_CLASS_MAPPING.keys() if class_counts.get(name, 0) == 0]
    lines.append("total boxes: %d" % len(rows))
    lines.append("missing classes: %s" % (", ".join(missing) if missing else "none"))
    lines.append("")
    lines.append("classes with majority hard:")
    for class_name in sorted(class_counts.keys()):
        class_rows = [row for row in rows if row["class_name"] == class_name]
        hard = len([row for row in class_rows if row["difficulty"] == "hard"])
        if class_rows and float(hard) / float(len(class_rows)) > 0.5:
            lines.append("  %s: %.2f%% hard" % (class_name, float(hard) / float(len(class_rows)) * 100.0))
    lines.append("")
    lines.append("good cameras per class:")
    for class_name in sorted(class_counts.keys()):
        lines.append("  %s: %s" % (class_name, _top_group_for_class(rows, class_name, "camera_id")))
    lines.append("")
    lines.append("good scenes per class:")
    for class_name in sorted(class_counts.keys()):
        lines.append("  %s: %s" % (class_name, _top_group_for_class(rows, class_name, "scene_name")))
    lines.append("")
    lines.append("preliminary recommendations:")
    lines.append("  easy export: use rows with difficulty=easy and area_norm >= 0.015.")
    lines.append("  medium export: use rows with difficulty in {easy, medium} and area_norm >= 0.004.")
    lines.append("  avoid hard-only camera/class pairs for first detector debugging.")
    return "\n".join(lines)


def _count(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts = {}
    for row in rows:
        key = str(row[field])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _top_group_for_class(rows: List[Dict[str, Any]], class_name: str, group_field: str) -> List[Dict[str, Any]]:
    selected = [row for row in rows if row["class_name"] == class_name and row["difficulty"] in ("easy", "medium")]
    counts = _count(selected, group_field)
    ranked = [{"name": key, "count": value} for key, value in counts.items()]
    return sorted(ranked, key=lambda item: item["count"], reverse=True)[:5]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize bbox audit CSV.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summarize_bbox_audit_csv(args)


if __name__ == "__main__":
    main()

