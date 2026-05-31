"""Find class-rich frames from a bbox audit CSV."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from deep_oc_sort_3d.detection2d.yolo_bbox_audit import load_bbox_audit_csv


FIELDS = [
    "split",
    "scene_name",
    "camera_id",
    "frame_id",
    "class_counts_json",
    "max_area_norm",
    "mean_area_norm",
    "num_target_objects",
    "recommended_for_easy_export",
    "recommended_for_medium_export",
]


def find_class_rich_frames(args: Any) -> None:
    """Rank frames with target classes and sufficiently large boxes."""
    rows = load_bbox_audit_csv(args.audit_csv)
    selected = [
        row
        for row in rows
        if row["class_name"] in set(args.classes)
        and float(row["area_norm"]) >= float(args.min_area_norm)
        and row["difficulty"] in set(args.difficulty)
    ]
    grouped = _group_rows(selected)
    ranked = sorted(
        grouped,
        key=lambda row: (int(row["recommended_for_easy_export"]), row["num_target_objects"], row["max_area_norm"]),
        reverse=True,
    )[: args.top_k]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)
    print("Wrote %d rows to %s" % (len(ranked), args.output))


def _group_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = {}
    for row in rows:
        key = (row["split"], row["scene_name"], row["camera_id"], int(row["frame_id"]))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    result = []
    for key, items in grouped.items():
        areas = [float(item["area_norm"]) for item in items]
        class_counts = {}
        for item in items:
            class_name = str(item["class_name"])
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        difficulties = set(item["difficulty"] for item in items)
        result.append(
            {
                "split": key[0],
                "scene_name": key[1],
                "camera_id": key[2],
                "frame_id": key[3],
                "class_counts_json": json.dumps(class_counts, sort_keys=True),
                "max_area_norm": max(areas) if areas else 0.0,
                "mean_area_norm": sum(areas) / float(len(areas)) if areas else 0.0,
                "num_target_objects": len(items),
                "recommended_for_easy_export": "easy" in difficulties,
                "recommended_for_medium_export": bool(difficulties.intersection(set(["easy", "medium"]))),
            }
        )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find class-rich frames for future curriculum export.")
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--min-area-norm", type=float, default=0.004)
    parser.add_argument("--difficulty", nargs="+", default=["easy", "medium"])
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    find_class_rich_frames(args)


if __name__ == "__main__":
    main()

