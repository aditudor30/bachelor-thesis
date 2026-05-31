"""Compare per-class metrics from multiple YOLO evaluation JSON files."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def compare_yolo_models(args: Any) -> None:
    """Write a CSV comparing class precision/recall across models."""
    metrics_list = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    rows = []
    all_classes = sorted(_all_classes(metrics_list))
    baseline = metrics_list[0] if metrics_list else {}
    for model_name, metrics in zip(args.names, metrics_list):
        for class_name in all_classes:
            class_metrics = metrics.get("per_class", {}).get(class_name, {})
            base_metrics = baseline.get("per_class", {}).get(class_name, {})
            rows.append(
                {
                    "model": model_name,
                    "class_name": class_name,
                    "precision": class_metrics.get("precision", 0.0),
                    "recall": class_metrics.get("recall", 0.0),
                    "matches": class_metrics.get("matches", 0),
                    "gt_visible": class_metrics.get("gt_visible", 0),
                    "delta_precision_vs_first": float(class_metrics.get("precision", 0.0))
                    - float(base_metrics.get("precision", 0.0)),
                    "delta_recall_vs_first": float(class_metrics.get("recall", 0.0))
                    - float(base_metrics.get("recall", 0.0)),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "class_name",
                "precision",
                "recall",
                "matches",
                "gt_visible",
                "delta_precision_vs_first",
                "delta_recall_vs_first",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("Wrote %s" % args.output)


def _all_classes(metrics_list: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for metrics in metrics_list:
        names.update(metrics.get("per_class", {}).keys())
    return sorted(names)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare YOLO model metric JSON files.")
    parser.add_argument("--metrics", nargs="+", required=True, type=Path)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_yolo_models(args)


if __name__ == "__main__":
    main()

