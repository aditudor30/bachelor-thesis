"""Evaluate split-wide YOLO metrics at multiple confidence thresholds."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.detection2d.yolo_split_eval import evaluate_yolo_dataset_predictions


CSV_FIELDS = [
    "threshold",
    "class_name",
    "detections",
    "gt_visible",
    "matches",
    "precision",
    "recall",
    "mean_iou",
]


def compare_yolo_split_conf_thresholds(args: Any) -> None:
    """Evaluate official-val or holdout detections across confidence thresholds."""
    rows = []
    for threshold in args.thresholds:
        metrics = evaluate_yolo_dataset_predictions(
            root=args.root,
            split=args.split,
            scenes=args.scenes,
            detections_dir=args.detections_dir,
            camera_id=_parse_camera_id(args.camera_id),
            iou_threshold=args.iou_threshold,
            conf_threshold=float(threshold),
            max_frames_per_scene=args.max_frames_per_scene,
            frame_stride=args.frame_stride,
        )
        rows.extend(_metrics_to_rows(float(threshold), metrics))
    _write_csv(rows, args.output)
    print("Wrote %s" % args.output)
    _print_recommendations(rows, args.min_precision)


def _metrics_to_rows(threshold: float, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [_metric_row(threshold, "all", metrics)]
    for class_name, class_metrics in metrics.get("per_class", {}).items():
        rows.append(_metric_row(threshold, class_name, class_metrics))
    return rows


def _metric_row(threshold: float, class_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "threshold": float(threshold),
        "class_name": class_name,
        "detections": int(metrics.get("detections", 0)),
        "gt_visible": int(metrics.get("gt_visible", 0)),
        "matches": int(metrics.get("matches", 0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "mean_iou": metrics.get("mean_iou"),
    }


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_recommendations(rows: List[Dict[str, Any]], min_precision: float) -> None:
    class_names = sorted(set(str(row["class_name"]) for row in rows))
    print("Recommended thresholds by class (min_precision=%.2f):" % float(min_precision))
    for class_name in class_names:
        best = _recommend_threshold([row for row in rows if row["class_name"] == class_name], min_precision)
        if best is None:
            print("  %s: None" % class_name)
        else:
            print(
                "  %s: threshold=%.4f precision=%.4f recall=%.4f matches=%d"
                % (
                    class_name,
                    float(best["threshold"]),
                    float(best["precision"]),
                    float(best["recall"]),
                    int(best["matches"]),
                )
            )


def _recommend_threshold(rows: List[Dict[str, Any]], min_precision: float) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if float(row["precision"]) >= float(min_precision)]
    if not candidates:
        candidates = list(rows)
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (float(row["recall"]), float(row["precision"])), reverse=True)[0]


def _parse_camera_id(value: str) -> Optional[str]:
    if value is None or str(value).lower() == "all":
        return None
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare split-wide YOLO confidence thresholds.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--detections-dir", required=True, type=Path)
    parser.add_argument("--camera-id", default="all")
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-frames-per-scene", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-precision", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_yolo_split_conf_thresholds(args)


if __name__ == "__main__":
    main()
