"""Compare overlap-only and transition-enabled global MTMC runs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def compare_overlap_vs_transition(args: Any) -> None:
    """Compare two global association runs."""
    overlap = _summarize_run("overlap", args.overlap_run)
    transition = _summarize_run("transition", args.transition_run)
    rows = [overlap, transition, _delta_row(overlap, transition)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "summary_files",
        "global_tracks",
        "multi_camera_tracks",
        "singleton_tracks",
        "accepted_edges",
        "rejected_edges",
        "purity_mean",
        "false_merge_rate",
        "fragmentation_approx",
        "per_class_tracks_json",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("overlap multi-camera: %s" % overlap.get("multi_camera_tracks"))
    print("transition multi-camera: %s" % transition.get("multi_camera_tracks"))
    print("overlap purity: %s" % overlap.get("purity_mean"))
    print("transition purity: %s" % transition.get("purity_mean"))
    print("Wrote %s" % args.output)


def _summarize_run(name: str, root: Path) -> Dict[str, Any]:
    files = sorted(root.rglob("summary.json"))
    output = {
        "name": name,
        "summary_files": len(files),
        "global_tracks": 0,
        "multi_camera_tracks": 0,
        "singleton_tracks": 0,
        "accepted_edges": 0,
        "rejected_edges": 0,
        "purity_mean": None,
        "false_merge_rate": None,
        "fragmentation_approx": 0,
        "per_class_tracks_json": "{}",
    }
    purities = []
    false_rates = []
    per_class = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        output["global_tracks"] += int(data.get("global_tracks", 0))
        output["multi_camera_tracks"] += int(data.get("multi_camera_tracks", 0))
        output["singleton_tracks"] += int(data.get("singleton_tracks", 0))
        output["accepted_edges"] += int(data.get("accepted_edges", 0))
        output["rejected_edges"] += int(data.get("rejected_edges", 0))
        for key, value in data.get("per_class_tracks", {}).items():
            per_class[str(key)] = per_class.get(str(key), 0) + int(value)
        metrics = data.get("diagnostic_gt_metrics", {})
        if isinstance(metrics, dict):
            if metrics.get("global_purity_mean") is not None:
                purities.append(float(metrics.get("global_purity_mean")))
            if metrics.get("false_merge_rate") is not None:
                false_rates.append(float(metrics.get("false_merge_rate")))
            output["fragmentation_approx"] += int(metrics.get("fragmentation_approx", 0) or 0)
    output["purity_mean"] = _mean(purities)
    output["false_merge_rate"] = _mean(false_rates)
    output["per_class_tracks_json"] = json.dumps(per_class, sort_keys=True)
    return output


def _delta_row(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    row = {"name": "delta_transition_minus_overlap", "summary_files": ""}
    for key in ["global_tracks", "multi_camera_tracks", "singleton_tracks", "accepted_edges", "rejected_edges", "fragmentation_approx"]:
        row[key] = int(right.get(key, 0)) - int(left.get(key, 0))
    for key in ["purity_mean", "false_merge_rate"]:
        if left.get(key) is None or right.get(key) is None:
            row[key] = None
        else:
            row[key] = float(right.get(key)) - float(left.get(key))
    row["per_class_tracks_json"] = ""
    return row


def _mean(values: List[float]) -> Any:
    if not values:
        return None
    return sum(values) / float(len(values))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare overlap-only and transition-enabled MTMC runs.")
    parser.add_argument("--overlap-run", required=True, type=Path)
    parser.add_argument("--transition-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_overlap_vs_transition(args)


if __name__ == "__main__":
    main()
