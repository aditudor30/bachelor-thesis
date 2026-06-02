"""Compare multiple global MTMC association runs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def compare_global_association_runs(args: Any) -> None:
    """Compare run-level summaries."""
    names = args.names if args.names is not None else [path.name for path in args.runs]
    if len(names) != len(args.runs):
        raise ValueError("--names must have same length as --runs")
    rows = []
    for name, root in zip(names, args.runs):
        rows.append(_summarize_run(name, root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "root",
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
    print("runs: %d" % len(rows))
    print("Wrote %s" % args.output)


def _summarize_run(name: str, root: Path) -> Dict[str, Any]:
    files = sorted(root.rglob("summary.json"))
    global_tracks = 0
    multi = 0
    singleton = 0
    accepted = 0
    rejected = 0
    per_class = {}
    purity_values = []
    false_merge_rates = []
    fragmentation = 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        global_tracks += int(data.get("global_tracks", 0))
        multi += int(data.get("multi_camera_tracks", 0))
        singleton += int(data.get("singleton_tracks", 0))
        accepted += int(data.get("accepted_edges", 0))
        rejected += int(data.get("rejected_edges", 0))
        for key, value in data.get("per_class_tracks", {}).items():
            per_class[str(key)] = per_class.get(str(key), 0) + int(value)
        gt_metrics = data.get("diagnostic_gt_metrics", {})
        if isinstance(gt_metrics, dict):
            if gt_metrics.get("global_purity_mean") is not None:
                purity_values.append(float(gt_metrics.get("global_purity_mean")))
            if gt_metrics.get("false_merge_rate") is not None:
                false_merge_rates.append(float(gt_metrics.get("false_merge_rate")))
            fragmentation += int(gt_metrics.get("fragmentation_approx", 0) or 0)
    return {
        "name": name,
        "root": str(root),
        "summary_files": len(files),
        "global_tracks": global_tracks,
        "multi_camera_tracks": multi,
        "singleton_tracks": singleton,
        "accepted_edges": accepted,
        "rejected_edges": rejected,
        "purity_mean": _mean(purity_values),
        "false_merge_rate": _mean(false_merge_rates),
        "fragmentation_approx": fragmentation,
        "per_class_tracks_json": json.dumps(per_class, sort_keys=True),
    }


def _mean(values: List[float]) -> Any:
    if not values:
        return None
    return sum(values) / float(len(values))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare global MTMC association runs.")
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_global_association_runs(args)


if __name__ == "__main__":
    main()
