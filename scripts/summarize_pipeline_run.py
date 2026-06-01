"""Summarize an existing detection-to-observation pipeline run."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def summarize_pipeline_run(args: Any) -> None:
    """Print compact summaries from a pipeline run root."""
    summaries = args.run_root / "summaries"
    inference_rows = _read_csv(summaries / "inference_summary.csv")
    observation_rows = _read_csv(summaries / "observations_summary.csv")
    per_class = _read_csv(summaries / "per_class_summary.csv")

    print("run_root: %s" % args.run_root)
    if inference_rows:
        print("detections: %d" % sum(_int(row.get("num_detections")) for row in inference_rows))
        print("inference cameras: %d" % len(inference_rows))
        print("inference errors: %d" % len([row for row in inference_rows if row.get("status") == "error"]))
    if observation_rows:
        print("observations: %d" % sum(_int(row.get("num_observations")) for row in observation_rows))
        print("matched_gt: %d" % sum(_int(row.get("matched_gt")) for row in observation_rows))
        print("unmatched: %d" % sum(_int(row.get("unmatched")) for row in observation_rows))
        print("depth_valid: %d" % sum(_int(row.get("depth_valid")) for row in observation_rows))
        print("observation errors: %d" % len([row for row in observation_rows if row.get("status") == "error"]))
        print("per subset: %s" % json.dumps(_count_by_subset(observation_rows), sort_keys=True))
    if per_class:
        print("per class:")
        for row in per_class:
            print(
                "  %s/%s: obs=%s matched=%s unmatched=%s mean_iou=%s depth_valid=%s"
                % (
                    row.get("subset"),
                    row.get("class_name"),
                    row.get("observations"),
                    row.get("matched_gt"),
                    row.get("unmatched"),
                    row.get("mean_iou"),
                    row.get("depth_valid"),
                )
            )


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _count_by_subset(rows: List[Dict[str, str]]) -> Dict[str, int]:
    counts = {}
    for row in rows:
        subset = row.get("subset", "")
        counts[subset] = counts.get(subset, 0) + _int(row.get("num_observations"))
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize an existing pipeline run.")
    parser.add_argument("--run-root", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summarize_pipeline_run(args)


if __name__ == "__main__":
    main()
