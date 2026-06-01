"""Compare per-class summaries from multiple pipeline runs."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple


FIELDS = [
    "subset",
    "class_name",
    "metric",
    "best_run",
    "delta_first_to_second",
]


def compare_pipeline_runs(args: Any) -> None:
    """Compare per-class summary CSV files across pipeline runs."""
    names = list(args.names)
    if len(names) != len(args.runs):
        raise ValueError("--names must match --runs length")
    summaries = []
    for run_root in args.runs:
        summaries.append(_load_summary(run_root / "summaries" / "per_class_summary.csv"))
    rows = []
    for subset, class_name in _all_keys(summaries):
        for metric in [
            "detections",
            "matched_gt",
            "recall_proxy",
            "unmatched",
            "mean_iou",
            "depth_valid",
            "center_3d_available",
        ]:
            values = [_float(summary.get((subset, class_name), {}).get(metric)) for summary in summaries]
            best_index = _best_index(metric, values)
            delta = ""
            if len(values) >= 2:
                delta = values[1] - values[0]
            row = {
                "subset": subset,
                "class_name": class_name,
                "metric": metric,
                "best_run": names[best_index] if best_index is not None else "",
                "delta_first_to_second": delta,
            }
            for name, value in zip(names, values):
                row[name] = value
            rows.append(row)
    fields = list(FIELDS) + names
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("Wrote %s" % args.output)


def _load_summary(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["recall_proxy"] = _recall_proxy(row)
            rows[(row.get("subset", ""), row.get("class_name", ""))] = row
    return rows


def _all_keys(summaries: List[Dict[Tuple[str, str], Dict[str, str]]]) -> List[Tuple[str, str]]:
    keys = set()
    for summary in summaries:
        keys.update(summary.keys())
    return sorted(keys)


def _float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _recall_proxy(row: Dict[str, str]) -> str:
    gt_visible = row.get("gt_visible")
    if gt_visible not in (None, ""):
        value = float(gt_visible)
        if value > 0.0:
            return str(float(row.get("matched_gt", 0.0)) / value)
    return ""


def _best_index(metric: str, values: List[float]) -> Any:
    if not values:
        return None
    if metric == "unmatched":
        return min(range(len(values)), key=lambda index: values[index])
    return max(range(len(values)), key=lambda index: values[index])


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare pipeline run per-class summaries.")
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_pipeline_runs(args)


if __name__ == "__main__":
    main()
