"""Compare geometry-only and ReID global association runs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def compare_reid_global_runs(args: Any) -> List[Dict[str, Any]]:
    """Compare a baseline run against one or more ReID runs."""
    names = args.names if args.names is not None else [path.name for path in args.runs]
    if len(names) != len(args.runs):
        raise ValueError("--names must have same length as --runs")
    baseline = summarize_global_run("baseline", args.baseline)
    rows = [baseline]
    for name, root in zip(names, args.runs):
        row = summarize_global_run(name, root)
        row.update(_delta_fields(row, baseline))
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set([key for row in rows for key in row.keys()]))
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print("runs: %d" % len(rows))
    print("Wrote %s" % args.output)
    return rows


def summarize_global_run(name: str, root: Path) -> Dict[str, Any]:
    """Aggregate per-scene global summaries under a run root."""
    summary_files = _find_summary_files(root)
    global_tracks = 0
    multi = 0
    singleton = 0
    accepted = 0
    accepted_reid = 0
    transition_accepted = 0
    transition_relation_found = False
    per_class = {}
    per_class_multi = {}
    appearance_values = []
    purity_values = []
    false_merge_rates = []
    fragmentation = 0
    for path in summary_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        global_tracks += int(data.get("global_tracks", 0) or 0)
        multi += int(data.get("multi_camera_tracks", 0) or 0)
        singleton += int(data.get("singleton_tracks", 0) or 0)
        accepted += int(data.get("accepted_edges", 0) or 0)
        accepted_reid += int(data.get("accepted_edges_with_reid", 0) or 0)
        relations = data.get("accepted_edge_temporal_relations", {})
        if isinstance(relations, dict):
            for key, value in relations.items():
                if str(key) != "overlap":
                    transition_accepted += int(value)
                    transition_relation_found = True
        for key, value in data.get("per_class_tracks", {}).items():
            per_class[str(key)] = per_class.get(str(key), 0) + int(value)
        for key, value in data.get("per_class_multi_camera_tracks", data.get("per_class_multi_camera", {})).items():
            per_class_multi[str(key)] = per_class_multi.get(str(key), 0) + int(value)
        if data.get("mean_appearance_distance_accepted") is not None:
            appearance_values.append(float(data.get("mean_appearance_distance_accepted")))
        gt_metrics = data.get("diagnostic_gt_metrics", {})
        if isinstance(gt_metrics, dict):
            if gt_metrics.get("global_purity_mean") is not None:
                purity_values.append(float(gt_metrics.get("global_purity_mean")))
            if gt_metrics.get("false_merge_rate") is not None:
                false_merge_rates.append(float(gt_metrics.get("false_merge_rate")))
            fragmentation += int(gt_metrics.get("fragmentation_approx", 0) or 0)
    if not transition_relation_found:
        transition_accepted = _count_accepted_transition_edges(root)
    return {
        "name": name,
        "root": str(root),
        "summary_files": len(summary_files),
        "num_global_tracks": global_tracks,
        "multi_camera_tracks": multi,
        "singleton_tracks": singleton,
        "accepted_edges": accepted,
        "accepted_edges_with_reid": accepted_reid,
        "transition_edges_accepted": transition_accepted,
        "global_purity_mean": _mean(purity_values),
        "false_merge_rate": _mean(false_merge_rates),
        "fragmentation_approx": fragmentation,
        "mean_appearance_distance_accepted": _mean(appearance_values),
        "per_class_tracks_json": json.dumps(per_class, sort_keys=True),
        "per_class_multi_camera_json": json.dumps(per_class_multi, sort_keys=True),
    }


def _find_summary_files(root: Path) -> List[Path]:
    files = []
    for path in sorted(root.rglob("summary.json")):
        parts = set(path.parts)
        if "summaries" in parts:
            continue
        files.append(path)
    return files


def _count_accepted_transition_edges(root: Path) -> int:
    csv_files = sorted(root.rglob("transition_edges.csv"))
    jsonl_files = sorted(root.rglob("transition_edges.jsonl"))
    if csv_files:
        return _count_accepted_transition_edges_csv(csv_files)
    return _count_accepted_transition_edges_jsonl(jsonl_files)


def _count_accepted_transition_edges_csv(paths: List[Path]) -> int:
    count = 0
    for path in paths:
        if "summaries" in set(path.parts):
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if _truthy(row.get("accepted")):
                    count += 1
    return count


def _count_accepted_transition_edges_jsonl(paths: List[Path]) -> int:
    count = 0
    for path in paths:
        if "summaries" in set(path.parts):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if _truthy(data.get("accepted")):
                count += 1
    return count


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in ("true", "1", "yes")


def _delta_fields(row: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    purity_delta = _delta(row.get("global_purity_mean"), baseline.get("global_purity_mean"))
    fragmentation_delta = _delta(row.get("fragmentation_approx"), baseline.get("fragmentation_approx"))
    multi_delta = _delta(row.get("multi_camera_tracks"), baseline.get("multi_camera_tracks"))
    return {
        "delta_multi_camera_tracks": multi_delta,
        "delta_global_purity_mean": purity_delta,
        "delta_false_merge_rate": _delta(row.get("false_merge_rate"), baseline.get("false_merge_rate")),
        "delta_fragmentation_approx": fragmentation_delta,
        "purity_change": _label_delta(purity_delta, higher_is_better=True),
        "fragmentation_change": _label_delta(fragmentation_delta, higher_is_better=False),
        "multi_camera_change": _label_delta(multi_delta, higher_is_better=True),
        "per_class_delta_json": _per_class_delta(row, baseline),
    }


def _per_class_delta(row: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    current = _json_dict(row.get("per_class_tracks_json"))
    base = _json_dict(baseline.get("per_class_tracks_json"))
    keys = sorted(set(list(current.keys()) + list(base.keys())))
    delta = {key: int(current.get(key, 0)) - int(base.get(key, 0)) for key in keys}
    return json.dumps(delta, sort_keys=True)


def _json_dict(value: Any) -> Dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def _delta(current: Any, baseline: Any) -> Optional[float]:
    if current in (None, "") or baseline in (None, ""):
        return None
    return float(current) - float(baseline)


def _label_delta(delta: Any, higher_is_better: bool) -> str:
    if delta is None:
        return "unknown"
    if abs(float(delta)) <= 1e-12:
        return "same"
    improved = float(delta) > 0.0 if higher_is_better else float(delta) < 0.0
    return "better" if improved else "worse"


def _mean(values: List[float]) -> Any:
    if not values:
        return None
    return sum(values) / float(len(values))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare geometry-only and ReID global MTMC runs.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    compare_reid_global_runs(args)


if __name__ == "__main__":
    main()
