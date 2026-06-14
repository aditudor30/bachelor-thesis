"""Individual and combined convention hypothesis sweep for Step 23A."""

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from deep_oc_sort_3d.official_failure_audit.failure_io import progress_iter, write_csv, write_json
from deep_oc_sort_3d.official_failure_audit.hypothesis_transforms import individual_definitions, transform_rows
from deep_oc_sort_3d.official_failure_audit.local_matcher import evaluate_predictions, summary_row
from deep_oc_sort_3d.official_failure_audit.track1_parser import AuditTrack1Row


def write_original_matching(
    predictions: Sequence[AuditTrack1Row], ground_truth: Sequence[AuditTrack1Row],
    config: Dict[str, Any], output_root: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    operations = _original_operations()
    transformed = transform_rows(predictions, operations, config)
    summary, details = evaluate_predictions(transformed, ground_truth, config, "original")
    directory = output_root / "matching_original"
    write_json(directory / "original_match_summary.json", summary)
    write_csv(directory / "original_per_scene.csv", _rate_rows(summary.get("per_scene_match_rate", {}), "scene_id"))
    write_csv(directory / "original_per_class.csv", _rate_rows(summary.get("per_class_match_rate", {}), "class_id"))
    samples = sorted(
        details,
        key=lambda row: (bool(row.get("matched")), -(float(row.get("center_error") or 0.0))),
    )[:500]
    write_csv(directory / "original_error_samples.csv", samples)
    return summary, details


def run_hypothesis_sweep(
    predictions: Sequence[AuditTrack1Row], ground_truth: Sequence[AuditTrack1Row],
    config: Dict[str, Any], output_root: Path, progress: bool = True,
) -> Dict[str, Any]:
    directory = output_root / "hypothesis_sweep"
    individual: List[Dict[str, Any]] = []
    definitions = individual_definitions(config)
    for definition in progress_iter(definitions, progress, "23A individual hypotheses"):
        transformed = transform_rows(predictions, definition["operations"], config)
        summary, _details = evaluate_predictions(transformed, ground_truth, config, definition["name"])
        individual.append(summary_row(summary, definition["category"], definition["operations"]))
    individual = sorted(individual, key=_rank_key)
    write_csv(directory / "individual_hypotheses_summary.csv", individual)
    write_json(directory / "individual_hypotheses_summary.json", {"hypotheses": individual})

    combined_definitions = _combined_definitions(individual, config)
    combined: List[Dict[str, Any]] = []
    for definition in progress_iter(combined_definitions, progress, "23A combined hypotheses"):
        transformed = transform_rows(predictions, definition["operations"], config)
        summary, _details = evaluate_predictions(transformed, ground_truth, config, definition["name"])
        combined.append(summary_row(summary, "combined", definition["operations"]))
    combined = sorted(combined, key=_rank_key)
    write_csv(directory / "combined_hypotheses_summary.csv", combined)
    write_json(directory / "combined_hypotheses_summary.json", {"hypotheses": combined})

    ranked = _deduplicate_ranked(sorted(individual + combined, key=_rank_key))
    top_k = int(config.get("hypothesis_sweep", {}).get("run_combined_top_k", 10))
    top = ranked[:max(10, top_k)]
    best = top[0] if top else {}
    write_json(directory / "best_hypothesis.json", best)
    write_csv(directory / "top_hypotheses.csv", top[:10])
    return {"individual": individual, "combined": combined, "top": top, "best": best}


def _combined_definitions(
    individual: Sequence[Dict[str, Any]], config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in individual:
        by_category[str(row.get("category"))].append(row)
    for category in by_category:
        by_category[category] = sorted(
            by_category[category], key=lambda row, name=category: _category_rank_key(name, row),
        )
    best = {category: _operation(row) for category, rows in by_category.items() for row in rows[:1]}
    templates = [
        ("best_axis_best_dimension", ["axis", "dimension"]),
        ("best_axis_best_center", ["axis", "center"]),
        ("best_axis_best_yaw", ["axis", "yaw"]),
        ("best_dimension_best_yaw", ["dimension", "yaw"]),
        ("best_axis_best_dimension_best_yaw", ["axis", "dimension", "yaw"]),
        ("best_axis_best_center_best_dimension_best_yaw", ["axis", "center", "dimension", "yaw"]),
        ("best_frame_best_axis_best_dimension", ["frame", "axis", "dimension"]),
        ("best_class_best_axis_best_dimension", ["class", "axis", "dimension"]),
    ]
    output: List[Dict[str, Any]] = []
    seen = set()
    for name, categories in templates:
        operations = {category: best[category] for category in categories if category in best}
        _append_unique(output, seen, name, operations)

    promising: List[Tuple[str, str]] = []
    for category in ["axis", "center", "dimension", "yaw", "frame", "class"]:
        for row in by_category.get(category, [])[:2]:
            promising.append((category, _operation(row)))
    maximum = int(config.get("hypothesis_sweep", {}).get("max_combined_hypotheses", 50))
    for size in [2, 3, 4]:
        for values in itertools.combinations(promising, size):
            categories = [item[0] for item in values]
            if len(set(categories)) != len(categories):
                continue
            operations = {category: operation for category, operation in values}
            name = "combined:" + "+".join("%s=%s" % item for item in values)
            _append_unique(output, seen, name, operations)
            if len(output) >= maximum:
                return output
    return output[:maximum]


def _append_unique(
    output: List[Dict[str, Any]], seen: Set[Tuple[Tuple[str, str], ...]], name: str,
    operations: Dict[str, str],
) -> None:
    if len(operations) < 2:
        return
    key = tuple(sorted(operations.items()))
    if key in seen:
        return
    seen.add(key)
    output.append({"name": name, "operations": operations})


def _operation(row: Dict[str, Any]) -> str:
    operations = row.get("operations", {})
    if isinstance(operations, dict) and operations:
        return str(next(iter(operations.values())))
    return ""


def _rank_key(row: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, str]:
    match_rate = _number(row.get("match_rate_at_2m"), -1.0)
    median = _number(row.get("center_error_median"), float("inf"))
    nearest = _number(row.get("nearest_distance_median"), float("inf"))
    iou = _number(row.get("iou3d_proxy_median"), -1.0)
    dimension = _dimension_deviation(row)
    yaw = _number(row.get("yaw_error_median"), float("inf"))
    return (-match_rate, median, nearest, -iou, dimension, yaw, str(row.get("hypothesis", "")))


def _category_rank_key(category: str, row: Dict[str, Any]) -> Tuple[Any, ...]:
    if category == "dimension":
        return (_dimension_deviation(row), -_number(row.get("iou3d_proxy_median"), -1.0), str(row.get("hypothesis", "")))
    if category == "yaw":
        return (_number(row.get("yaw_error_median"), float("inf")), str(row.get("hypothesis", "")))
    return _rank_key(row)


def _dimension_deviation(row: Dict[str, Any]) -> float:
    values = [
        _number(row.get("dimension_ratio_width_median"), float("inf")),
        _number(row.get("dimension_ratio_length_median"), float("inf")),
        _number(row.get("dimension_ratio_height_median"), float("inf")),
    ]
    if any(value <= 0.0 or value == float("inf") for value in values):
        return float("inf")
    import math

    return sum(abs(math.log(value)) for value in values)


def _number(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _original_operations() -> Dict[str, str]:
    return {
        "axis": "original", "center": "center_original", "dimension": "w_l_h_original",
        "yaw": "yaw_original", "frame": "frame_original", "class": "official_mapping",
    }


def _deduplicate_ranked(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        effective = _original_operations()
        operations = row.get("operations", {})
        if isinstance(operations, dict):
            effective.update({str(key): str(value) for key, value in operations.items()})
        key = tuple(sorted(effective.items()))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _rate_rows(values: Dict[str, Any], field: str) -> List[Dict[str, Any]]:
    return [{field: key, "match_rate": value} for key, value in sorted(values.items())]
