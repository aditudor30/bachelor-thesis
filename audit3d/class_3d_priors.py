"""Class-wise 3D dimension prior audit."""

from typing import Any, Dict, List, Sequence

from deep_oc_sort_3d.audit3d.audit3d_io import (
    finite_dimensions,
    finite_float,
    flatten_stats,
    group_rows,
    numeric_stats,
)


PRIOR_DIMENSION_FIELDS = ["width_3d", "length_3d", "height_3d"]
TRACK1_PRIOR_DIMENSION_FIELDS = ["width", "length", "height"]


def compute_class_dimension_priors(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute width/length/height/yaw priors for each class."""
    normalized = [_normalize_dimension_row(row) for row in rows]
    grouped = group_rows(normalized, ["class_id"])
    classes = {}
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0][0])):
        class_id = key[0]
        class_name = _first_non_empty(group, "class_name")
        class_key = str(class_id)
        dimension_summary = {
            "width": numeric_stats([row.get("width") for row in group]),
            "length": numeric_stats([row.get("length") for row in group]),
            "height": numeric_stats([row.get("height") for row in group]),
            "yaw": numeric_stats([row.get("yaw") for row in group]),
        }
        tuple_summary = _dimension_tuple_summary(group)
        classes[class_key] = {
            "class_id": class_id,
            "class_name": class_name,
            "count": len(group),
            "dimensions": dimension_summary,
            "dimension_tuple_summary": tuple_summary,
            "looks_constant_or_default": bool(tuple_summary.get("constant_ratio") is not None and float(tuple_summary.get("constant_ratio")) >= 0.95),
        }
    return {"class_count": len(classes), "row_count": len(rows), "classes": classes}


def compare_class_priors_between_subsets(priors_a: Dict[str, Any], priors_b: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two class prior summaries, such as val/holdout versus test."""
    rows = []
    classes_a = priors_a.get("classes", {})
    classes_b = priors_b.get("classes", {})
    all_keys = sorted(set(list(classes_a.keys()) + list(classes_b.keys())), key=lambda value: str(value))
    for class_key in all_keys:
        item_a = classes_a.get(class_key, {})
        item_b = classes_b.get(class_key, {})
        row = {
            "class_id": class_key,
            "class_name_a": item_a.get("class_name", ""),
            "class_name_b": item_b.get("class_name", ""),
            "count_a": item_a.get("count", 0),
            "count_b": item_b.get("count", 0),
            "rare_in_a": int(item_a.get("count", 0) or 0) < 20,
            "rare_in_b": int(item_b.get("count", 0) or 0) < 20,
            "constant_or_default_a": item_a.get("looks_constant_or_default", False),
            "constant_or_default_b": item_b.get("looks_constant_or_default", False),
        }
        for field in ["width", "length", "height"]:
            median_a = _prior_stat(item_a, field, "median")
            median_b = _prior_stat(item_b, field, "median")
            row["%s_median_a" % field] = median_a
            row["%s_median_b" % field] = median_b
            row["%s_median_delta" % field] = _delta(median_a, median_b)
            row["%s_p05_a" % field] = _prior_stat(item_a, field, "p05")
            row["%s_p95_a" % field] = _prior_stat(item_a, field, "p95")
            row["%s_p05_b" % field] = _prior_stat(item_b, field, "p05")
            row["%s_p95_b" % field] = _prior_stat(item_b, field, "p95")
        rows.append(row)
    return {"class_count": len(rows), "rows": rows}


def class_priors_to_rows(priors: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten class priors for CSV output."""
    rows = []
    for _class_key, item in sorted(priors.get("classes", {}).items(), key=lambda pair: str(pair[0])):
        row = {
            "class_id": item.get("class_id"),
            "class_name": item.get("class_name"),
            "count": item.get("count"),
            "looks_constant_or_default": item.get("looks_constant_or_default"),
        }
        for field, stats in item.get("dimensions", {}).items():
            row.update(flatten_stats(field, stats))
        tuple_summary = item.get("dimension_tuple_summary", {})
        row["dimension_tuple_unique_count"] = tuple_summary.get("unique_count")
        row["dimension_tuple_constant_ratio"] = tuple_summary.get("constant_ratio")
        row["dimension_tuple_most_common"] = tuple_summary.get("most_common")
        rows.append(row)
    return rows


def comparison_to_rows(comparison: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return comparison rows for CSV output."""
    return list(comparison.get("rows", []))


def split_rows_by_subset(rows: List[Dict[str, Any]], subset_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Filter rows to named subsets."""
    wanted = set([str(name) for name in subset_names])
    return [row for row in rows if str(row.get("subset", "")) in wanted]


def _normalize_dimension_row(row: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(row)
    output["width"] = _first_finite(row, ["width_3d", "width"])
    output["length"] = _first_finite(row, ["length_3d", "length"])
    output["height"] = _first_finite(row, ["height_3d", "height"])
    output["yaw"] = finite_float(row.get("yaw"))
    return output


def _first_finite(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _first_non_empty(rows: List[Dict[str, Any]], key: str) -> str:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _dimension_tuple_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tuples = []
    for row in rows:
        dims = finite_dimensions(row, "width", "length", "height")
        if dims is not None:
            tuples.append(tuple(round(value, 6) for value in dims))
    if not tuples:
        return {"valid_count": 0, "unique_count": 0, "constant_ratio": None, "most_common": None}
    counts = {}
    for dims in tuples:
        counts[dims] = counts.get(dims, 0) + 1
    most_common = max(counts.items(), key=lambda item: item[1])
    return {
        "valid_count": len(tuples),
        "unique_count": len(counts),
        "constant_ratio": float(most_common[1]) / float(len(tuples)),
        "most_common": list(most_common[0]),
    }


def _prior_stat(item: Dict[str, Any], field: str, stat_name: str) -> Any:
    dimensions = item.get("dimensions", {})
    stats = dimensions.get(field, {}) if isinstance(dimensions, dict) else {}
    return stats.get(stat_name) if isinstance(stats, dict) else None


def _delta(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    return float(b) - float(a)

