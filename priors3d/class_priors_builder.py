"""Build final class-wise 3D dimension priors from Step 15A outputs."""

from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.audit3d.audit3d_io import optional_float, optional_int


EXPECTED_CLASSES = [
    {"class_id": 0, "class_name": "Person"},
    {"class_id": 1, "class_name": "Forklift"},
    {"class_id": 2, "class_name": "PalletTruck"},
    {"class_id": 3, "class_name": "Transporter"},
    {"class_id": 4, "class_name": "FourierGR1T2"},
    {"class_id": 5, "class_name": "AgilityDigit"},
    {"class_id": 6, "class_name": "NovaCarter"},
]

DIMENSIONS = ["width", "length", "height"]


def build_final_class_priors(
    prior_json: Optional[Dict[str, Any]] = None,
    prior_csv_rows: Optional[List[Dict[str, Any]]] = None,
    classes: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build final class priors using medians or available aggregate stats.

    This function never invents missing dimension values. Classes without usable
    statistics are emitted with ``fallback_required=True``.
    """
    cfg = config or {}
    class_defs = classes or EXPECTED_CLASSES
    source_rows = _normalize_source_rows(prior_json or {}, prior_csv_rows or [])
    source_by_class = {str(row.get("class_id")): row for row in source_rows}
    final_classes = []
    for class_def in class_defs:
        class_id = int(class_def["class_id"])
        class_name = str(class_def["class_name"])
        source = source_by_class.get(str(class_id), {})
        final_classes.append(_build_one_class_prior(class_id, class_name, source, cfg))
    return {
        "name": "baseline_v1_geometry_only_final_class_priors",
        "source": "step15a_class_priors",
        "robust_method": cfg.get("robust_method", "median"),
        "class_count": len(final_classes),
        "classes": final_classes,
        "classes_by_id": {str(item["class_id"]): item for item in final_classes},
    }


def final_priors_to_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten final priors into CSV rows."""
    rows = []
    for item in summary.get("classes", []):
        row = {
            "class_id": item.get("class_id"),
            "class_name": item.get("class_name"),
            "count": item.get("count"),
            "robust_width": item.get("robust_width"),
            "robust_length": item.get("robust_length"),
            "robust_height": item.get("robust_height"),
            "selected_prior_source": item.get("selected_prior_source"),
            "confidence_level": item.get("confidence_level"),
            "fallback_required": item.get("fallback_required"),
            "looks_constant": item.get("looks_constant"),
            "outlier_flag": item.get("outlier_flag"),
            "notes": item.get("notes"),
        }
        for dim in DIMENSIONS:
            stats = item.get("%s_stats" % dim, {})
            for key in ["mean", "median", "std", "p05", "p95"]:
                row["%s_%s" % (dim, key)] = stats.get(key)
        rows.append(row)
    return rows


def build_class_priors_report(summary: Dict[str, Any]) -> str:
    """Build a Markdown report for final class priors."""
    lines = [
        "# Final Class-wise 3D Dimension Priors",
        "",
        "These priors consolidate Step 15A class dimension statistics. Missing values are not invented.",
        "",
        "## Summary",
        "",
        "- Classes: %s" % summary.get("class_count", 0),
        "- Robust method: `%s`" % summary.get("robust_method", "median"),
        "",
        "## Per-class Priors",
        "",
    ]
    for item in summary.get("classes", []):
        lines.extend(
            [
                "### %s (%s)" % (item.get("class_name"), item.get("class_id")),
                "",
                "- Count: %s" % item.get("count"),
                "- Robust dimensions: width=%s, length=%s, height=%s"
                % (item.get("robust_width"), item.get("robust_length"), item.get("robust_height")),
                "- Confidence: `%s`" % item.get("confidence_level"),
                "- Source: `%s`" % item.get("selected_prior_source"),
                "- Fallback required: `%s`" % item.get("fallback_required"),
                "- Notes: %s" % item.get("notes"),
                "",
            ]
        )
    return "\n".join(lines)


def _normalize_source_rows(prior_json: Dict[str, Any], prior_csv_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if prior_csv_rows:
        return [_normalize_csv_row(row) for row in prior_csv_rows]
    classes = prior_json.get("classes", {})
    rows = []
    if isinstance(classes, dict):
        values = classes.values()
    elif isinstance(classes, list):
        values = classes
    else:
        values = []
    for item in values:
        if isinstance(item, dict):
            rows.append(_normalize_json_class(item))
    return rows


def _normalize_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "class_id": optional_int(row.get("class_id")),
        "class_name": row.get("class_name", ""),
        "count": optional_int(row.get("count")),
        "looks_constant_or_default": _bool(row.get("looks_constant_or_default")),
    }
    for dim in DIMENSIONS:
        out["%s_stats" % dim] = {
            "mean": optional_float(row.get("%s_mean" % dim)),
            "median": optional_float(row.get("%s_median" % dim)),
            "std": optional_float(row.get("%s_std" % dim)),
            "p05": optional_float(row.get("%s_p05" % dim)),
            "p95": optional_float(row.get("%s_p95" % dim)),
        }
    return out


def _normalize_json_class(item: Dict[str, Any]) -> Dict[str, Any]:
    dimensions = item.get("dimensions", {})
    out = {
        "class_id": optional_int(item.get("class_id")),
        "class_name": item.get("class_name", ""),
        "count": optional_int(item.get("count")),
        "looks_constant_or_default": bool(item.get("looks_constant_or_default", False)),
    }
    for dim in DIMENSIONS:
        stats = dimensions.get(dim, {}) if isinstance(dimensions, dict) else {}
        out["%s_stats" % dim] = {
            "mean": optional_float(stats.get("mean")),
            "median": optional_float(stats.get("median")),
            "std": optional_float(stats.get("std")),
            "p05": optional_float(stats.get("p05")),
            "p95": optional_float(stats.get("p95")),
        }
    return out


def _build_one_class_prior(class_id: int, class_name: str, source: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    count = int(source.get("count") or 0)
    robust = {}
    stats_out = {}
    for dim in DIMENSIONS:
        stats = source.get("%s_stats" % dim, {})
        stats_out["%s_stats" % dim] = stats
        robust[dim] = _select_robust_value(stats, config.get("robust_method", "median"))
    fallback_required = any(robust[dim] is None for dim in DIMENSIONS)
    looks_constant = _looks_constant(source, config)
    outlier_flag = _has_outlier_shape(source)
    confidence = _confidence_level(count, fallback_required, config)
    notes = _notes(count, fallback_required, looks_constant, outlier_flag, config)
    result = {
        "class_id": class_id,
        "class_name": class_name,
        "count": count,
        "robust_width": robust["width"],
        "robust_length": robust["length"],
        "robust_height": robust["height"],
        "selected_prior_source": "step15a_median_prior" if not fallback_required else "missing_data_requires_fallback",
        "confidence_level": confidence,
        "fallback_required": fallback_required,
        "looks_constant": looks_constant,
        "outlier_flag": outlier_flag,
        "notes": notes,
    }
    result.update(stats_out)
    return result


def _select_robust_value(stats: Dict[str, Any], method: str) -> Optional[float]:
    median = optional_float(stats.get("median"))
    if method == "median":
        return median
    if method == "trimmed_mean":
        p05 = optional_float(stats.get("p05"))
        p95 = optional_float(stats.get("p95"))
        mean = optional_float(stats.get("mean"))
        values = [value for value in [p05, median, p95, mean] if value is not None]
        if not values:
            return None
        return sum(values) / float(len(values))
    return median


def _looks_constant(source: Dict[str, Any], config: Dict[str, Any]) -> bool:
    if bool(source.get("looks_constant_or_default", False)):
        return True
    threshold = float(config.get("constant_cv_threshold", 0.02))
    valid = 0
    constant = 0
    for dim in DIMENSIONS:
        stats = source.get("%s_stats" % dim, {})
        mean = optional_float(stats.get("mean"))
        std = optional_float(stats.get("std"))
        if mean is None or std is None or abs(mean) < 1e-12:
            continue
        valid += 1
        if abs(std / mean) <= threshold:
            constant += 1
    return valid > 0 and constant == valid


def _has_outlier_shape(source: Dict[str, Any]) -> bool:
    for dim in DIMENSIONS:
        stats = source.get("%s_stats" % dim, {})
        p05 = optional_float(stats.get("p05"))
        p95 = optional_float(stats.get("p95"))
        median = optional_float(stats.get("median"))
        if p05 is None or p95 is None or median is None or median <= 0.0:
            continue
        if p95 / median > 2.0 or p05 / median < 0.5:
            return True
    return False


def _confidence_level(count: int, fallback_required: bool, config: Dict[str, Any]) -> str:
    if fallback_required:
        return "low"
    high = int(config.get("min_count_high_confidence", 1000))
    medium = int(config.get("min_count_medium_confidence", 100))
    if count >= high:
        return "high"
    if count >= medium:
        return "medium"
    return "low"


def _notes(
    count: int,
    fallback_required: bool,
    looks_constant: bool,
    outlier_flag: bool,
    config: Dict[str, Any],
) -> str:
    notes = []
    if fallback_required:
        notes.append("insufficient dimension statistics; fallback required")
    if count < int(config.get("min_count_medium_confidence", 100)):
        notes.append("rare class or low support")
    if looks_constant:
        notes.append("dimensions look near-constant; verify default/class-prior source")
    if outlier_flag:
        notes.append("wide robust range; inspect outliers before using as a hard prior")
    if not notes:
        notes.append("usable class prior")
    return "; ".join(notes)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")
