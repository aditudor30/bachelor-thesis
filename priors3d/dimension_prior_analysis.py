"""Compare final 3D dimension priors with generic export dimensions."""

from typing import Any, Dict, List, Optional, Sequence

from deep_oc_sort_3d.audit3d.audit3d_io import finite_float, group_rows, numeric_stats, optional_float


DIMENSION_KEY_MAP = {
    "width": ["width_3d", "width"],
    "length": ["length_3d", "length"],
    "height": ["height_3d", "height"],
}


def compare_priors_to_generic_rows(
    priors: Dict[str, Any],
    generic_rows: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare observed generic dimensions with final class priors."""
    cfg = config or {}
    compare_subsets = cfg.get("compare_subsets", ["official_val", "internal_holdout", "test"])
    low = float(cfg.get("dimension_ratio_warning_low", 0.5))
    high = float(cfg.get("dimension_ratio_warning_high", 2.0))
    rows = []
    grouped = group_rows(generic_rows, ["subset", "class_id"])
    for subset_name in compare_subsets:
        for class_item in priors.get("classes", []):
            class_id = class_item.get("class_id")
            group = grouped.get((subset_name, class_id), [])
            rows.append(_compare_one_group(str(subset_name), class_item, group, low, high))
    return {
        "row_count": len(rows),
        "dimension_ratio_warning_low": low,
        "dimension_ratio_warning_high": high,
        "rows": rows,
        "warning_counts": _warning_counts(rows),
    }


def comparison_to_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return CSV rows from comparison summary."""
    return list(summary.get("rows", []))


def build_dimension_comparison_report(summary: Dict[str, Any]) -> str:
    """Build Markdown report for dimension prior versus generic/test comparison."""
    lines = [
        "# Dimension Prior vs Generic/Test Comparison",
        "",
        "This report compares final class priors against dimensions already present in generic exports.",
        "",
        "## Warning Counts",
        "",
    ]
    warning_counts = summary.get("warning_counts", {})
    if warning_counts:
        for warning, count in sorted(warning_counts.items()):
            lines.append("- `%s`: %s" % (warning, count))
    else:
        lines.append("No warning counts were produced.")
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "Ratios near 1.0 mean generic/test dimensions are close to the selected class prior. "
        "Very low or high ratios suggest default mismatches, outliers, or unstable dimensions."
    )
    return "\n".join(lines)


def _compare_one_group(
    subset_name: str,
    prior: Dict[str, Any],
    rows: List[Dict[str, Any]],
    low: float,
    high: float,
) -> Dict[str, Any]:
    out = {
        "subset": subset_name,
        "class_id": prior.get("class_id"),
        "class_name": prior.get("class_name"),
        "record_count": len(rows),
        "records_close_to_prior": 0,
        "records_extreme": 0,
        "warnings": "",
    }
    warnings = []
    close_count = 0
    extreme_count = 0
    for dim in ["width", "length", "height"]:
        values = [_first_finite(row, DIMENSION_KEY_MAP[dim]) for row in rows]
        finite_values = [value for value in values if value is not None]
        stats = numeric_stats(finite_values)
        prior_value = optional_float(prior.get("robust_%s" % dim))
        ratio_median = _ratio(stats.get("median"), prior_value)
        ratio_p05 = _ratio(stats.get("p05"), prior_value)
        ratio_p95 = _ratio(stats.get("p95"), prior_value)
        cv = _cv(stats.get("mean"), stats.get("std"))
        out["%s_prior" % dim] = prior_value
        out["%s_median" % dim] = stats.get("median")
        out["%s_p05" % dim] = stats.get("p05")
        out["%s_p95" % dim] = stats.get("p95")
        out["%s_cv" % dim] = cv
        out["%s_median_ratio_to_prior" % dim] = ratio_median
        out["%s_p05_ratio_to_prior" % dim] = ratio_p05
        out["%s_p95_ratio_to_prior" % dim] = ratio_p95
        if ratio_median is not None and (ratio_median < low or ratio_median > high):
            warnings.append("%s_median_ratio_outside_range" % dim)
        if cv is not None and cv > 0.5:
            warnings.append("%s_high_cv" % dim)
    for row in rows:
        record_status = _record_ratio_status(row, prior, low, high)
        if record_status == "close":
            close_count += 1
        elif record_status == "extreme":
            extreme_count += 1
    out["records_close_to_prior"] = close_count
    out["records_extreme"] = extreme_count
    out["warnings"] = ";".join(sorted(set(warnings)))
    return out


def _record_ratio_status(row: Dict[str, Any], prior: Dict[str, Any], low: float, high: float) -> str:
    ratios = []
    for dim in ["width", "length", "height"]:
        value = _first_finite(row, DIMENSION_KEY_MAP[dim])
        prior_value = optional_float(prior.get("robust_%s" % dim))
        ratio = _ratio(value, prior_value)
        if ratio is not None:
            ratios.append(ratio)
    if not ratios:
        return "unknown"
    if all(abs(ratio - 1.0) <= 0.05 for ratio in ratios):
        return "close"
    if any(ratio < low or ratio > high for ratio in ratios):
        return "extreme"
    return "normal"


def _first_finite(row: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _ratio(value: Any, prior_value: Any) -> Optional[float]:
    value_f = optional_float(value)
    prior_f = optional_float(prior_value)
    if value_f is None or prior_f is None or abs(prior_f) < 1e-12:
        return None
    return value_f / prior_f


def _cv(mean: Any, std: Any) -> Optional[float]:
    mean_f = optional_float(mean)
    std_f = optional_float(std)
    if mean_f is None or std_f is None or abs(mean_f) < 1e-12:
        return None
    return abs(std_f / mean_f)


def _warning_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {}
    for row in rows:
        for warning in str(row.get("warnings", "")).split(";"):
            if not warning:
                continue
            counts[warning] = counts.get(warning, 0) + 1
    return counts
