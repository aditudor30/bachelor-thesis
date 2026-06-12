"""Stage and dimension retention analysis for Step 21C."""

from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.bytetrack_tuning.tuning_metrics import STAGE_NAMES


def compute_retention(candidate: Any, baseline: Any) -> Optional[float]:
    """Return candidate/baseline when the baseline is positive."""
    try:
        baseline_value = float(baseline)
        if baseline_value <= 0.0:
            return None
        return float(candidate) / baseline_value
    except (TypeError, ValueError):
        return None


def build_stage_retention_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build all-stage retention rows versus V2 current and Step 21B."""
    baselines = metrics.get("baselines", {})
    v2 = baselines.get("baseline_v2_current", {}).get("stage_counts", {})
    step21b = baselines.get("baseline_21b_bytetrack", {}).get("stage_counts", {})
    rows = []
    for variant_name, variant in sorted(metrics.get("variants", {}).items()):
        counts = variant.get("stage_counts", {})
        phase_a_only = variant.get("status") == "phase_a_only"
        v2_counts = baselines.get("baseline_v2_current", {}).get(
            "phase_a_stage_counts" if phase_a_only else "stage_counts",
            v2,
        )
        step21b_counts = baselines.get("baseline_21b_bytetrack", {}).get(
            "phase_a_stage_counts" if phase_a_only else "stage_counts",
            step21b,
        )
        for stage in STAGE_NAMES:
            absolute = int(counts.get(stage, 0) or 0)
            v2_count = int(v2_counts.get(stage, 0) or 0)
            step21b_count = int(step21b_counts.get(stage, 0) or 0)
            rows.append(
                {
                    "variant": variant_name,
                    "stage": stage,
                    "absolute_count": absolute,
                    "baseline_v2_count": v2_count,
                    "baseline_21b_count": step21b_count,
                    "retention_vs_v2_current": compute_retention(absolute, v2_count),
                    "retention_vs_21b_bytetrack": compute_retention(absolute, step21b_count),
                    "delta_vs_v2_current": absolute - v2_count,
                    "delta_vs_21b_bytetrack": absolute - step21b_count,
                }
            )
    return rows


def build_dimension_retention_rows(metrics: Dict[str, Any], dimension: str) -> List[Dict[str, Any]]:
    """Build per-scene, per-class, per-camera or person-group retention rows."""
    baselines = metrics.get("baselines", {})
    v2 = baselines.get("baseline_v2_current", {}).get("dimensions", {})
    step21b = baselines.get("baseline_21b_bytetrack", {}).get("dimensions", {})
    rows = []
    dimension_key = "person_vs_nonperson" if dimension == "person_vs_nonperson" else "per_%s" % dimension
    for variant_name, variant in sorted(metrics.get("variants", {}).items()):
        dimensions = variant.get("dimensions", {})
        phase_a_only = variant.get("status") == "phase_a_only"
        v2_dimensions = baselines.get("baseline_v2_current", {}).get(
            "phase_a_dimensions" if phase_a_only else "dimensions",
            v2,
        )
        step21b_dimensions = baselines.get("baseline_21b_bytetrack", {}).get(
            "phase_a_dimensions" if phase_a_only else "dimensions",
            step21b,
        )
        stages = sorted(set(dimensions.keys()) | set(v2_dimensions.keys()) | set(step21b_dimensions.keys()))
        for stage in stages:
            candidate_values = _dimension_values(dimensions, stage, dimension_key)
            v2_values = _dimension_values(v2_dimensions, stage, dimension_key)
            step21b_values = _dimension_values(step21b_dimensions, stage, dimension_key)
            keys = sorted(set(candidate_values.keys()) | set(v2_values.keys()) | set(step21b_values.keys()))
            for key in keys:
                absolute = int(candidate_values.get(key, 0) or 0)
                v2_count = int(v2_values.get(key, 0) or 0)
                step21b_count = int(step21b_values.get(key, 0) or 0)
                rows.append(
                    {
                        "variant": variant_name,
                        "stage": stage,
                        "dimension": dimension,
                        "key": key,
                        "absolute_count": absolute,
                        "baseline_v2_count": v2_count,
                        "baseline_21b_count": step21b_count,
                        "retention_vs_v2_current": compute_retention(absolute, v2_count),
                        "retention_vs_21b_bytetrack": compute_retention(absolute, step21b_count),
                        "delta_vs_v2_current": absolute - v2_count,
                        "delta_vs_21b_bytetrack": absolute - step21b_count,
                    }
                )
    return rows


def _dimension_values(values: Dict[str, Any], stage: str, dimension_key: str) -> Dict[str, Any]:
    stage_values = values.get(stage, {})
    if not isinstance(stage_values, dict):
        return {}
    result = stage_values.get(dimension_key, {})
    return result if isinstance(result, dict) else {}
