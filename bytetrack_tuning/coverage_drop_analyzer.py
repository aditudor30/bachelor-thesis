"""Locate the largest coverage losses between pipeline stages."""

import json
from typing import Any, Dict, List


TRANSITIONS = [
    ("observations", "local_records"),
    ("local_records", "tracklets"),
    ("tracklets", "mtmc_candidates"),
    ("mtmc_candidates", "motion_clean_candidates"),
    ("motion_clean_candidates", "global_tracks"),
    ("global_tracks", "final_export_rows"),
    ("final_export_rows", "track1_rows"),
]


def compute_coverage_drop_rows(variants: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute counts and loss percentages for each configured transition."""
    rows = []
    for variant_name, variant in sorted(variants.items()):
        stage_counts = variant.get("stage_counts", {})
        dimensions = variant.get("dimensions", {})
        for input_stage, output_stage in TRANSITIONS:
            input_count = int(stage_counts.get(input_stage, 0) or 0)
            output_count = int(stage_counts.get(output_stage, 0) or 0)
            drop_count = input_count - output_count
            retention = None if input_count <= 0 else float(output_count) / float(input_count)
            rows.append(
                {
                    "variant": variant_name,
                    "transition": "%s -> %s" % (input_stage, output_stage),
                    "input_stage": input_stage,
                    "output_stage": output_stage,
                    "input_count": input_count,
                    "output_count": output_count,
                    "retention": retention,
                    "drop_count": drop_count,
                    "drop_percentage": None if input_count <= 0 else 100.0 * float(drop_count) / float(input_count),
                    "top_affected_classes": json.dumps(
                        _top_drops(dimensions, input_stage, output_stage, "per_class"),
                        sort_keys=True,
                    ),
                    "top_affected_scenes": json.dumps(
                        _top_drops(dimensions, input_stage, output_stage, "per_scene"),
                        sort_keys=True,
                    ),
                }
            )
    return rows


def _top_drops(
    dimensions: Dict[str, Any],
    input_stage: str,
    output_stage: str,
    dimension: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    left = dimensions.get(input_stage, {}).get(dimension, {})
    right = dimensions.get(output_stage, {}).get(dimension, {})
    if not isinstance(left, dict) or not isinstance(right, dict):
        return []
    output = []
    for key in set(left.keys()) | set(right.keys()):
        input_count = int(left.get(key, 0) or 0)
        output_count = int(right.get(key, 0) or 0)
        output.append({"key": key, "drop": input_count - output_count, "input": input_count, "output": output_count})
    output.sort(key=lambda row: int(row.get("drop", 0)), reverse=True)
    return output[:limit]

