"""Stage-retention audit with explicit unit compatibility."""

import json
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.bytetrack_audit.artifact_inventory import build_artifact_inventories
from deep_oc_sort_3d.bytetrack_audit.audit_config import output_root
from deep_oc_sort_3d.bytetrack_audit.audit_io import write_csv, write_json
from deep_oc_sort_3d.bytetrack_audit.unit_keys import TRANSITIONS, transition_units


def run_stage_retention_audit(
    config: Dict[str, Any],
    progress: bool = True,
    inventories: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute stage transitions for all compared variants."""
    root = output_root(config)
    values = inventories or build_artifact_inventories(config, progress=progress)
    rows = []
    dimension_rows = {"scene": [], "class": [], "camera": [], "person_group": []}
    warnings = []
    for variant_name, inventory in values.items():
        counts = inventory.get("stage_counts", {})
        dimensions = _dimensions_by_stage(inventory.get("dimension_rows", []))
        for stage_from, stage_to in TRANSITIONS:
            count_from = int(counts.get(stage_from, 0) or 0)
            count_to = int(counts.get(stage_to, 0) or 0)
            unit_from, unit_to, comparison_type = transition_units(stage_from, stage_to)
            retention = None if count_from <= 0 else float(count_to) / float(count_from)
            row = {
                "variant_name": variant_name,
                "stage_from": stage_from,
                "stage_to": stage_to,
                "unit_from": unit_from,
                "unit_to": unit_to,
                "count_from": count_from,
                "count_to": count_to,
                "retention": retention,
                "drop_count": count_from - count_to,
                "drop_ratio": None if count_from <= 0 else float(count_from - count_to) / float(count_from),
                "unit_comparison_type": comparison_type,
                "top_affected_scenes": json.dumps(_top_drops(dimensions, stage_from, stage_to, "scene")),
                "top_affected_classes": json.dumps(_top_drops(dimensions, stage_from, stage_to, "class")),
                "top_affected_cameras": json.dumps(_top_drops(dimensions, stage_from, stage_to, "camera")),
            }
            rows.append(row)
            if comparison_type == "diagnostic_only":
                warnings.append("%s %s -> %s compares %s with %s" % (
                    variant_name, stage_from, stage_to, unit_from, unit_to
                ))
            for dimension in dimension_rows.keys():
                dimension_rows[dimension].extend(
                    _dimension_retention(variant_name, stage_from, stage_to, dimensions, dimension)
                )
    output_dir = root / "stage_retention"
    write_csv(output_dir / "stage_retention_summary.csv", rows)
    write_json(output_dir / "stage_retention_summary.json", {"rows": rows})
    write_csv(output_dir / "per_scene_stage_retention.csv", dimension_rows["scene"])
    write_csv(output_dir / "per_class_stage_retention.csv", dimension_rows["class"])
    write_csv(output_dir / "per_camera_stage_retention.csv", dimension_rows["camera"])
    write_csv(output_dir / "person_vs_nonperson_stage_retention.csv", dimension_rows["person_group"])
    write_csv(
        output_dir / "consistent_unit_retention.csv",
        [row for row in rows if row.get("unit_comparison_type") == "consistent"],
    )
    write_json(output_dir / "unit_mismatch_warnings.json", {"warnings": sorted(set(warnings))})
    return {"rows": rows, "dimension_rows": dimension_rows, "warnings": warnings, "inventories": values}


def _dimensions_by_stage(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    output = {}
    for row in rows:
        stage = str(row.get("stage_name", ""))
        dimension = str(row.get("dimension", ""))
        key = str(row.get("key", ""))
        output.setdefault(stage, {}).setdefault(dimension, {})[key] = int(row.get("count", 0) or 0)
    return output


def _top_drops(
    dimensions: Dict[str, Dict[str, Dict[str, int]]],
    stage_from: str,
    stage_to: str,
    dimension: str,
) -> List[Dict[str, Any]]:
    left = dimensions.get(stage_from, {}).get(dimension, {})
    right = dimensions.get(stage_to, {}).get(dimension, {})
    rows = []
    for key in set(left.keys()) | set(right.keys()):
        before = int(left.get(key, 0) or 0)
        after = int(right.get(key, 0) or 0)
        rows.append({"key": key, "before": before, "after": after, "drop": before - after})
    rows.sort(key=lambda item: int(item.get("drop", 0)), reverse=True)
    return rows[:5]


def _dimension_retention(
    variant: str,
    stage_from: str,
    stage_to: str,
    dimensions: Dict[str, Dict[str, Dict[str, int]]],
    dimension: str,
) -> List[Dict[str, Any]]:
    left = dimensions.get(stage_from, {}).get(dimension, {})
    right = dimensions.get(stage_to, {}).get(dimension, {})
    unit_from, unit_to, comparison_type = transition_units(stage_from, stage_to)
    output = []
    for key in sorted(set(left.keys()) | set(right.keys())):
        before = int(left.get(key, 0) or 0)
        after = int(right.get(key, 0) or 0)
        output.append(
            {
                "variant_name": variant,
                "stage_from": stage_from,
                "stage_to": stage_to,
                "unit_from": unit_from,
                "unit_to": unit_to,
                "unit_comparison_type": comparison_type,
                "dimension": dimension,
                "key": key,
                "count_from": before,
                "count_to": after,
                "retention": None if before <= 0 else float(after) / float(before),
                "drop_count": before - after,
            }
        )
    return output
