"""Diagnostics for the SmartSpaces Person ReID dataset."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from deep_oc_sort_3d.reid_training.reid_dataset_config import output_root_from_config
from deep_oc_sort_3d.reid_training.reid_dataset_io import (
    count_by,
    group_by,
    numeric_summary,
    read_csv_rows,
    read_json,
    write_csv_rows,
    write_json,
)


def build_reid_dataset_diagnostics_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build diagnostics and write summary files."""
    output_root = output_root_from_config(config)
    rows, _fields = read_csv_rows(output_root / "metadata" / "all_crops.csv")
    pairs_summary = read_json(output_root / "pairs_triplets" / "pairs_triplets_summary.json") or {}
    summary, tables, warnings = build_dataset_diagnostics(rows, pairs_summary, config)
    diagnostics_root = output_root / "diagnostics"
    write_json(summary, diagnostics_root / "dataset_summary.json")
    write_json(warnings, diagnostics_root / "warnings.json")
    write_json(tables.get("crop_quality_summary", {}), diagnostics_root / "crop_quality_summary.json")
    write_csv_rows(tables.get("identity_distribution", []), diagnostics_root / "identity_distribution.csv")
    write_csv_rows(tables.get("camera_distribution", []), diagnostics_root / "camera_distribution.csv")
    write_csv_rows(tables.get("scene_distribution", []), diagnostics_root / "scene_distribution.csv")
    write_csv_rows(tables.get("per_identity_stats", []), diagnostics_root / "per_identity_stats.csv")
    write_csv_rows(tables.get("per_camera_stats", []), diagnostics_root / "per_camera_stats.csv")
    write_csv_rows(tables.get("per_scene_stats", []), diagnostics_root / "per_scene_stats.csv")
    return summary


def build_dataset_diagnostics(
    rows: List[Dict[str, Any]],
    pairs_summary: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return dataset summary, table rows, and warnings."""
    valid_rows = [row for row in rows if str(row.get("is_valid_crop", "")) == "1"]
    invalid_rows = [row for row in rows if str(row.get("is_valid_crop", "")) != "1"]
    train_rows = [row for row in valid_rows if row.get("split") == "train"]
    val_rows = [row for row in valid_rows if row.get("split") == "val"]
    identity_groups = group_by(valid_rows, "identity_id")
    train_identities = sorted(set([str(row.get("identity_id", "")) for row in train_rows]))
    val_identities = sorted(set([str(row.get("identity_id", "")) for row in val_rows]))
    per_identity_stats = _per_group_stats(valid_rows, "identity_id")
    per_camera_stats = _per_group_stats(valid_rows, "camera_id")
    per_scene_stats = _per_group_stats(valid_rows, "scene_name")
    warnings = build_warnings(valid_rows, invalid_rows, train_identities, val_identities, config)
    summary = {
        "total_crops": len(valid_rows),
        "metadata_rows": len(rows),
        "train_crops": len(train_rows),
        "val_crops": len(val_rows),
        "num_train_identities": len(train_identities),
        "num_val_identities": len(val_identities),
        "identity_overlap_train_val": len(set(train_identities).intersection(set(val_identities))),
        "crops_per_identity": numeric_summary([len(values) for values in identity_groups.values()]),
        "identities_under_2_crops": len([values for values in identity_groups.values() if len(values) < 2]),
        "identities_under_5_crops": len([values for values in identity_groups.values() if len(values) < 5]),
        "scene_distribution": count_by(valid_rows, "scene_name"),
        "camera_distribution": count_by(valid_rows, "camera_id"),
        "bbox_area_distribution": numeric_summary([row.get("bbox_area") for row in valid_rows]),
        "crop_width_distribution": numeric_summary([row.get("crop_width") for row in valid_rows]),
        "crop_height_distribution": numeric_summary([row.get("crop_height") for row in valid_rows]),
        "invalid_crops": len(invalid_rows),
        "invalid_reasons": count_by(invalid_rows, "invalid_reason"),
        "positive_pairs_train": pairs_summary.get("positive_pairs_train", 0),
        "negative_pairs_train": pairs_summary.get("negative_pairs_train", 0),
        "triplets_train": pairs_summary.get("triplets_train", 0),
        "positive_pairs_val": pairs_summary.get("positive_pairs_val", 0),
        "negative_pairs_val": pairs_summary.get("negative_pairs_val", 0),
        "triplets_val": pairs_summary.get("triplets_val", 0),
        "verdict": verdict_from_summary(len(valid_rows), len(train_identities), pairs_summary, warnings),
    }
    tables = {
        "crop_quality_summary": {
            "valid_crops": len(valid_rows),
            "invalid_crops": len(invalid_rows),
            "invalid_reasons": count_by(invalid_rows, "invalid_reason"),
        },
        "identity_distribution": [{"identity_id": row["group"], "num_crops": row["num_crops"]} for row in per_identity_stats],
        "camera_distribution": [{"camera_id": key, "num_crops": value} for key, value in sorted(count_by(valid_rows, "camera_id").items())],
        "scene_distribution": [{"scene_name": key, "num_crops": value} for key, value in sorted(count_by(valid_rows, "scene_name").items())],
        "per_identity_stats": per_identity_stats,
        "per_camera_stats": per_camera_stats,
        "per_scene_stats": per_scene_stats,
    }
    return summary, tables, warnings


def build_warnings(
    valid_rows: List[Dict[str, Any]],
    invalid_rows: List[Dict[str, Any]],
    train_identities: List[str],
    val_identities: List[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build warning dictionary."""
    diagnostics_cfg = config.get("diagnostics", {})
    min_crops = int(diagnostics_cfg.get("min_crops_per_identity_warning", 5))
    min_ids = int(diagnostics_cfg.get("min_identities_warning", 20))
    identity_groups = group_by(valid_rows, "identity_id")
    rare = sorted([identity for identity, values in identity_groups.items() if len(values) < min_crops])
    overlap = sorted(set(train_identities).intersection(set(val_identities)))
    warnings = {
        "rare_identities": rare,
        "rare_identity_count": len(rare),
        "train_val_identity_overlap": overlap,
        "train_identity_count_below_warning": len(train_identities) < min_ids,
        "val_identity_count_below_warning": len(val_identities) < min_ids,
        "invalid_crop_count": len(invalid_rows),
        "invalid_reasons": count_by(invalid_rows, "invalid_reason"),
    }
    return warnings


def verdict_from_summary(total_crops: int, num_train_identities: int, pairs_summary: Dict[str, Any], warnings: Dict[str, Any]) -> str:
    """Return readiness verdict."""
    if total_crops <= 0 or num_train_identities <= 0:
        return "reid_dataset_invalid_fix_required"
    if total_crops < 100 or num_train_identities < 5:
        return "reid_dataset_too_small"
    if int(pairs_summary.get("triplets_train", 0)) <= 0 or int(pairs_summary.get("positive_pairs_train", 0)) <= 0:
        return "reid_dataset_invalid_fix_required"
    if warnings.get("rare_identity_count", 0) > 0 or warnings.get("train_identity_count_below_warning"):
        return "reid_dataset_usable_with_warnings"
    return "reid_dataset_ready"


def _per_group_stats(rows: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    groups = group_by(rows, field)
    output: List[Dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        output.append(
            {
                "group": key,
                field: key,
                "num_crops": len(values),
                "num_scenes": len(set([str(row.get("scene_name", "")) for row in values])),
                "num_cameras": len(set([str(row.get("camera_id", "")) for row in values])),
                "bbox_area_mean": numeric_summary([row.get("bbox_area") for row in values]).get("mean"),
                "crop_width_median": numeric_summary([row.get("crop_width") for row in values]).get("median"),
                "crop_height_median": numeric_summary([row.get("crop_height") for row in values]).get("median"),
            }
        )
    return output
