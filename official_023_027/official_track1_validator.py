"""Strict validation for official scenes Warehouse_023 through Warehouse_027."""

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from deep_oc_sort_3d.official_023_027.official_config import scene_ids
from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row, progress_iter, write_json


def validate_official_track1(
    path: Path,
    config: Dict[str, Any],
    progress: bool = True,
) -> Dict[str, Any]:
    """Validate schema, scenes, mapping, rounding, dimensions, duplicates and sorting."""
    expected_scenes = set(scene_ids(config))
    valid_classes = set(int(value) for value in config.get("official_track1", {}).get("valid_class_ids", range(7)))
    decimals = int(config.get("official_track1", {}).get("round_float_decimals", 2))
    if not path.exists() or path.stat().st_size <= 0:
        return _missing_report(path, expected_scenes)
    errors = []
    rows = []
    malformed = 0
    rounding_issues = 0
    with path.open("r", encoding="utf-8") as handle:
        raw_lines = [line.rstrip("\n") for line in handle if line.strip()]
    for line_number, line in enumerate(progress_iter(raw_lines, progress, "validate %s" % path.name), start=1):
        parts = line.strip().split()
        if len(parts) != 11:
            malformed += 1
            errors.append("line_%d_invalid_column_count" % line_number)
            continue
        try:
            row = OfficialTrack1Row(
                scene_id=_parse_integer(parts[0]), class_id=_parse_integer(parts[1]), object_id=_parse_integer(parts[2]), frame_id=_parse_integer(parts[3]),
                x=float(parts[4]), y=float(parts[5]), z=float(parts[6]), width=float(parts[7]), length=float(parts[8]), height=float(parts[9]), yaw=float(parts[10]),
                source_line=line_number,
            )
        except (TypeError, ValueError):
            errors.append("line_%d_non_numeric" % line_number)
            continue
        for value in [row.x, row.y, row.z, row.width, row.length, row.height, row.yaw]:
            if not math.isfinite(value):
                errors.append("line_%d_nan_or_inf" % line_number)
        if not all(_has_fixed_decimals(value, decimals) for value in parts[4:11]):
            rounding_issues += 1
            errors.append("line_%d_float_rounding_not_%d_decimals" % (line_number, decimals))
        rows.append(row)
    per_scene = defaultdict(int)
    per_class = defaultdict(int)
    duplicate_count = 0
    invalid_scene = 0
    invalid_class = 0
    invalid_object = 0
    negative_frame = 0
    non_positive_dimensions = 0
    seen = set()
    for row in rows:
        per_scene[str(row.scene_id)] += 1
        per_class[str(row.class_id)] += 1
        if row.scene_id not in expected_scenes:
            invalid_scene += 1
        if row.class_id not in valid_classes:
            invalid_class += 1
        if row.object_id < 0:
            invalid_object += 1
        if row.frame_id < 0:
            negative_frame += 1
        if row.width <= 0.0 or row.length <= 0.0 or row.height <= 0.0:
            non_positive_dimensions += 1
        if row.key() in seen:
            duplicate_count += 1
        seen.add(row.key())
    missing_scene_ids = sorted(expected_scenes - set(int(key) for key in per_scene.keys()))
    sorting_issues = 0 if [row.key() for row in rows] == sorted(row.key() for row in rows) else 1
    checks = {
        "empty_file": 0 if rows else 1,
        "num_columns_invalid": malformed,
        "invalid_scene_id": invalid_scene,
        "missing_scene_ids": len(missing_scene_ids),
        "invalid_class_id": invalid_class,
        "invalid_object_id": invalid_object,
        "negative_frame_id": negative_frame,
        "non_positive_dimensions": non_positive_dimensions,
        "duplicate_key_count": duplicate_count,
        "sorting_issues": sorting_issues,
        "rounding_issues": rounding_issues,
        "nan_or_inf_values": sum(1 for error in errors if "nan_or_inf" in error),
    }
    for key, value in checks.items():
        if isinstance(value, int) and value > 0:
            errors.append("%s:%s" % (key, value))
    report = {
        "status": "ok" if not errors else "error",
        "num_errors": len(errors),
        "errors": errors,
        "total_rows": len(rows),
        "checks": checks,
        "expected_scene_ids": sorted(expected_scenes),
        "scene_ids": sorted(int(key) for key in per_scene.keys()),
        "missing_scene_ids": missing_scene_ids,
        "per_scene_rows": dict(sorted(per_scene.items(), key=lambda item: int(item[0]))),
        "per_class_rows": dict(sorted(per_class.items(), key=lambda item: int(item[0]))),
        "class_mapping": "official",
        "float_rounding_decimals": decimals,
        "track1_path": str(path),
    }
    return report


def validate_and_write(path: Path, output_path: Path, config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Validate one official file and write its report."""
    report = validate_official_track1(path, config, progress=progress)
    write_json(output_path, report)
    return report


def _has_fixed_decimals(value: str, decimals: int) -> bool:
    lowered = str(value).lower()
    if "e" in lowered or "." not in lowered:
        return False
    return len(lowered.rsplit(".", 1)[1]) == decimals


def _parse_integer(value: str) -> int:
    """Parse one integer column without silently truncating decimal values."""
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("Expected integer token: %s" % value)
    return int(numeric)


def _missing_report(path: Path, expected_scenes: Sequence[int]) -> Dict[str, Any]:
    return {
        "status": "error",
        "num_errors": 1,
        "errors": ["missing_or_empty_file"],
        "total_rows": 0,
        "checks": {"empty_file": 1},
        "expected_scene_ids": sorted(expected_scenes),
        "scene_ids": [],
        "missing_scene_ids": sorted(expected_scenes),
        "track1_path": str(path),
    }
