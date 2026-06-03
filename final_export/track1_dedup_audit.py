"""Deduplication audit between generic MVP export and official Track 1 output."""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

from deep_oc_sort_3d.final_export.track1_final_checks import read_track1_txt


def audit_generic_to_track1_dedup(
    generic_export_root: Union[str, Path],
    track1_path: Union[str, Path],
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Audit camera-level duplicate removal from generic export to Track 1."""
    generic_files = _find_generic_csv_files(Path(generic_export_root))
    generic_keys = []
    warnings = []
    per_scene_duplicates = {}
    per_class_duplicates = {}
    duplicate_key_counts = {}
    generic_rows_total = 0
    confidence_values = []
    for path in _progress_iter(generic_files, show_progress, "scan generic exports", "file"):
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            for column in ["scene_name", "class_id", "global_track_id", "frame_id"]:
                if column not in reader.fieldnames:
                    warnings.append("missing_generic_column:%s:%s" % (path, column))
            if "confidence" not in reader.fieldnames:
                warnings.append("missing_generic_column:%s:confidence" % path)
            for row in reader:
                generic_rows_total += 1
                confidence = _optional_float(row.get("confidence"))
                if confidence is not None:
                    confidence_values.append(confidence)
                key = _generic_row_key(row)
                generic_keys.append(key)
                duplicate_key_counts[key] = duplicate_key_counts.get(key, 0) + 1
    duplicate_keys = {key: count for key, count in duplicate_key_counts.items() if count > 1}
    for key, count in duplicate_keys.items():
        scene_id, class_id, _object_id, _frame_id = key
        per_scene_duplicates[str(scene_id)] = per_scene_duplicates.get(str(scene_id), 0) + count - 1
        per_class_duplicates[str(class_id)] = per_class_duplicates.get(str(class_id), 0) + count - 1
    track1_rows = read_track1_txt(track1_path)
    official_rows_total = len(track1_rows)
    unique_key_count = len(set(generic_keys))
    duplicate_rows_removed_estimated = max(0, generic_rows_total - unique_key_count)
    return {
        "generic_rows_total": generic_rows_total,
        "official_rows_total": official_rows_total,
        "unique_generic_keys": unique_key_count,
        "duplicate_rows_removed_estimated": duplicate_rows_removed_estimated,
        "dedup_ratio": _ratio(duplicate_rows_removed_estimated, generic_rows_total),
        "duplicate_keys_count": len(duplicate_keys),
        "top_duplicate_keys": _top_duplicate_keys(duplicate_keys),
        "per_scene_duplicates": per_scene_duplicates,
        "per_class_duplicates": per_class_duplicates,
        "confidence_range_from_generic": _range(confidence_values),
        "official_rows_match_unique_generic_keys": official_rows_total == unique_key_count,
        "dedup_rule": "highest_confidence_observation_per_scene_class_object_frame",
        "warnings": sorted(set(warnings)),
    }


def write_dedup_audit_report(report: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Write dedup audit report JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _find_generic_csv_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    direct = sorted(root.glob("*.csv"))
    if direct:
        return direct
    return sorted(root.rglob("*.csv"))


def _generic_row_key(row: Dict[str, Any]) -> Tuple[int, int, int, int]:
    scene_id = _scene_name_to_id(row.get("scene_name", ""))
    return (
        scene_id,
        _safe_int(row.get("class_id")),
        _safe_int(row.get("global_track_id")),
        _safe_int(row.get("frame_id")),
    )


def _scene_name_to_id(value: Any) -> int:
    regex_result = re.search(r"(\d+)$", str(value))
    if regex_result is None:
        return -1
    return int(regex_result.group(1))


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _optional_float(value: Any) -> Any:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _range(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"min": None, "max": None}
    return {"min": min(values), "max": max(values)}


def _top_duplicate_keys(duplicates: Dict[Tuple[int, int, int, int], int], top_k: int = 25) -> List[Dict[str, Any]]:
    items = sorted(duplicates.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {
            "scene_id": key[0],
            "class_id": key[1],
            "object_id": key[2],
            "frame_id": key[3],
            "count": count,
            "duplicates_removed": count - 1,
        }
        for key, count in items
    ]


def _progress_iter(values: List[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: %d/%d %s" % (desc, index + 1, total, value))
        yield value
