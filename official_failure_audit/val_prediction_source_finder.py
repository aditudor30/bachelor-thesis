"""Find or reconstruct validation Track1-like predictions without GT leakage."""

import csv
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.official_failure_audit.box3d_utils import normalize_yaw
from deep_oc_sort_3d.official_failure_audit.failure_audit_config import internal_to_official, official_class_names, scene_id, val_scenes
from deep_oc_sort_3d.official_failure_audit.failure_io import iter_jsonl, progress_iter, read_json, write_json
from deep_oc_sort_3d.official_failure_audit.track1_parser import AuditTrack1Row, read_track1_like, write_track1_like


SUPPORTED_SUFFIXES = {".txt", ".csv", ".jsonl"}


def build_val_prediction_files(
    config: Dict[str, Any], output_root: Path, progress: bool = True,
) -> Tuple[List[AuditTrack1Row], Dict[str, Any]]:
    directory = output_root / "val_predictions"
    v2_rows, v2_summary = _build_variant("v2", config, directory, progress)
    v5_rows, v5_summary = _build_variant("v5", config, directory, progress)
    if not v5_rows and v2_rows:
        corrections_path = _find_v5_corrections(config)
        corrections = read_json(corrections_path) if corrections_path is not None else {}
        if corrections:
            v5_rows, applied = _apply_v5_corrections(v2_rows, corrections)
            v5_summary = {
                "status": "reconstructed_from_v2_val_plus_existing_v5_corrections",
                "prediction_origin": "v2_val_geometry_with_v5_corrections",
                "base_rows": len(v2_rows), "rows": len(v5_rows),
                "corrections_path": str(corrections_path), "applied": applied,
                "gt_used_for_prediction": False,
                "coordinate_frame_distribution": _count_values([row.coordinate_frame for row in v5_rows]),
            }
    decimals = int(config.get("official_track1", {}).get("round_float_decimals", 2))
    write_track1_like(directory / "v2_val_track1_like.txt", v2_rows, decimals=decimals)
    write_track1_like(directory / "v5_val_track1_like.txt", v5_rows, decimals=decimals)
    selected_name = "v5" if v5_rows else ("v2" if v2_rows else "missing")
    selected_rows = v5_rows if v5_rows else v2_rows
    summary = {
        "status": "ok" if selected_rows else "val_prediction_source_missing_fix_required",
        "selected_variant": selected_name, "selected_rows": len(selected_rows),
        "gt_used_for_prediction": False, "v2": v2_summary, "v5": v5_summary,
        "v2_output": str(directory / "v2_val_track1_like.txt"),
        "v5_output": str(directory / "v5_val_track1_like.txt"),
    }
    write_json(directory / "selected_val_prediction_source.json", summary)
    return selected_rows, summary


def _build_variant(
    variant: str, config: Dict[str, Any], output_directory: Path, progress: bool,
) -> Tuple[List[AuditTrack1Row], Dict[str, Any]]:
    roots = [Path(str(value)) for value in config.get("paths", {}).get("%s_roots" % variant, [])]
    candidates = _discover_candidates(roots, val_scenes(config), progress)
    selected = _select_candidates(candidates)
    rows: List[AuditTrack1Row] = []
    rejected_gt_derived = 0
    errors: List[str] = []
    for candidate in selected:
        try:
            candidate_rows, rejected = _read_candidate(candidate, config)
            rows.extend(candidate_rows)
            rejected_gt_derived += rejected
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append("%s: %s" % (candidate["path"], exc))
    rows = _deduplicate_rows(rows)
    summary = {
        "status": "found_existing_val_predictions" if rows else "not_found",
        "variant": variant, "roots": [str(path) for path in roots],
        "candidate_files": len(candidates), "selected_files": [str(item["path"]) for item in selected],
        "rows": len(rows), "tracks": len(set((row.scene_id, row.class_id, row.object_id) for row in rows)),
        "rejected_gt_derived_rows": rejected_gt_derived, "errors": errors,
        "prediction_origin": "existing_artifacts", "gt_used_for_prediction": False,
        "coordinate_frame_distribution": _count_values([row.coordinate_frame for row in rows]),
    }
    write_json(output_directory / (variant + "_source_discovery.json"), summary)
    return rows, summary


def _discover_candidates(
    roots: Sequence[Path], scenes: Sequence[str], progress: bool,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for root_index, root in enumerate(roots):
        if not root.is_dir():
            continue
        paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
        for path in progress_iter(paths, progress, "23A scan %s" % root.name):
            lowered = str(path).lower()
            if not any(scene.lower() in lowered for scene in scenes) and "track1" not in path.name.lower():
                continue
            inspected = _inspect_candidate(path, scenes)
            if inspected is None:
                continue
            inspected["root_index"] = root_index
            output.append(inspected)
    return sorted(output, key=lambda item: (-int(item["score"]), int(item["root_index"]), str(item["path"])))


def _inspect_candidate(path: Path, scenes: Sequence[str]) -> Optional[Dict[str, Any]]:
    scene_set = set(scenes)
    if path.suffix.lower() == ".txt":
        try:
            rows = read_track1_like(path)
        except (OSError, ValueError):
            return None
        found = sorted(set("Warehouse_%03d" % row.scene_id for row in rows) & scene_set)
        if not found:
            return None
        return {"path": path, "kind": "track1", "score": 1000, "scene_keys": found, "camera_keys": []}
    first = _first_structured_row(path)
    if first is None or not _has_geometry(first):
        return None
    scene = _scene_name(first, path)
    if scene not in scene_set:
        return None
    camera = str(first.get("camera_id", ""))
    has_global = first.get("global_track_id") not in (None, "")
    has_local = any(first.get(key) not in (None, "") for key in ["local_track_id", "track_id", "object_id"])
    score = 700 if has_global else (500 if has_local else 300)
    lowered = str(path).lower()
    if "frame_global" in lowered or "final" in lowered or "export" in lowered:
        score += 100
    if "stabilized" in lowered:
        score += 30
    if "raw" in lowered:
        score -= 30
    return {
        "path": path, "kind": "structured", "score": score,
        "scene_keys": [scene], "camera_keys": ["%s/%s" % (scene, camera)] if camera else [],
    }


def _select_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    track1 = [item for item in candidates if item["kind"] == "track1"]
    if track1:
        return [track1[0]]
    selected: List[Dict[str, Any]] = []
    covered = set()
    for item in candidates:
        keys = item.get("camera_keys") or item.get("scene_keys") or []
        if any(key not in covered for key in keys):
            selected.append(item)
            covered.update(keys)
    return selected


def _read_candidate(candidate: Dict[str, Any], config: Dict[str, Any]) -> Tuple[List[AuditTrack1Row], int]:
    path = Path(candidate["path"])
    wanted_scene_ids = set(scene_id(scene) for scene in val_scenes(config))
    if candidate["kind"] == "track1":
        return [row for row in read_track1_like(path) if row.scene_id in wanted_scene_ids], 0
    rows: List[AuditTrack1Row] = []
    rejected = 0
    for line_number, raw in enumerate(_iter_structured_rows(path), 1):
        if _gt_derived(raw):
            rejected += 1
            continue
        converted = _structured_to_track1(raw, path, line_number, config)
        if converted is not None and converted.scene_id in wanted_scene_ids:
            rows.append(converted)
    return rows, rejected


def _structured_to_track1(
    raw: Dict[str, Any], path: Path, line_number: int, config: Dict[str, Any],
) -> Optional[AuditTrack1Row]:
    scene_name = _scene_name(raw, path)
    sid = scene_id(scene_name)
    frame_id = _optional_int(_first(raw, ["frame_id", "frame", "frame_index"]))
    raw_class = _optional_int(_first(raw, ["class_id", "category_id", "label_id"]))
    if sid < 0 or frame_id is None or raw_class is None:
        return None
    center = _vector(raw.get("center_3d")) or _values(raw, ["center_x", "center_y", "center_z"])
    if center is None:
        center = _values(raw, ["x", "y", "z"])
    dims = _vector(raw.get("dimensions_3d")) or _values(raw, ["width_3d", "length_3d", "height_3d"])
    if dims is None:
        dims = _values(raw, ["width", "length", "height"])
    yaw = _optional_float(_first(raw, ["yaw", "rotation_y", "heading"]))
    if center is None or dims is None or yaw is None:
        return None
    official_class = _official_class(raw_class, raw.get("class_name"), config)
    object_id, source_kind = _object_id(raw, scene_name, str(raw.get("camera_id", "")), raw_class, line_number)
    confidence = _optional_float(_first(raw, ["confidence", "confidence_3d", "confidence_2d", "score"]))
    return AuditTrack1Row(
        scene_id=sid, class_id=official_class, object_id=object_id, frame_id=frame_id,
        x=center[0], y=center[1], z=center[2], width=dims[0], length=dims[1], height=dims[2], yaw=yaw,
        raw_class_id=raw_class, source_class_space="internal", source_path=str(path),
        source_kind=source_kind, confidence=confidence,
        coordinate_frame=str(raw.get("coordinate_frame") or "unknown").lower(),
    )


def _deduplicate_rows(rows: Sequence[AuditTrack1Row]) -> List[AuditTrack1Row]:
    selected: Dict[Tuple[int, int, int, int], AuditTrack1Row] = {}
    for row in rows:
        current = selected.get(row.key())
        if current is None or _confidence(row) > _confidence(current):
            selected[row.key()] = row
    return sorted(selected.values(), key=lambda item: item.key())


def _apply_v5_corrections(
    rows: Sequence[AuditTrack1Row], corrections: Dict[str, Any],
) -> Tuple[List[AuditTrack1Row], Dict[str, int]]:
    output: List[AuditTrack1Row] = []
    counts = {"dimension": 0, "center": 0, "yaw": 0}
    for row in rows:
        key = str(row.class_id)
        changes: Dict[str, Any] = {}
        dimension = corrections.get("dimension", {}).get(key, {})
        center = corrections.get("center", {}).get(key, {})
        yaw = corrections.get("yaw", {}).get(key, {})
        if dimension.get("selected"):
            scale = np.asarray(dimension.get("scale", [1.0, 1.0, 1.0]), dtype=float)
            dims = np.asarray([row.width, row.length, row.height], dtype=float) * scale
            if np.all(np.isfinite(dims)) and np.all(dims > 0.0):
                changes.update(width=float(dims[0]), length=float(dims[1]), height=float(dims[2]))
                counts["dimension"] += 1
        if center.get("selected"):
            bias = np.asarray(center.get("bias", [0.0, 0.0, 0.0]), dtype=float)
            xyz = np.asarray([row.x, row.y, row.z], dtype=float) + bias
            if np.all(np.isfinite(xyz)):
                changes.update(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
                counts["center"] += 1
        if yaw.get("selected"):
            changes["yaw"] = normalize_yaw(row.yaw + float(yaw.get("bias_rad", 0.0)))
            counts["yaw"] += 1
        output.append(row.clone(**changes) if changes else row)
    return output, counts


def _find_v5_corrections(config: Dict[str, Any]) -> Optional[Path]:
    roots = [Path(str(value)) for value in config.get("paths", {}).get("v5_roots", [])]
    candidates = []
    for root in roots:
        direct = root / "learned_corrections" / "selected_corrections.json"
        if direct.is_file():
            candidates.append(direct)
        if root.is_dir():
            candidates.extend(root.rglob("selected_corrections.json"))
    return sorted(set(candidates), key=str)[0] if candidates else None


def _iter_structured_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for row in iter_jsonl(path):
            yield row
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield dict(row)


def _first_structured_row(path: Path) -> Optional[Dict[str, Any]]:
    try:
        for row in _iter_structured_rows(path):
            return row
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def _has_geometry(row: Dict[str, Any]) -> bool:
    center_fields = row.get("center_3d") is not None or all(key in row for key in ["center_x", "center_y", "center_z"]) or all(key in row for key in ["x", "y", "z"])
    dim_fields = row.get("dimensions_3d") is not None or all(key in row for key in ["width_3d", "length_3d", "height_3d"]) or all(key in row for key in ["width", "length", "height"])
    return bool(center_fields and dim_fields and any(key in row for key in ["yaw", "rotation_y", "heading"]))


def _scene_name(row: Dict[str, Any], path: Path) -> str:
    value = _first(row, ["scene_name", "scene", "sequence"])
    if value not in (None, ""):
        text = str(value)
        if text.startswith("Warehouse_"):
            return text
        try:
            return "Warehouse_%03d" % int(float(text))
        except ValueError:
            pass
    for part in path.parts:
        if str(part).startswith("Warehouse_"):
            return str(part)
    sid = _optional_int(row.get("scene_id"))
    return "Warehouse_%03d" % sid if sid is not None else ""


def _official_class(raw_class: int, class_name: Any, config: Dict[str, Any]) -> int:
    names = official_class_names(config)
    if class_name not in (None, "") and str(class_name) in names:
        return names[str(class_name)]
    return internal_to_official(config).get(raw_class, raw_class)


def _object_id(
    row: Dict[str, Any], scene: str, camera: str, class_id: int, line_number: int,
) -> Tuple[int, str]:
    for key, kind in [
        ("global_track_id", "global_frame_record"), ("object_id", "object_record"),
        ("local_track_id", "local_track_record"), ("track_id", "track_record"),
        ("detection_id", "detection_record"),
    ]:
        value = _optional_int(row.get(key))
        if value is not None:
            if key == "global_track_id":
                return value, kind
            return _stable_id("%s|%s|%s|%s" % (scene, camera, class_id, value)), kind
    return _stable_id("%s|%s|%s|line|%s" % (scene, camera, class_id, line_number)), "row_fallback"


def _stable_id(value: str) -> int:
    return int(zlib.crc32(value.encode("utf-8")) & 0x7FFFFFFF)


def _gt_derived(row: Dict[str, Any]) -> bool:
    explicit = _bool(row.get("is_gt_derived"))
    matched = _bool(row.get("matched_gt"))
    return explicit or (matched and str(row.get("is_gt_derived", "")).lower() not in ("false", "0", "no"))


def _vector(value: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        result = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _values(row: Dict[str, Any], fields: Sequence[str]) -> Optional[Tuple[float, float, float]]:
    try:
        result = (float(row[fields[0]]), float(row[fields[1]]), float(row[fields[2]]))
    except (KeyError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _first(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if row.get(key) not in (None, ""):
            return row.get(key)
    return None


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool:
    return value is True or str(value).lower() in ("true", "1", "yes")


def _confidence(row: AuditTrack1Row) -> float:
    return float(row.confidence) if row.confidence is not None else -1.0


def _count_values(values: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items()))
