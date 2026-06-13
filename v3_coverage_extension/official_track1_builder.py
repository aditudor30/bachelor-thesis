"""Build official Track1 variants by appending V3-owned recovery tracks."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row, read_track1_rows, write_track1_rows
from deep_oc_sort_3d.official_023_027.official_track1_remap import remap_rows_to_official, stable_deduplicate_rows
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import id_offset, official_to_internal, internal_to_official, variant_root
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import write_csv, write_json
from deep_oc_sort_3d.v3_coverage_extension.recovery_source_loader import RecoveryTrack, record_geometry_valid


def build_variant_track1(
    config: Dict[str, Any],
    variant: str,
    selected_tracks: Sequence[RecoveryTrack],
    selection_summary: Dict[str, Any],
    progress: bool = True,
) -> Dict[str, Any]:
    """Append selected V3 local tracks and emit internal plus official files."""
    del progress
    root = variant_root(config, variant)
    baseline_path = Path(str(config.get("paths", {}).get("v3_official_track1", "")))
    baseline_official = read_track1_rows(baseline_path, progress=False)
    inverse = official_to_internal(config)
    baseline_internal = [_copy_with_class(row, inverse.get(int(row.class_id), int(row.class_id))) for row in baseline_official]
    additions, track_manifest = tracks_to_internal_rows(config, variant, selected_tracks)
    internal_rows, internal_dedup = stable_deduplicate_rows(baseline_internal + additions)
    official_rows, remap_summary = remap_rows_to_official(internal_rows, internal_to_official(config))
    official_rows, official_dedup = stable_deduplicate_rows(official_rows)
    decimals = int(config.get("official_track1", {}).get("round_float_decimals", 2))
    write_track1_rows(root / "track1_internal.txt", internal_rows, decimals=decimals)
    write_track1_rows(root / "track1_official.txt", official_rows, decimals=decimals)
    added_summary = _added_rows_summary(baseline_official, official_rows, track_manifest)
    manifest = {
        "variant": variant,
        "source_policy": "V3-only local detection records; V2 is comparison-only",
        "baseline_track1": str(baseline_path),
        "baseline_rows": len(baseline_official),
        "selected_tracks": len(selected_tracks),
        "added_internal_rows_before_dedup": len(additions),
        "output_rows": len(official_rows),
        "internal_dedup": internal_dedup,
        "official_dedup": official_dedup,
        "class_remap": remap_summary,
        "selection": selection_summary,
        "track_manifest": track_manifest,
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "added_rows_summary.json", added_summary)
    write_csv(root / "per_scene_summary.csv", _distribution_rows(official_rows, "scene_id"), ["scene_id", "rows", "unique_tracks"])
    write_csv(root / "per_class_summary.csv", _distribution_rows(official_rows, "class_id"), ["class_id", "rows", "unique_tracks"])
    return {"variant": variant, "root": str(root), "track1_path": str(root / "track1_official.txt"), "manifest": manifest, "added_rows_summary": added_summary}


def tracks_to_internal_rows(config: Dict[str, Any], variant: str, tracks: Sequence[RecoveryTrack]) -> Tuple[List[OfficialTrack1Row], List[Dict[str, Any]]]:
    """Convert V3 local records to internal-class Track1 rows and stable IDs."""
    rows = []
    manifest = []
    offset = id_offset(config, variant)
    sorted_tracks = sorted(tracks, key=lambda item: item.key)
    for index, track in enumerate(sorted_tracks):
        object_id = int(offset + int(track.scene_id) * 100000 + index + 1)
        valid_records = [record for record in track.records if record_geometry_valid(record)]
        for record in valid_records:
            center = record.center_3d
            dims = record.dimensions_3d
            rows.append(OfficialTrack1Row(
                scene_id=int(track.scene_id), class_id=int(track.internal_class_id), object_id=object_id,
                frame_id=int(record.frame_id), x=float(center[0]), y=float(center[1]), z=float(center[2]),
                width=float(dims[0]), length=float(dims[1]), height=float(dims[2]), yaw=float(record.yaw),
                confidence=float(record.confidence),
            ))
        manifest.append({
            "scene_id": track.scene_id, "camera_id": track.camera_id, "local_track_id": track.local_track_id,
            "internal_class_id": track.internal_class_id, "official_class_id": track.official_class_id,
            "new_object_id": object_id, "rows": len(valid_records), "mean_confidence": track.mean_confidence,
            "p95_step_distance": track.p95_step_distance, "max_step_distance": track.max_step_distance,
            "jump_ratio": track.jump_ratio, "states": sorted(track.states),
        })
    return rows, manifest


def _copy_with_class(row: OfficialTrack1Row, class_id: int) -> OfficialTrack1Row:
    return OfficialTrack1Row(
        scene_id=row.scene_id, class_id=int(class_id), object_id=row.object_id, frame_id=row.frame_id,
        x=row.x, y=row.y, z=row.z, width=row.width, length=row.length, height=row.height, yaw=row.yaw,
        source_line=row.source_line, confidence=row.confidence,
    )


def _added_rows_summary(baseline: Sequence[OfficialTrack1Row], output: Sequence[OfficialTrack1Row], track_manifest: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_keys = set(row.key() for row in baseline)
    added = [row for row in output if row.key() not in baseline_keys]
    by_scene = defaultdict(int)
    by_class = defaultdict(int)
    by_scene_class = defaultdict(int)
    for row in added:
        by_scene[str(row.scene_id)] += 1
        by_class[str(row.class_id)] += 1
        by_scene_class["%d:%d" % (row.scene_id, row.class_id)] += 1
    return {
        "baseline_rows": len(baseline), "output_rows": len(output), "added_rows": len(added),
        "added_tracks": len(track_manifest), "added_rows_by_scene": dict(sorted(by_scene.items())),
        "added_rows_by_class": dict(sorted(by_class.items())),
        "added_rows_by_scene_class": dict(sorted(by_scene_class.items())),
    }


def _distribution_rows(rows: Sequence[OfficialTrack1Row], field: str) -> List[Dict[str, Any]]:
    counts = defaultdict(int)
    tracks = defaultdict(set)
    for row in rows:
        value = getattr(row, field)
        counts[value] += 1
        tracks[value].add((row.scene_id, row.class_id, row.object_id))
    return [{field: key, "rows": counts[key], "unique_tracks": len(tracks[key])} for key in sorted(counts.keys())]
