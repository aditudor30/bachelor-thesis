"""Audit official V2/V3 coverage gaps and available V3 recovery sources."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row, read_track1_rows
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import output_root
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import write_csv, write_json, write_yaml
from deep_oc_sort_3d.v3_coverage_extension.recovery_source_loader import discover_recovery_roots, load_recovery_sources


def run_coverage_gap_audit(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Write baseline summaries, scene/class gaps and source availability."""
    root = output_root(config)
    audit_root = root / "audit"
    write_yaml(root / "configs" / "resolved_config.yaml", _clean_config(config))
    v2_path = Path(str(config.get("paths", {}).get("v2_official_track1", "")))
    v3_path = Path(str(config.get("paths", {}).get("v3_official_track1", "")))
    v2_rows = read_track1_rows(v2_path, progress=progress)
    v3_rows = read_track1_rows(v3_path, progress=progress)
    v2_summary = summarize_track1_rows(v2_rows, v2_path)
    v3_summary = summarize_track1_rows(v3_rows, v3_path)
    write_json(audit_root / "v2_official_reference_summary.json", v2_summary)
    write_json(audit_root / "v3_official_baseline_summary.json", v3_summary)
    gaps = coverage_gap_rows(v2_rows, v3_rows)
    write_csv(audit_root / "coverage_gaps_by_scene_class.csv", gaps, ["scene_id", "class_id", "v2_rows", "v3_rows", "delta_v3_minus_v2", "recovery_priority"])
    roots = discover_recovery_roots(config)
    availability = {
        "v2_official_track1": _path_status(v2_path), "v3_official_track1": _path_status(v3_path),
        "recovery_roots": {key: [_path_status(path) for path in values] for key, values in roots.items()},
    }
    write_json(audit_root / "input_availability_audit.json", availability)
    sources = load_recovery_sources(config, progress=progress)
    source_summary = {
        "policy": "V3-only; V2 rows are never recovery input",
        "local_roots": [str(path) for path in sources.local_roots],
        "tracklet_roots": [str(path) for path in sources.tracklet_roots],
        "candidate_roots": [str(path) for path in sources.candidate_roots],
        "final_export_roots": [str(path) for path in sources.final_export_roots],
        "local_tracks": len(sources.tracks),
        "already_covered_local_tracks": len(sources.covered_track_keys),
        "uncovered_local_tracks": sum(1 for track in sources.tracks if not track.baseline_covered),
        "geometry_complete_uncovered_tracks": sum(1 for track in sources.tracks if not track.baseline_covered and track.geometry_valid_count == track.length),
        "warnings": sources.warnings,
        "mechanisms_available": {
            "short_track_safe": bool(sources.local_roots and sources.covered_track_keys),
            "associated_tentative_export": bool(sources.local_roots and sources.covered_track_keys),
            "scene_class_targeted_recovery": bool(sources.local_roots and sources.covered_track_keys),
            "single_camera_keep_clean": bool(sources.local_roots and sources.covered_track_keys),
        },
    }
    write_json(audit_root / "candidate_recovery_sources.json", source_summary)
    return {"v2": v2_summary, "v3": v3_summary, "coverage_gaps": gaps, "sources": source_summary, "availability": availability}


def summarize_track1_rows(rows: Sequence[OfficialTrack1Row], path: Path) -> Dict[str, Any]:
    """Summarize one official Track1 file."""
    scenes = defaultdict(int)
    classes = defaultdict(int)
    tracks = set()
    lengths = defaultdict(int)
    for row in rows:
        scenes[str(row.scene_id)] += 1
        classes[str(row.class_id)] += 1
        key = (row.scene_id, row.class_id, row.object_id)
        tracks.add(key)
        lengths[key] += 1
    values = sorted(lengths.values())
    return {
        "path": str(path), "rows": len(rows), "unique_tracks": len(tracks),
        "rows_per_track_mean": float(len(rows)) / float(len(tracks)) if tracks else None,
        "rows_per_track_median": _percentile(values, 50),
        "scene_distribution": dict(sorted(scenes.items(), key=lambda item: int(item[0]))),
        "class_distribution": dict(sorted(classes.items(), key=lambda item: int(item[0]))),
    }


def coverage_gap_rows(v2_rows: Sequence[OfficialTrack1Row], v3_rows: Sequence[OfficialTrack1Row]) -> List[Dict[str, Any]]:
    """Return V2/V3 row deltas for every official scene/class cell."""
    v2 = defaultdict(int)
    v3 = defaultdict(int)
    for row in v2_rows:
        v2[(row.scene_id, row.class_id)] += 1
    for row in v3_rows:
        v3[(row.scene_id, row.class_id)] += 1
    output = []
    for scene_id, class_id in sorted(set(v2.keys()).union(v3.keys())):
        delta = v3[(scene_id, class_id)] - v2[(scene_id, class_id)]
        priority = "high" if scene_id in (23, 24, 26) and class_id in (0, 1, 3) else "conservative"
        output.append({"scene_id": scene_id, "class_id": class_id, "v2_rows": v2[(scene_id, class_id)], "v3_rows": v3[(scene_id, class_id)], "delta_v3_minus_v2": delta, "recovery_priority": priority})
    return output


def _percentile(values: Sequence[int], percentile: float) -> Any:
    if not values:
        return None
    import numpy as np
    return float(np.percentile(values, percentile))


def _path_status(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "is_file": path.is_file(), "is_dir": path.is_dir(), "size_bytes": path.stat().st_size if path.is_file() else None}


def _clean_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}
