"""Discover and load V3-owned recovery sources without using V2 rows."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from deep_oc_sort_3d.final_export.generic_export import read_global_frame_records_file
from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.tracklets.tracklet_io import read_tracklets_file
from deep_oc_sort_3d.tracking.track_io import read_local_tracks_csv
from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import expected_scene_ids, internal_to_official
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import existing_paths, progress_iter


TrackKey = Tuple[int, str, int, int]


@dataclass
class RecoveryTrack:
    """One V3 local track eligible for controlled export analysis."""

    scene_id: int
    scene_name: str
    camera_id: str
    local_track_id: int
    internal_class_id: int
    official_class_id: int
    records: List[LocalTrackRecord] = field(default_factory=list)
    source_root: str = ""
    tracklet_quality_flag: str = "unknown"
    tracklet_valid_for_mtmc: Optional[bool] = None
    candidate_quality_flag: str = "unknown"
    candidate_is_candidate: Optional[bool] = None
    candidate_reject_reason: Optional[str] = None
    baseline_covered: bool = False
    p95_step_distance: Optional[float] = None
    max_step_distance: Optional[float] = None
    jump_ratio: Optional[float] = None

    @property
    def key(self) -> TrackKey:
        """Return a class-aware local-track identity."""
        return (self.scene_id, self.camera_id, self.local_track_id, self.internal_class_id)

    @property
    def length(self) -> int:
        """Return the number of detection-associated frame records."""
        return len(self.records)

    @property
    def mean_confidence(self) -> float:
        """Return mean detector confidence."""
        if not self.records:
            return 0.0
        return float(np.mean([float(record.confidence) for record in self.records]))

    @property
    def states(self) -> Set[str]:
        """Return normalized ByteTrack states observed for the track."""
        return set(str(record.track_state).strip().lower() for record in self.records)

    @property
    def geometry_valid_count(self) -> int:
        """Return records with finite center, positive dimensions and finite yaw."""
        return sum(1 for record in self.records if record_geometry_valid(record))

    @property
    def geometry_valid_ratio(self) -> float:
        """Return the valid-geometry fraction."""
        return float(self.geometry_valid_count) / float(self.length) if self.length else 0.0


@dataclass
class RecoverySources:
    """Loaded V3 source inventory and local tracks."""

    tracks: List[RecoveryTrack]
    local_roots: List[Path]
    tracklet_roots: List[Path]
    candidate_roots: List[Path]
    final_export_roots: List[Path]
    covered_track_keys: Set[TrackKey]
    warnings: List[str]


def discover_recovery_roots(config: Dict[str, Any]) -> Dict[str, List[Path]]:
    """Auto-detect old 023-025 and extension 026-027 V3 artifacts."""
    paths = config.get("paths", {})
    tuning = _configured_path(paths, "bytetrack_tuning_best_root")
    official = _configured_path(paths, "official_023_027_root", "output/official_023_027")
    v3_official = _configured_path(paths, "v3_official_root", str(official / "v3_gap_aware_soft"))
    extension = v3_official / "extension_026_027" / "pipeline_outputs"
    gap_root = _configured_path(paths, "v3_gap_aware_soft_root")
    roots = {
        "local": existing_paths([
            tuning / "local_tracks",
            _configured_path(paths, "bytetrack_local_root"),
            extension / "local_tracks",
        ]),
        "tracklets": existing_paths([
            tuning / "tracklets",
            _configured_path(paths, "bytetrack_tracklets_root"),
            extension / "tracklets",
        ]),
        "candidates": existing_paths([
            tuning / "candidates",
            _configured_path(paths, "bytetrack_candidates_root"),
            extension / "candidates",
        ]),
        "final_export": existing_paths([
            gap_root / "final_export",
            tuning / "final_export",
            extension / "gap_aware" / "filter_runs" / "gap_aware_soft" / "final_export",
            v3_official / "final_export",
        ]),
    }
    return roots


def load_recovery_sources(config: Dict[str, Any], progress: bool = True) -> RecoverySources:
    """Load local records and enrich them with V3 tracklet/candidate metadata."""
    roots = discover_recovery_roots(config)
    scene_ids = set(expected_scene_ids(config))
    mapping = internal_to_official(config)
    warnings = []
    track_map = {}
    local_files = _preferred_data_files(roots["local"], ".csv")
    for path in progress_iter(local_files, progress, "load V3 local tracks"):
        if "summar" in path.name.lower():
            continue
        try:
            records = read_local_tracks_csv(path)
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append("local_file_skipped:%s:%s" % (path, exc))
            continue
        for record in records:
            if int(record.scene_id) not in scene_ids or int(record.class_id) not in mapping:
                continue
            key = (int(record.scene_id), str(record.camera_id), int(record.local_track_id), int(record.class_id))
            track = track_map.get(key)
            if track is None:
                track = RecoveryTrack(
                    scene_id=int(record.scene_id), scene_name=str(record.scene_name), camera_id=str(record.camera_id),
                    local_track_id=int(record.local_track_id), internal_class_id=int(record.class_id),
                    official_class_id=int(mapping[int(record.class_id)]), source_root=str(path.parent.parent.parent),
                )
                track_map[key] = track
            track.records.append(record)
    tracklets = _load_tracklet_metadata(roots["tracklets"], scene_ids, warnings, progress)
    candidates = _load_candidate_metadata(roots["candidates"], scene_ids, warnings, progress)
    covered = _load_covered_track_keys(roots["final_export"], scene_ids, warnings, progress)
    for key, candidate in candidates.items():
        if candidate.global_track_id is not None:
            covered.add(key)
    for track in track_map.values():
        track.records = sorted(track.records, key=lambda item: (item.frame_id, item.detection_id))
        tracklet = tracklets.get(track.key)
        if tracklet is not None:
            track.tracklet_quality_flag = str(tracklet.quality_flag or "unknown")
            track.tracklet_valid_for_mtmc = bool(tracklet.is_valid_for_mtmc)
        candidate = candidates.get(track.key)
        if candidate is not None:
            track.candidate_quality_flag = str(candidate.quality_flag or "unknown")
            track.candidate_is_candidate = bool(candidate.is_candidate)
            track.candidate_reject_reason = candidate.reject_reason
        track.baseline_covered = track.key in covered
        track.p95_step_distance, track.max_step_distance, track.jump_ratio = motion_statistics(track.records)
    if not roots["local"]:
        warnings.append("no_local_track_roots_available")
    if not roots["final_export"]:
        warnings.append("no_final_export_roots_available_using_candidate_global_ids_only")
    return RecoverySources(
        tracks=sorted(track_map.values(), key=lambda item: item.key),
        local_roots=roots["local"], tracklet_roots=roots["tracklets"], candidate_roots=roots["candidates"],
        final_export_roots=roots["final_export"], covered_track_keys=covered, warnings=warnings,
    )


def record_geometry_valid(record: LocalTrackRecord) -> bool:
    """Return True for a detection row with valid Track1 geometry."""
    if int(record.detection_id) < 0 or record.center_3d is None or record.dimensions_3d is None or record.yaw is None:
        return False
    center = np.asarray(record.center_3d, dtype=float).reshape(-1)
    dims = np.asarray(record.dimensions_3d, dtype=float).reshape(-1)
    return bool(center.size >= 3 and dims.size >= 3 and np.all(np.isfinite(center[:3])) and np.all(np.isfinite(dims[:3])) and np.all(dims[:3] > 0.0) and math.isfinite(float(record.yaw)))


def motion_statistics(records: Sequence[LocalTrackRecord]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute gap-normalized 3D step diagnostics for one local track."""
    points = []
    for record in records:
        if record.center_3d is not None:
            center = np.asarray(record.center_3d, dtype=float).reshape(-1)
            if center.size >= 3 and np.all(np.isfinite(center[:3])):
                points.append((int(record.frame_id), center[:3]))
    if len(points) < 2:
        return None, None, None
    steps = []
    jumps = 0
    for previous, current in zip(points[:-1], points[1:]):
        gap = max(1, int(current[0]) - int(previous[0]))
        distance = float(np.linalg.norm(current[1] - previous[1])) / float(gap)
        steps.append(distance)
        if distance > 15.0:
            jumps += 1
    return float(np.percentile(steps, 95)), float(max(steps)), float(jumps) / float(len(steps))


def _load_tracklet_metadata(roots: Sequence[Path], scene_ids: Set[int], warnings: List[str], progress: bool) -> Dict[TrackKey, Any]:
    output = {}
    files = _preferred_data_files(roots, ".csv")
    for path in progress_iter(files, progress, "load V3 tracklet metadata"):
        if "summar" in path.name.lower():
            continue
        try:
            items = read_tracklets_file(path)
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append("tracklet_file_skipped:%s:%s" % (path, exc))
            continue
        for item in items:
            if int(item.scene_id) in scene_ids:
                output.setdefault((int(item.scene_id), str(item.camera_id), int(item.local_track_id), int(item.class_id)), item)
    return output


def _load_candidate_metadata(roots: Sequence[Path], scene_ids: Set[int], warnings: List[str], progress: bool) -> Dict[TrackKey, Any]:
    output = {}
    files = _preferred_data_files(roots, ".csv")
    for path in progress_iter(files, progress, "load V3 candidate metadata"):
        if "summar" in path.name.lower():
            continue
        try:
            items = read_candidates_file(path)
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append("candidate_file_skipped:%s:%s" % (path, exc))
            continue
        for item in items:
            if int(item.scene_id) in scene_ids:
                output.setdefault((int(item.scene_id), str(item.camera_id), int(item.local_track_id), int(item.class_id)), item)
    return output


def _load_covered_track_keys(roots: Sequence[Path], scene_ids: Set[int], warnings: List[str], progress: bool) -> Set[TrackKey]:
    output = set()
    files = _preferred_data_files(roots, ".csv")
    for path in progress_iter(files, progress, "load V3 assigned local tracks"):
        if "summar" in path.name.lower() or "generic" in path.name.lower():
            continue
        try:
            records = read_global_frame_records_file(path)
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append("final_export_file_skipped:%s:%s" % (path, exc))
            continue
        for record in records:
            if record.global_track_id is not None and int(record.scene_id) in scene_ids:
                output.add((int(record.scene_id), str(record.camera_id), int(record.local_track_id), int(record.class_id)))
    return output


def _preferred_data_files(roots: Sequence[Path], suffix: str) -> List[Path]:
    files = []
    seen_signatures = set()
    for root in roots:
        for path in sorted(root.rglob("*%s" % suffix)):
            if not path.is_file():
                continue
            signature = _logical_file_signature(path)
            if signature not in seen_signatures:
                files.append(path)
                seen_signatures.add(signature)
    return files


def _configured_path(values: Dict[str, Any], key: str, default: str = "") -> Path:
    value = values.get(key, default)
    if value in (None, ""):
        return Path("__missing_22b_path__") / key
    return Path(str(value))


def _logical_file_signature(path: Path) -> Tuple[str, str]:
    scene_name = ""
    for part in path.parts:
        if str(part).startswith("Warehouse_"):
            scene_name = str(part)
    return (scene_name or path.parent.name, path.name)
