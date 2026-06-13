"""Run requested V4 geometry variants over frozen V3.1 rows."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row
from deep_oc_sort_3d.v4_geometry_refinement.balanced_geometry_refiner import refine_geometry_balanced
from deep_oc_sort_3d.v4_geometry_refinement.dimension_consistency import stabilize_track_dimensions
from deep_oc_sort_3d.v4_geometry_refinement.geometry_metrics import compact_metrics, compute_geometry_metrics
from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import v31_track1_path, variant_root
from deep_oc_sort_3d.v4_geometry_refinement.outlier_repair import repair_position_outliers
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import read_geometry_rows, write_csv, write_json, write_variant_rows
from deep_oc_sort_3d.v4_geometry_refinement.track_smoothing import smooth_track_positions
from deep_oc_sort_3d.v4_geometry_refinement.yaw_refinement import refine_track_yaw


def run_geometry_variants(config: Dict[str, Any], variants: Sequence[str], progress: bool = True) -> Dict[str, Any]:
    """Build each requested variant while preserving every V3.1 row key."""
    baseline = read_geometry_rows(v31_track1_path(config), progress=progress)
    if not baseline:
        raise RuntimeError("V3.1 input is missing or empty: %s" % v31_track1_path(config))
    output = {}
    for variant in variants:
        refined, stage_changes = _run_variant(variant, baseline, config, progress)
        root = variant_root(config, variant)
        write_variant_rows(root / "track1.txt", refined, config)
        official_rows = read_geometry_rows(root / "track1.txt", progress=False)
        metrics = compute_geometry_metrics(official_rows, config, baseline_rows=baseline, stage_changes=stage_changes)
        compact = compact_metrics(metrics)
        write_json(root / "geometry_summary.json", compact)
        write_json(root / "changed_points_summary.json", _change_summary(metrics, stage_changes))
        write_csv(root / "per_scene_summary.csv", _per_group_rows(metrics, "scene"))
        write_csv(root / "per_class_summary.csv", _per_group_rows(metrics, "class"))
        output[variant] = {"rows": len(official_rows), "geometry_summary": compact, "stage_change_events": len(stage_changes)}
    return output


def _run_variant(variant: str, rows: Sequence[OfficialTrack1Row], config: Dict[str, Any], progress: bool) -> Tuple[List[OfficialTrack1Row], List[Dict[str, Any]]]:
    if variant == "v4_smooth_only":
        refined, changes = smooth_track_positions(rows, config, progress=progress)
        return refined, _label_stage(changes, "track_smoothing")
    if variant == "v4_outlier_repair":
        refined, changes = repair_position_outliers(rows, config, progress=progress)
        return refined, _label_stage(changes, "outlier_repair")
    if variant == "v4_dimension_consistency":
        refined, changes = stabilize_track_dimensions(rows, config, progress=progress)
        return refined, _label_stage(changes, "dimension_consistency")
    if variant == "v4_yaw_refinement":
        refined, changes = refine_track_yaw(rows, config, progress=progress)
        return refined, _label_stage(changes, "yaw_refinement")
    if variant == "v4_geometry_refined_balanced":
        return refine_geometry_balanced(rows, config, progress=progress)
    raise ValueError("Unknown V4 variant: %s" % variant)


def _label_stage(changes: Sequence[Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
    """Attach a stable stage name to standalone-variant change records."""
    output: List[Dict[str, Any]] = []
    for change in changes:
        item = dict(change)
        item.setdefault("stage", stage)
        output.append(item)
    return output


def _change_summary(metrics: Dict[str, Any], stage_changes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_stage = defaultdict(int)
    by_scene = defaultdict(int)
    by_class = defaultdict(int)
    for row in stage_changes:
        by_stage[str(row.get("stage", row.get("reason", "unknown")))] += 1
        if row.get("scene_id") is not None:
            by_scene[str(row.get("scene_id"))] += 1
        if row.get("class_id") is not None:
            by_class[str(row.get("class_id"))] += 1
    return {
        "points_changed": metrics.get("points_changed"), "points_repaired": metrics.get("points_repaired"),
        "tracks_with_repairs": metrics.get("tracks_with_repairs"), "mean_position_change_m": metrics.get("mean_position_change_m"),
        "p95_position_change_m": metrics.get("p95_position_change_m"), "max_position_change_m": metrics.get("max_position_change_m"),
        "dimension_change_count": metrics.get("dimension_change_count"), "yaw_changed_count": metrics.get("yaw_changed_count"),
        "stage_change_events": len(stage_changes), "change_events_by_stage": dict(sorted(by_stage.items())),
        "change_events_by_scene": dict(sorted(by_scene.items())), "change_events_by_class": dict(sorted(by_class.items())),
    }


def _per_group_rows(metrics: Dict[str, Any], group: str) -> List[Dict[str, Any]]:
    suffix = "scene" if group == "scene" else "class"
    distribution = metrics.get("%s_distribution" % suffix, {})
    step = metrics.get("step_p95_by_%s" % suffix, {})
    suspect = metrics.get("suspect_tracks_by_%s" % suffix, {})
    repairs = metrics.get("points_repaired_by_%s" % suffix, {})
    keys = sorted(set(distribution.keys()).union(step.keys()).union(str(key) for key in suspect.keys()).union(str(key) for key in repairs.keys()), key=int)
    dimension_changes = metrics.get("dimension_changes_by_class", {}) if group == "class" else {}
    yaw_changes = metrics.get("yaw_changes_by_class", {}) if group == "class" else {}
    return [{
        "%s_id" % suffix: int(key),
        "rows": distribution.get(key, 0),
        "step_p95": step.get(key),
        "suspect_tracks": suspect.get(key, suspect.get(int(key), 0)),
        "points_repaired": repairs.get(key, repairs.get(int(key), 0)),
        "dimension_changes": dimension_changes.get(key, dimension_changes.get(int(key), 0)),
        "yaw_changes": yaw_changes.get(key, yaw_changes.get(int(key), 0)),
    } for key in keys]
