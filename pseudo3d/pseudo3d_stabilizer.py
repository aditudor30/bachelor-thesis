"""Track-level temporal stabilizer for isolated pseudo-3D predictions."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, List, Tuple

import numpy as np

from deep_oc_sort_3d.pseudo3d.bbox_stability import mark_small_bbox_unstable
from deep_oc_sort_3d.pseudo3d.depth_stabilization import stabilize_depth_sequence
from deep_oc_sort_3d.pseudo3d.jump_guard import apply_jump_guard, compute_step_distances
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import prediction_summary
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput
from deep_oc_sort_3d.pseudo3d.stabilized_yaw import estimate_yaw_from_smoothed_motion
from deep_oc_sort_3d.pseudo3d.temporal_smoothing import smooth_center_sequence


class Pseudo3DStabilizer:
    """Apply isolated temporal smoothing, jump guard, and yaw stabilization."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config or {}

    def stabilize_track(self, outputs: List[Pseudo3DOutput]) -> Tuple[List[Pseudo3DOutput], Dict[str, Any]]:
        """Stabilize one track-like sequence sorted by frame id."""
        if not outputs:
            return [], self._empty_track_report()
        ordered = sorted(outputs, key=lambda item: int(item.frame_id))
        frame_ids = [int(item.frame_id) for item in ordered]
        raw_centers, valid_centers = _centers_array(ordered)
        raw_depths, valid_depths = _depths_array(ordered)
        centers = raw_centers.copy()
        depths = raw_depths.copy()
        changed_by_smoothing = [False for _ in ordered]
        changed_depth = [False for _ in ordered]

        center_cfg = self._section("center_smoothing")
        if bool(center_cfg.get("enabled", True)):
            centers = smooth_center_sequence(
                centers,
                str(center_cfg.get("method", "median_filter")),
                int(center_cfg.get("window", 5)),
                float(center_cfg.get("alpha", 0.3)),
            )
            if not bool(center_cfg.get("interpolate_missing", False)):
                centers[~valid_centers, :] = np.nan
            changed_by_smoothing = _changed_center_indices(raw_centers, centers)

        depth_cfg = self._section("depth_smoothing")
        depth_report = {"num_clamped": 0, "changed_indices": []}
        if bool(depth_cfg.get("enabled", True)):
            depths, depth_report = stabilize_depth_sequence(
                depths,
                str(depth_cfg.get("method", "median_filter")),
                int(depth_cfg.get("window", 5)),
                float(depth_cfg.get("max_relative_change", 0.5)),
            )
            if not bool(depth_cfg.get("interpolate_missing", False)):
                depths[~valid_depths] = np.nan
            changed_depth = _changed_depth_indices(raw_depths, depths)

        step_before_guard = compute_step_distances(centers)
        jump_cfg = self._section("jump_guard")
        jump_report = {"num_jumps": 0, "jump_indices": [], "max_step_before": _max_finite(step_before_guard), "max_step_after": _max_finite(step_before_guard)}
        changed_by_jump = [False for _ in ordered]
        if bool(jump_cfg.get("enabled", True)):
            centers, jump_report = apply_jump_guard(
                centers,
                frame_ids,
                float(jump_cfg.get("max_step_m", 6.0)),
                str(jump_cfg.get("strategy", "hold_previous")),
            )
            for index in jump_report.get("corrected_indices", []):
                if 0 <= int(index) < len(changed_by_jump):
                    changed_by_jump[int(index)] = True

        small_bbox_cfg = self._section("small_bbox_guard")
        small_bbox_marks = [False for _ in ordered]
        if bool(small_bbox_cfg.get("enabled", True)):
            small_bbox_marks = mark_small_bbox_unstable(ordered, float(small_bbox_cfg.get("min_bbox_height_px", 12.0)))
            if bool(small_bbox_cfg.get("use_previous_depth_if_available", True)):
                depths = _hold_previous_depth_for_small_bboxes(depths, small_bbox_marks)
                changed_depth = _changed_depth_indices(raw_depths, depths)

        yaw_cfg = self._section("yaw")
        if bool(yaw_cfg.get("recompute_from_smoothed_motion", True)):
            yaws, yaw_sources = estimate_yaw_from_smoothed_motion(
                centers,
                frame_ids,
                float(yaw_cfg.get("min_displacement", 0.5)),
                float(yaw_cfg.get("default_yaw", 0.0)),
            )
        else:
            yaws = [float(item.yaw or 0.0) for item in ordered]
            yaw_sources = [str(item.yaw_source or "class_default") for item in ordered]

        stabilized = []
        for index, output in enumerate(ordered):
            stabilized.append(
                self._updated_output(
                    output,
                    centers[index],
                    depths[index],
                    yaws[index],
                    yaw_sources[index],
                    changed_by_smoothing[index],
                    changed_depth[index],
                    changed_by_jump[index],
                    small_bbox_marks[index],
                )
            )
        report = self._track_report(ordered, raw_centers, centers, depth_report, jump_report, changed_by_smoothing, changed_depth, changed_by_jump, small_bbox_marks)
        return stabilized, report

    def stabilize_batch(self, outputs: List[Pseudo3DOutput]) -> Tuple[List[Pseudo3DOutput], Dict[str, Any]]:
        """Group outputs by track and stabilize each group independently."""
        grouped = self._group_outputs(outputs)
        stabilized = []
        track_reports = []
        for key, group in grouped:
            track_outputs, report = self.stabilize_track(group)
            report["track_key"] = key
            stabilized.extend(track_outputs)
            track_reports.append(report)
        stabilized = sorted(stabilized, key=lambda item: (item.subset, item.scene_name, item.camera_id, int(item.frame_id), str(item.local_track_id), str(item.global_track_id), str(item.candidate_id)))
        summary = self._batch_report(outputs, stabilized, track_reports)
        return stabilized, summary

    def _updated_output(
        self,
        raw: Pseudo3DOutput,
        center: np.ndarray,
        depth: float,
        yaw: float,
        yaw_source: str,
        center_smoothed: bool,
        depth_smoothed: bool,
        jump_corrected: bool,
        small_bbox_guarded: bool,
    ) -> Pseudo3DOutput:
        output = replace(raw)
        output.dimensions_3d = deepcopy(raw.dimensions_3d)
        output.dimensions_3d_source = "class_prior" if raw.dimensions_3d is not None else raw.dimensions_3d_source
        output.is_gt_derived = False
        output.is_estimated_for_test = True
        output.pseudo3d_version = str(self._section("metadata").get("pseudo3d_version", "0.2_stabilized"))
        output.yaw = float(yaw)
        output.yaw_source = yaw_source
        if np.all(np.isfinite(center)):
            output.center_3d = np.asarray(center, dtype=float).copy()
            if raw.center_3d is None and output.failure_reason:
                output.failure_reason = None
                output.projection_valid = None
                output.projection_error_reason = None
                output.coordinate_frame = raw.coordinate_frame if raw.coordinate_frame != "unknown" else "world"
        else:
            output.center_3d = None
        if np.isfinite(depth):
            output.depth = float(depth)
        if jump_corrected:
            output.center_3d_source = "pseudo3d_jump_guarded"
            output.depth_source = "jump_guard"
            output.pseudo3d_method = "bbox_height_depth+jump_guard"
            output.confidence_3d = float(output.confidence_3d) * float(self._section("jump_guard").get("reduce_confidence_factor", 0.5))
        elif center_smoothed or depth_smoothed:
            output.center_3d_source = "pseudo3d_motion_smoothed" if output.center_3d is not None else output.center_3d_source
            output.depth_source = "temporal_smoothing" if output.depth is not None else output.depth_source
            output.pseudo3d_method = "bbox_height_depth+temporal_smoothing"
        if small_bbox_guarded:
            output.confidence_3d = float(output.confidence_3d) * float(self._section("small_bbox_guard").get("reduce_confidence_factor", 0.5))
            output.source_notes = _append_note(output.source_notes, "small_bbox_depth_guard")
        output.confidence_3d = max(0.0, min(1.0, float(output.confidence_3d)))
        if output.center_3d is not None and output.center_3d_source in ("", "unknown"):
            output.center_3d_source = "pseudo3d_bbox_height"
        if output.depth is not None and output.depth_source in ("", "unknown"):
            output.depth_source = "bbox_height_prior"
        return output

    def _section(self, name: str) -> Dict[str, Any]:
        if isinstance(self.config.get(name), dict):
            return self.config.get(name, {})
        nested = self.config.get("pseudo3d_stabilization", {})
        if isinstance(nested, dict) and isinstance(nested.get(name), dict):
            return nested.get(name, {})
        return {}

    def _group_outputs(self, outputs: List[Pseudo3DOutput]) -> List[Tuple[str, List[Pseudo3DOutput]]]:
        groups = {}
        for index, output in enumerate(outputs):
            key = _track_group_key(output, index)
            groups.setdefault(key, []).append(output)
        return [(key, groups[key]) for key in sorted(groups.keys())]

    def _empty_track_report(self) -> Dict[str, Any]:
        return {"num_records": 0, "num_valid_raw": 0, "num_center_smoothed": 0, "num_depth_smoothed": 0, "num_jump_corrected": 0}

    def _track_report(
        self,
        ordered: List[Pseudo3DOutput],
        raw_centers: np.ndarray,
        centers: np.ndarray,
        depth_report: Dict[str, Any],
        jump_report: Dict[str, Any],
        center_smoothed: List[bool],
        depth_smoothed: List[bool],
        jump_corrected: List[bool],
        small_bbox_marks: List[bool],
    ) -> Dict[str, Any]:
        before_steps = compute_step_distances(raw_centers)
        after_steps = compute_step_distances(centers)
        return {
            "subset": ordered[0].subset,
            "scene_name": ordered[0].scene_name,
            "camera_id": ordered[0].camera_id,
            "class_id": ordered[0].class_id,
            "class_name": ordered[0].class_name,
            "local_track_id": ordered[0].local_track_id,
            "global_track_id": ordered[0].global_track_id,
            "num_records": len(ordered),
            "num_valid_raw": sum(1 for item in ordered if item.center_3d is not None),
            "num_center_smoothed": sum(1 for value in center_smoothed if value),
            "num_depth_smoothed": sum(1 for value in depth_smoothed if value),
            "num_jump_corrected": sum(1 for value in jump_corrected if value),
            "num_small_bbox_guarded": sum(1 for value in small_bbox_marks if value),
            "raw_max_step_m": _max_finite(before_steps),
            "stabilized_max_step_m": _max_finite(after_steps),
            "jump_guard_num_jumps": jump_report.get("num_jumps", 0),
            "jump_guard_strategy": jump_report.get("strategy"),
            "depth_num_clamped": depth_report.get("num_clamped", 0),
        }

    def _batch_report(self, raw_outputs: List[Pseudo3DOutput], stabilized_outputs: List[Pseudo3DOutput], track_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = prediction_summary(stabilized_outputs)
        summary.update(
            {
                "num_raw_predictions": len(raw_outputs),
                "num_tracks": len(track_reports),
                "num_center_smoothed": sum(int(row.get("num_center_smoothed", 0) or 0) for row in track_reports),
                "num_depth_smoothed": sum(int(row.get("num_depth_smoothed", 0) or 0) for row in track_reports),
                "num_jump_corrected": sum(int(row.get("num_jump_corrected", 0) or 0) for row in track_reports),
                "num_small_bbox_guarded": sum(int(row.get("num_small_bbox_guarded", 0) or 0) for row in track_reports),
                "track_reports": track_reports,
            }
        )
        return summary


def _track_group_key(output: Pseudo3DOutput, index: int) -> str:
    prefix = "%s/%s/%s" % (output.subset, output.scene_name, output.camera_id)
    if output.local_track_id is not None:
        return "%s/local/%s" % (prefix, output.local_track_id)
    if output.global_track_id is not None:
        return "%s/global/%s" % (prefix, output.global_track_id)
    return "%s/singleton/%s/%s/%s" % (prefix, output.class_id, output.frame_id, index)


def _centers_array(outputs: List[Pseudo3DOutput]) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.full((len(outputs), 3), np.nan, dtype=float)
    valid = np.zeros((len(outputs),), dtype=bool)
    for index, output in enumerate(outputs):
        if output.center_3d is None:
            continue
        array = np.asarray(output.center_3d, dtype=float).reshape(-1)
        if array.size >= 3 and np.all(np.isfinite(array[:3])):
            centers[index, :] = array[:3]
            valid[index] = True
    return centers, valid


def _depths_array(outputs: List[Pseudo3DOutput]) -> Tuple[np.ndarray, np.ndarray]:
    depths = np.full((len(outputs),), np.nan, dtype=float)
    valid = np.zeros((len(outputs),), dtype=bool)
    for index, output in enumerate(outputs):
        if output.depth is None:
            continue
        value = float(output.depth)
        if np.isfinite(value):
            depths[index] = value
            valid[index] = True
    return depths, valid


def _changed_center_indices(raw_centers: np.ndarray, centers: np.ndarray) -> List[bool]:
    changed = []
    for raw, current in zip(raw_centers, centers):
        if np.all(np.isfinite(raw)) and np.all(np.isfinite(current)):
            changed.append(float(np.linalg.norm(raw - current)) > 1e-6)
        else:
            changed.append(False)
    return changed


def _changed_depth_indices(raw_depths: np.ndarray, depths: np.ndarray) -> List[bool]:
    changed = []
    for raw, current in zip(raw_depths, depths):
        if np.isfinite(raw) and np.isfinite(current):
            changed.append(abs(float(raw) - float(current)) > 1e-6)
        else:
            changed.append(False)
    return changed


def _hold_previous_depth_for_small_bboxes(depths: np.ndarray, small_bbox_marks: List[bool]) -> np.ndarray:
    output = depths.copy()
    previous = np.nan
    for index, value in enumerate(output):
        if bool(small_bbox_marks[index]) and np.isfinite(previous):
            output[index] = previous
        elif np.isfinite(value):
            previous = float(value)
    return output


def _append_note(existing: str, note: str) -> str:
    if note in str(existing):
        return str(existing)
    if existing:
        return "%s; %s" % (existing, note)
    return note


def _max_finite(values: np.ndarray) -> Any:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    return float(np.max(finite))
