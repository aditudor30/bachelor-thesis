"""Isolated orchestration for missing official test scenes."""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deep_oc_sort_3d.official_023_027.official_config import (
    dataset_root,
    output_root,
    scene_names,
    variant_extension_root,
)
from deep_oc_sort_3d.official_023_027.official_track1_io import write_json, write_yaml


def run_official_extension(
    config: Dict[str, Any],
    mode: str = "incremental",
    progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Run existing pipeline components for new scenes in isolated roots."""
    if mode not in ("incremental", "rerun_all"):
        raise ValueError("Unsupported mode: %s" % mode)
    scenes = scene_names(config, "new" if mode == "incremental" else "all")
    preflight = extension_preflight(config, scenes)
    write_json(output_root(config) / "audit" / "extension_preflight.json", preflight)
    if preflight.get("status") != "ok":
        raise RuntimeError("Extension preflight failed: %s" % "; ".join(preflight.get("errors", [])))
    generated = prepare_extension_configs(config, scenes, mode)
    commands = build_extension_commands(config, generated, progress, overwrite)
    rows = []
    started = time.time()
    for item in commands:
        marker = Path(str(item.get("marker"))) if item.get("marker") else None
        if skip_existing and not overwrite and marker is not None and marker.exists():
            rows.append({"stage": item.get("stage"), "status": "skipped_existing", "marker": str(marker), "command": ""})
            continue
        command = list(item.get("command", []))
        print("[%s] %s" % (item.get("stage"), " ".join(command)))
        stage_started = time.time()
        try:
            subprocess.check_call(command)
            rows.append(
                {
                    "stage": item.get("stage"),
                    "status": "ok",
                    "runtime_seconds": time.time() - stage_started,
                    "command": " ".join(command),
                }
            )
        except subprocess.CalledProcessError as exc:
            rows.append({"stage": item.get("stage"), "status": "error", "returncode": exc.returncode, "command": " ".join(command)})
            break
    stage_failed = any(row.get("status") == "error" for row in rows)
    if not stage_failed:
        _copy_internal_track1(config, overwrite=overwrite)
    expected_outputs = [
        variant_extension_root(config, "v2_current") / "track1_internal_026_027.txt",
        variant_extension_root(config, "v3_gap_aware_soft") / "track1_internal_026_027.txt",
    ]
    missing_outputs = [str(path) for path in expected_outputs if not path.is_file() or path.stat().st_size <= 0]
    status = "error" if stage_failed or missing_outputs else "ok"
    summary = {
        "status": status,
        "mode": mode,
        "scenes": scenes,
        "runtime_seconds": time.time() - started,
        "generated_configs": {key: str(value) for key, value in generated.items()},
        "stages": rows,
        "missing_outputs": missing_outputs,
        "v2_internal_track1": str(variant_extension_root(config, "v2_current") / "track1_internal_026_027.txt"),
        "v3_internal_track1": str(variant_extension_root(config, "v3_gap_aware_soft") / "track1_internal_026_027.txt"),
    }
    for variant in ("v2_current", "v3_gap_aware_soft"):
        write_json(variant_extension_root(config, variant) / "summary.json", summary)
    return summary


def extension_preflight(config: Dict[str, Any], scenes: List[str]) -> Dict[str, Any]:
    """Check dataset scenes, detector config/checkpoint and reusable configs."""
    paths = config.get("paths", {})
    detector_config_path = Path(str(paths.get("detector_pipeline_config", "deep_oc_sort_3d/configs/pipeline_yolo11m_medium_conf001.yaml")))
    detector_config = _load_yaml(detector_config_path)
    detector_model = Path(str(detector_config.get("pipeline", {}).get("detector_model", "")))
    errors = []
    if not detector_config_path.is_file():
        errors.append("detector_pipeline_config_missing:%s" % detector_config_path)
    if not detector_model.is_file():
        errors.append("detector_checkpoint_missing:%s" % detector_model)
    configured_internal = _normalize_mapping(config.get("class_mapping", {}).get("internal", {}))
    detector_internal = _normalize_mapping(detector_config.get("classes", {}))
    if configured_internal != detector_internal:
        errors.append("detector_internal_class_mapping_mismatch")
    scene_rows = []
    for scene in scenes:
        root = dataset_root(config) / "test" / scene
        missing = []
        for name in ("videos", "calibration.json", "map.png"):
            path = root / name
            valid = path.is_dir() if name == "videos" else path.is_file()
            if not valid:
                missing.append(name)
        video_files = []
        videos = root / "videos"
        if videos.is_dir():
            video_files = [path for path in videos.iterdir() if path.is_file() and path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
            if not video_files:
                missing.append("video_files")
        if missing:
            errors.append("scene_missing_required:%s:%s" % (scene, ",".join(missing)))
        scene_rows.append({"scene_name": scene, "root": str(root), "video_files": len(video_files), "missing": missing, "status": "ok" if not missing else "error"})
    required_configs = [
        Path("deep_oc_sort_3d/configs/pseudo3d_isolated_debug.yaml"),
        Path("deep_oc_sort_3d/configs/pseudo3d_stabilization_debug.yaml"),
        Path("deep_oc_sort_3d/configs/mtmc_motion_quality_medium_conf001.yaml"),
        Path("deep_oc_sort_3d/configs/global_mtmc_transition_medium_conf001.yaml"),
        Path("deep_oc_sort_3d/configs/track1_schema_confirmed.yaml"),
    ]
    for path in required_configs:
        if not path.is_file():
            errors.append("required_config_missing:%s" % path)
    return {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "scenes": scene_rows,
        "detector_pipeline_config": str(detector_config_path),
        "detector_model": str(detector_model),
    }


def prepare_extension_configs(config: Dict[str, Any], scenes: List[str], mode: str) -> Dict[str, Path]:
    """Generate configs consumed by existing pipeline scripts."""
    processing_root = output_root(config)
    generated_root = processing_root / "generated_configs" / mode
    v2_root = variant_extension_root(config, "v2_current") / "pipeline_outputs"
    v3_root = variant_extension_root(config, "v3_gap_aware_soft") / "pipeline_outputs"
    detector_base_path = Path(str(config.get("paths", {}).get("detector_pipeline_config", "deep_oc_sort_3d/configs/pipeline_yolo11m_medium_conf001.yaml")))
    detector = _load_yaml(detector_base_path)
    detector_run_name = "yolo11m_medium_conf001_official_%s" % ("026_027" if mode == "incremental" else "023_027")
    detector.setdefault("pipeline", {})["root"] = str(dataset_root(config))
    detector["pipeline"]["output_root"] = str(v2_root / "detection_pipeline")
    detector["pipeline"]["run_name"] = detector_run_name
    detector["pipeline"]["max_frames"] = None
    detector["pipeline"]["camera_ids"] = None
    detector["subsets"] = {"test": {"split": "test", "scenes": scenes}}
    detection_run_root = v2_root / "detection_pipeline" / detector_run_name

    pseudo_root = v2_root / "pseudo3d_fullcam"
    pseudo = {
        "step15g": {"name": "official_023_027_pseudo3d", "output_root": str(pseudo_root), "progress": True},
        "paths": {
            "dataset_root": str(dataset_root(config)),
            "observations_root": str(detection_run_root / "observations3d"),
            "class_priors": config.get("paths", {}).get("class_priors_path"),
            "raw_output_root": str(pseudo_root / "predictions_raw"),
            "stabilized_output_root": str(pseudo_root / "predictions_stabilized"),
        },
        "input_selection": {"source": "observations3d", "subsets": ["test"], "scenes": {"test": scenes}, "camera_ids": None},
        "generation": {"skip_existing": True, "overwrite": False, "backend_config": "deep_oc_sort_3d/configs/pseudo3d_isolated_debug.yaml", "output_format": {"jsonl": True, "csv": True}},
        "stabilization": {"skip_existing": True, "overwrite": False, "config": "deep_oc_sort_3d/configs/pseudo3d_stabilization_debug.yaml", "output_format": {"jsonl": True, "csv": True}},
        "coverage": {"min_camera_file_coverage": 0.95, "min_prediction_success_rate": 0.90, "require_metadata_completeness": True, "check_projection": True, "check_smoothness": True},
        "smoothness": {"suspicious_step_m": 3.0, "invalid_step_m": 6.0, "suspicious_dimension_cv": 0.25, "invalid_dimension_cv": 0.50, "yaw_jump_threshold": 1.57},
    }
    observations_root = v2_root / "v2_observations"
    observations = {
        "baseline_v2_fullcam": {"name": "v2_current_official_extension", "progress": True},
        "paths": {
            "input_observations_root": str(detection_run_root / "observations3d"),
            "pseudo3d_predictions_root": str(pseudo_root / "predictions_stabilized"),
            "output_observations_root": str(observations_root / "observations3d"),
            "output_metadata_root": str(observations_root / "observations3d_metadata"),
            "output_comparison_root": str(v2_root / "observation_diagnostics"),
            "class_priors": config.get("paths", {}).get("class_priors_path"),
        },
        "pseudo3d_integration": {"output_source": "baseline_v2_pseudo3d_fullcam", "use_stabilized_pseudo3d": True, "prefer_pseudo3d_over_original": True, "fallback_to_original_3d_if_missing": True, "fallback_to_class_prior_dimensions": True, "drop_if_no_center_3d": False, "require_world_coordinates": True, "write_source_metadata": True, "bbox_iou_fallback_enabled": True, "bbox_iou_fallback_threshold": 0.8, "min_required_pseudo3d_used_rate": 0.95, "fail_if_below_threshold": True},
        "subsets": {"test": {"split": "test", "scenes": scenes}},
    }
    generated = {
        "detector": generated_root / "detector.yaml",
        "pseudo3d": generated_root / "pseudo3d.yaml",
        "observations": generated_root / "observations.yaml",
        "detection_run_root": detection_run_root,
    }
    write_yaml(generated["detector"], detector)
    write_yaml(generated["pseudo3d"], pseudo)
    write_yaml(generated["observations"], observations)
    generated.update(_write_v2_configs(config, generated_root, v2_root, observations_root, scenes))
    generated.update(_write_v3_configs(config, generated_root, v3_root, detection_run_root, observations_root, scenes))
    return generated


def build_extension_commands(
    config: Dict[str, Any],
    generated: Dict[str, Path],
    progress: bool,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    """Build the ordered extension command list."""
    v2_root = variant_extension_root(config, "v2_current") / "pipeline_outputs"
    v3_root = variant_extension_root(config, "v3_gap_aware_soft") / "pipeline_outputs"
    flag = "--progress" if progress else "--no-progress"
    overwrite_flag = ["--overwrite"] if overwrite else []
    commands = [
        _command("detection", "deep_oc_sort_3d.scripts.run_detection_to_observation_pipeline", generated["detector"], flag, overwrite_flag, generated["detection_run_root"] / "summaries" / "observations_summary.csv"),
        _command("pseudo3d", "deep_oc_sort_3d.scripts.run_step15g_fullcam_pseudo3d", generated["pseudo3d"], flag, overwrite_flag, v2_root / "pseudo3d_fullcam" / "summaries" / "step15g_summary.json"),
        _command("v2_observations", "deep_oc_sort_3d.scripts.build_fullcam_pseudo3d_observations", generated["observations"], flag, overwrite_flag, v2_root / "observation_diagnostics" / "diagnostics" / "pseudo3d_usage_fullcam.json"),
    ]
    for stage, module in [
        ("v2_local_tracking", "deep_oc_sort_3d.scripts.run_baseline_v2_local_tracking"),
        ("v2_tracklets", "deep_oc_sort_3d.scripts.run_baseline_v2_tracklets"),
        ("v2_candidates", "deep_oc_sort_3d.scripts.run_baseline_v2_candidates"),
        ("v2_motion_filtering", "deep_oc_sort_3d.scripts.run_baseline_v2_motion_filtering"),
        ("v2_global_association", "deep_oc_sort_3d.scripts.run_baseline_v2_global_association"),
        ("v2_final_export", "deep_oc_sort_3d.scripts.run_baseline_v2_final_export"),
        ("v2_track1_export", "deep_oc_sort_3d.scripts.export_track1_submission"),
    ]:
        commands.append(_command(stage, module, generated[stage], flag, [] if stage == "v2_track1_export" else overwrite_flag, _stage_marker(v2_root, stage)))
    byte_command = [sys.executable, "-m", "deep_oc_sort_3d.scripts.run_baseline_v2_bytetrack_local_stage", "--config", str(generated["v3_bytetrack"]), "--stage", "local_tracking", flag] + overwrite_flag
    commands.append({"stage": "v3_bytetrack_local", "command": byte_command, "marker": str(v3_root / "local_tracks" / "summaries" / "local_tracking_summary.json")})
    commands.append(_command("v3_tracklets", "deep_oc_sort_3d.scripts.run_baseline_v2_tracklets", generated["v3_tracklets"], flag, overwrite_flag, v3_root / "tracklets" / "summaries" / "tracklet_summary.json"))
    commands.append(_command("v3_candidates", "deep_oc_sort_3d.scripts.run_baseline_v2_candidates", generated["v3_candidates"], flag, overwrite_flag, v3_root / "candidates" / "summaries" / "candidate_summary.json"))
    gap_command = [sys.executable, "-m", "deep_oc_sort_3d.scripts.run_bytetrack_gap_aware_motion_filter_variant_21e", "--config", str(generated["v3_gap_aware"]), "--variant", "gap_aware_soft", flag]
    if overwrite:
        gap_command.append("--overwrite")
    commands.append({"stage": "v3_gap_aware_soft_downstream", "command": gap_command, "marker": str(v3_root / "gap_aware" / "filter_runs" / "gap_aware_soft" / "track1_submission" / "track1.txt")})
    return commands


def _write_v2_configs(config: Dict[str, Any], root: Path, v2_root: Path, observations_root: Path, scenes: List[str]) -> Dict[str, Path]:
    values = {
        "v2_local_tracking": {"tracking": {"run_root": str(observations_root), "output_root": str(v2_root / "local_tracks"), "subsets": ["test"], "scenes": scenes, "camera_ids": None, "mode": "hybrid", "min_confidence": 0.01, "min_hits": 2, "max_misses": 45, "cost_threshold": 0.8, "max_3d_distance": 4.0, "min_iou": 0.0, "class_must_match": True, "max_detections_per_frame": 200, "progress": True}},
        "v2_tracklets": _tracklet_config(v2_root / "local_tracks", v2_root / "tracklets", scenes),
        "v2_candidates": _candidate_config(v2_root / "tracklets", v2_root / "candidates", scenes),
        "v2_motion_filtering": {"motion_filtering": {"candidate_root": str(v2_root / "candidates"), "output_root": str(v2_root / "motion_clean"), "motion_quality_config": "deep_oc_sort_3d/configs/mtmc_motion_quality_medium_conf001.yaml", "subsets": ["test"], "scenes": scenes, "camera_ids": None, "require_3d_motion": True, "allow_suspicious_as_clean": False, "progress": True}},
        "v2_global_association": _global_config(v2_root / "motion_clean", v2_root / "global_mtmc", scenes),
        "v2_final_export": _final_export_config(config, v2_root / "local_tracks", v2_root / "global_mtmc", v2_root / "final_export", scenes),
        "v2_track1_export": _track1_export_config(v2_root / "final_export", v2_root / "track1_submission", scenes),
    }
    output = {}
    for key, value in values.items():
        path = root / (key + ".yaml")
        write_yaml(path, value)
        output[key] = path
    return output


def _write_v3_configs(config: Dict[str, Any], root: Path, v3_root: Path, detection_root: Path, observations_root: Path, scenes: List[str]) -> Dict[str, Path]:
    bytetrack = {
        "baseline_v2_bytetrack_local": {"name": "v3_gap_aware_soft_official_extension", "progress": True},
        "paths": {"dataset_root": str(dataset_root(config)), "yolo_pipeline_root": str(detection_root), "v2_observations_root": str(observations_root / "observations3d"), "output_precheck_root": str(v3_root / "precheck"), "output_local_tracks_root": str(v3_root / "local_tracks"), "output_comparison_root": str(v3_root / "comparison")},
        "precheck": {"enabled": False, "require_pass_before_full_rerun": False},
        "full_rerun": {"enabled": True, "run_if_precheck_passes": False, "subsets": {"test": {"split": "test", "scenes": scenes}}},
        "bytetrack_style": {"min_confidence_for_input": 0.001, "track_high_thresh": 0.3, "track_low_thresh": 0.05, "new_track_thresh": 0.4, "match_thresh": 0.8, "second_stage_match_thresh": 0.5, "track_buffer": 45, "class_agnostic_tracking": False, "allow_cross_class_matching": False},
    }
    paths = {
        "v3_bytetrack": root / "v3_bytetrack.yaml",
        "v3_tracklets": root / "v3_tracklets.yaml",
        "v3_candidates": root / "v3_candidates.yaml",
        "v3_gap_aware": root / "v3_gap_aware.yaml",
    }
    write_yaml(paths["v3_bytetrack"], bytetrack)
    write_yaml(paths["v3_tracklets"], _tracklet_config(v3_root / "local_tracks", v3_root / "tracklets", scenes))
    write_yaml(paths["v3_candidates"], _candidate_config(v3_root / "tracklets", v3_root / "candidates", scenes))
    gap = {
        "bytetrack_gap_aware_motion_filter": {"name": "v3_gap_aware_soft_official_extension", "output_root": str(v3_root / "gap_aware"), "progress": True, "random_seed": 42},
        "paths": {"dataset_root": str(dataset_root(config)), "bytetrack_21c_best_variant_root": str(v3_root), "bytetrack_21c_best_candidates_root": str(v3_root / "candidates")},
        "subsets": {"test": {"split": "test", "scenes": scenes}},
        "velocity_priors": {"estimate_from_gt_train_val": False, "estimate_from_clean_tracks_if_gt_missing": True, "percentile_for_vmax": 95, "margin_scale": 1.0, "min_samples_per_class": 100, "fallback_velocity_priors": _fallback_velocity_priors()},
        "current_motion_quality_config": "deep_oc_sort_3d/configs/mtmc_motion_quality_medium_conf001.yaml",
        "variants": {"gap_aware_soft": {"mode": "gap_aware_soft", "base_step_threshold": 6.0, "gap_factor": 0.8, "absolute_cap_multiplier": 1.0, "max_suspicious_jumps": 1, "max_accepted_jump_ratio": 0.25}},
        "pipeline": {"run_motion_filtering": True, "run_global_association": True, "run_final_export": True, "run_track1_export": True, "run_validation": True, "global_config": "deep_oc_sort_3d/configs/global_mtmc_transition_medium_conf001.yaml", "track1_schema": "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml"},
        "selection": {"require_track1_errors_zero": False, "max_allowed_purity_drop": 0.01, "max_allowed_false_merge_rate_delta": 0.01, "min_motion_clean_retention_gain": 0.0, "min_track1_rows_retention_gain": 0.0},
        "figures": {"enabled": False},
    }
    write_yaml(paths["v3_gap_aware"], gap)
    return paths


def _tracklet_config(local_root: Path, output: Path, scenes: List[str]) -> Dict[str, Any]:
    return {"tracklets": {"tracking_root": str(local_root), "output_root": str(output), "subsets": ["test"], "scenes": scenes, "camera_ids": None, "min_length": 3, "min_mean_confidence": 0.01, "smooth_trajectory": True, "smoothing_window": 5, "progress": True}}


def _candidate_config(tracklet_root: Path, output: Path, scenes: List[str]) -> Dict[str, Any]:
    return {"candidates": {"tracklet_root": str(tracklet_root), "output_root": str(output), "subsets": ["test"], "scenes": scenes, "camera_ids": None, "min_length": 3, "min_mean_confidence": 0.01, "allowed_quality_flags": ["good", "fragmented"], "require_valid_for_mtmc": True, "require_3d": False, "trajectory_sample_rate": 1, "max_trajectory_points": 200, "export_rejected": False, "progress": True}}


def _global_config(candidates: Path, output: Path, scenes: List[str]) -> Dict[str, Any]:
    return {"global_association": {"candidates_root": str(candidates), "output_root": str(output), "global_config": "deep_oc_sort_3d/configs/global_mtmc_transition_medium_conf001.yaml", "subsets": ["test"], "scenes": scenes, "class_names": None, "max_candidates_per_scene": None, "progress": True}}


def _final_export_config(config: Dict[str, Any], local: Path, global_root: Path, output: Path, scenes: List[str]) -> Dict[str, Any]:
    return {"final_export": {"root": str(dataset_root(config)), "local_tracks_root": str(local), "global_mtmc_root": str(global_root), "output_root": str(output), "include_unassigned": True, "namespace_global_ids": True, "global_id_stride": 100000, "drop_invalid_bbox": True, "drop_unassigned_for_generic_export": True, "drop_invalid_bbox_for_generic_export": True, "progress": True, "subsets": {"test": {"split": "test", "scenes": scenes}}}}


def _track1_export_config(final_root: Path, output: Path, scenes: List[str]) -> Dict[str, Any]:
    return {"track1_export": {"generic_export_root": str(final_root / "generic_tracking_export" / "test"), "output_root": str(output), "schema_confirmed": True, "schema_yaml": "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml", "force_unconfirmed_preview": False, "subsets": ["test"], "scenes": scenes, "progress": True}}


def _fallback_velocity_priors() -> Dict[str, Any]:
    return {"Person": {"v_max": 3.0, "margin": 1.5, "absolute_cap": 12.0}, "Forklift": {"v_max": 5.0, "margin": 2.0, "absolute_cap": 18.0}, "PalletTruck": {"v_max": 4.0, "margin": 2.0, "absolute_cap": 15.0}, "Transporter": {"v_max": 5.0, "margin": 2.0, "absolute_cap": 18.0}, "FourierGR1T2": {"v_max": 3.0, "margin": 1.5, "absolute_cap": 12.0}, "AgilityDigit": {"v_max": 3.0, "margin": 1.5, "absolute_cap": 12.0}, "NovaCarter": {"v_max": 4.0, "margin": 2.0, "absolute_cap": 15.0}}


def _command(stage: str, module: str, config_path: Path, progress_flag: str, extra: List[str], marker: Path) -> Dict[str, Any]:
    command = [sys.executable, "-m", module, "--config", str(config_path)]
    if module != "deep_oc_sort_3d.scripts.run_detection_to_observation_pipeline":
        command.append(progress_flag)
    elif progress_flag == "--no-progress":
        command.append("--no-progress")
    command.extend(extra)
    return {"stage": stage, "command": command, "marker": str(marker)}


def _stage_marker(root: Path, stage: str) -> Path:
    values = {"v2_local_tracking": root / "local_tracks" / "summaries" / "local_tracking_summary.csv", "v2_tracklets": root / "tracklets" / "summaries" / "tracklet_summary.json", "v2_candidates": root / "candidates" / "summaries" / "candidate_summary.json", "v2_motion_filtering": root / "motion_clean" / "summaries" / "motion_quality_summary.json", "v2_global_association": root / "global_mtmc" / "summaries" / "global_transition_summary.json", "v2_final_export": root / "final_export" / "generic_tracking_export", "v2_track1_export": root / "track1_submission" / "track1.txt"}
    return values[stage]


def _copy_internal_track1(config: Dict[str, Any], overwrite: bool) -> None:
    v2_source = variant_extension_root(config, "v2_current") / "pipeline_outputs" / "track1_submission" / "track1.txt"
    v3_source = variant_extension_root(config, "v3_gap_aware_soft") / "pipeline_outputs" / "gap_aware" / "filter_runs" / "gap_aware_soft" / "track1_submission" / "track1.txt"
    for variant, source in (("v2_current", v2_source), ("v3_gap_aware_soft", v3_source)):
        target = variant_extension_root(config, variant) / "track1_internal_026_027.txt"
        if not source.exists():
            continue
        if target.exists() and not overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def _normalize_mapping(value: Any) -> Dict[int, str]:
    if not isinstance(value, dict):
        return {}
    return {int(key): str(name) for key, name in value.items()}
