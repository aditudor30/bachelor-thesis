"""Runner and generated stage configs for baseline_v2_pseudo3d_6cam."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from deep_oc_sort_3d.experiments.sixcam_subset import SixCamItem, sixcam_items_from_config


def run_sixcam_v2(config: Dict[str, Any], progress: bool, overwrite: bool, dry_run: bool = False) -> Dict[str, Any]:
    """Generate stage configs and run the 6cam V2 pipeline."""
    items = sixcam_items_from_config(config)
    generated = write_stage_configs(config, items)
    commands = build_sixcam_commands(generated, progress=progress, overwrite=overwrite)
    rows = []
    for stage, command in commands:
        printable = " ".join(command)
        print(printable)
        if dry_run:
            rows.append({"stage": stage, "status": "dry_run", "command": printable})
            continue
        subprocess.check_call(command)
        rows.append({"stage": stage, "status": "ok", "command": printable})
    output_root = _output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"dry_run": dry_run, "stages": rows, "generated_configs": {key: str(value) for key, value in generated.items()}}
    (output_root / "runbook.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_stage_configs(config: Dict[str, Any], items: List[SixCamItem]) -> Dict[str, Path]:
    """Write concrete stage configs for the 6cam pipeline."""
    paths = config.get("paths", {})
    root = _output_root(config) / "generated_configs"
    root.mkdir(parents=True, exist_ok=True)
    subsets = _subset_config(items)
    scenes = sorted(set([item.scene_name for item in items]))
    camera_ids = sorted(set([item.camera_id for item in items]))
    configs = {
        "observations": root / "baseline_v2_pseudo3d_6cam_observations.yaml",
        "local_tracking": root / "baseline_v2_pseudo3d_6cam_local_tracking.yaml",
        "tracklets": root / "baseline_v2_pseudo3d_6cam_tracklets.yaml",
        "candidates": root / "baseline_v2_pseudo3d_6cam_candidates.yaml",
        "motion_filtering": root / "baseline_v2_pseudo3d_6cam_motion_filtering.yaml",
        "global_association": root / "baseline_v2_pseudo3d_6cam_global_association.yaml",
        "final_export": root / "baseline_v2_pseudo3d_6cam_final_export.yaml",
        "track1_export": root / "baseline_v2_pseudo3d_6cam_track1_export.yaml",
    }
    _write_yaml(_observation_config(paths, subsets), configs["observations"])
    _write_yaml({"tracking": _tracking_config(paths, scenes, camera_ids)}, configs["local_tracking"])
    _write_yaml({"tracklets": _tracklets_config(paths, scenes, camera_ids)}, configs["tracklets"])
    _write_yaml({"candidates": _candidates_config(paths, scenes, camera_ids)}, configs["candidates"])
    _write_yaml({"motion_filtering": _motion_config(paths, scenes, camera_ids)}, configs["motion_filtering"])
    _write_yaml({"global_association": _global_config(paths, scenes)}, configs["global_association"])
    final_config = _final_config(paths, subsets, camera_ids)
    _write_yaml({"final_export": final_config}, configs["final_export"])
    _write_yaml({"track1_export": _track1_config(paths, scenes)}, configs["track1_export"])
    return configs


def build_sixcam_commands(configs: Dict[str, Path], progress: bool, overwrite: bool) -> List[Tuple[str, List[str]]]:
    """Build stage commands for the 6cam pipeline."""
    flag = "--progress" if progress else "--no-progress"
    out = [
        ("observations", [sys.executable, "-m", "deep_oc_sort_3d.scripts.build_pseudo3d_observations", "--config", str(configs["observations"]), flag]),
        ("local_tracking", [sys.executable, "-m", "deep_oc_sort_3d.scripts.run_baseline_v2_local_tracking", "--config", str(configs["local_tracking"]), flag]),
        ("tracklets", [sys.executable, "-m", "deep_oc_sort_3d.scripts.run_baseline_v2_tracklets", "--config", str(configs["tracklets"]), flag]),
        ("candidates", [sys.executable, "-m", "deep_oc_sort_3d.scripts.run_baseline_v2_candidates", "--config", str(configs["candidates"]), flag]),
        ("motion_filtering", [sys.executable, "-m", "deep_oc_sort_3d.scripts.run_baseline_v2_motion_filtering", "--config", str(configs["motion_filtering"]), flag]),
        ("global_association", [sys.executable, "-m", "deep_oc_sort_3d.scripts.run_baseline_v2_global_association", "--config", str(configs["global_association"]), flag]),
        ("final_export", [sys.executable, "-m", "deep_oc_sort_3d.scripts.run_baseline_v2_final_export", "--config", str(configs["final_export"]), flag]),
        ("track1_export", [sys.executable, "-m", "deep_oc_sort_3d.scripts.export_track1_submission", "--config", str(configs["track1_export"]), flag]),
    ]
    if overwrite:
        for stage, command in out:
            if stage != "track1_export":
                command.append("--overwrite")
    return out


def _observation_config(paths: Dict[str, Any], subsets: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "progress": True,
        "baseline_v2": {"name": "baseline_v2_pseudo3d_6cam", "output_root": paths.get("v2_6cam_pipeline_root")},
        "paths": {
            "dataset_root": paths.get("dataset_root"),
            "input_observations_root": "%s/observations3d" % paths.get("v1_pipeline_root"),
            "pseudo3d_predictions_root": paths.get("pseudo3d_predictions_root"),
            "output_observations_root": "%s/observations3d" % paths.get("v2_6cam_pipeline_root"),
            "output_metadata_root": "%s/observations3d_metadata" % paths.get("v2_6cam_pipeline_root"),
            "class_priors": paths.get("class_priors", "output/3d_priors/baseline_v1_geometry_only/priors/class_dimension_priors_final.json"),
        },
        "pseudo3d_integration": {
            "use_stabilized_pseudo3d": True,
            "prefer_pseudo3d_over_original": True,
            "fallback_to_original_3d_if_missing": True,
            "fallback_to_class_prior_dimensions": True,
            "drop_if_no_center_3d": False,
            "require_world_coordinates": True,
            "write_source_metadata": True,
            "bbox_iou_fallback_enabled": True,
            "bbox_iou_fallback_threshold": 0.8,
        },
        "subsets": subsets,
    }


def _tracking_config(paths: Dict[str, Any], scenes: List[str], camera_ids: List[str]) -> Dict[str, Any]:
    return {
        "run_root": paths.get("v2_6cam_pipeline_root"),
        "output_root": paths.get("v2_6cam_local_tracks_root"),
        "subsets": ["official_val", "internal_holdout"],
        "scenes": scenes,
        "camera_ids": camera_ids,
        "mode": "hybrid",
        "min_confidence": 0.01,
        "min_hits": 2,
        "max_misses": 30,
        "cost_threshold": 0.7,
        "max_3d_distance": 3.0,
        "min_iou": 0.05,
        "class_must_match": True,
        "max_detections_per_frame": 200,
        "progress": True,
    }


def _tracklets_config(paths: Dict[str, Any], scenes: List[str], camera_ids: List[str]) -> Dict[str, Any]:
    return {
        "tracking_root": paths.get("v2_6cam_local_tracks_root"),
        "output_root": paths.get("v2_6cam_tracklets_root"),
        "subsets": ["official_val", "internal_holdout"],
        "scenes": scenes,
        "camera_ids": camera_ids,
        "min_length": 3,
        "min_mean_confidence": 0.01,
        "smooth_trajectory": True,
        "smoothing_window": 5,
        "progress": True,
    }


def _candidates_config(paths: Dict[str, Any], scenes: List[str], camera_ids: List[str]) -> Dict[str, Any]:
    return {
        "tracklet_root": paths.get("v2_6cam_tracklets_root"),
        "output_root": paths.get("v2_6cam_candidates_root"),
        "subsets": ["official_val", "internal_holdout"],
        "scenes": scenes,
        "camera_ids": camera_ids,
        "min_length": 3,
        "min_mean_confidence": 0.01,
        "allowed_quality_flags": ["good", "fragmented"],
        "require_valid_for_mtmc": True,
        "require_3d": False,
        "trajectory_sample_rate": 1,
        "max_trajectory_points": 200,
        "export_rejected": False,
        "progress": True,
    }


def _motion_config(paths: Dict[str, Any], scenes: List[str], camera_ids: List[str]) -> Dict[str, Any]:
    return {
        "candidate_root": paths.get("v2_6cam_candidates_root"),
        "output_root": paths.get("v2_6cam_motion_clean_root"),
        "motion_quality_config": "deep_oc_sort_3d/configs/mtmc_motion_quality_medium_conf001.yaml",
        "subsets": ["official_val", "internal_holdout"],
        "scenes": scenes,
        "camera_ids": camera_ids,
        "require_3d_motion": True,
        "allow_suspicious_as_clean": False,
        "progress": True,
    }


def _global_config(paths: Dict[str, Any], scenes: List[str]) -> Dict[str, Any]:
    return {
        "candidates_root": paths.get("v2_6cam_motion_clean_root"),
        "output_root": paths.get("v2_6cam_global_root"),
        "global_config": "deep_oc_sort_3d/configs/global_mtmc_transition_medium_conf001.yaml",
        "subsets": ["official_val", "internal_holdout"],
        "scenes": scenes,
        "class_names": None,
        "max_candidates_per_scene": None,
        "progress": True,
    }


def _final_config(paths: Dict[str, Any], subsets: Dict[str, Any], camera_ids: List[str]) -> Dict[str, Any]:
    return {
        "root": paths.get("dataset_root"),
        "local_tracks_root": paths.get("v2_6cam_local_tracks_root"),
        "global_mtmc_root": paths.get("v2_6cam_global_root"),
        "output_root": paths.get("v2_6cam_final_export_root"),
        "include_unassigned": True,
        "namespace_global_ids": True,
        "global_id_stride": 100000,
        "drop_invalid_bbox": True,
        "drop_unassigned_for_generic_export": True,
        "drop_invalid_bbox_for_generic_export": True,
        "progress": True,
        "camera_ids": camera_ids,
        "subsets": subsets,
    }


def _track1_config(paths: Dict[str, Any], scenes: List[str]) -> Dict[str, Any]:
    return {
        "generic_export_root": "%s/generic_tracking_export" % paths.get("v2_6cam_final_export_root"),
        "output_root": paths.get("v2_6cam_track1_root", "output/track1_submission/baseline_v2_pseudo3d_6cam"),
        "schema_confirmed": True,
        "schema_yaml": "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml",
        "force_unconfirmed_preview": False,
        "subsets": ["official_val", "internal_holdout"],
        "scenes": scenes,
        "progress": True,
    }


def _subset_config(items: List[SixCamItem]) -> Dict[str, Any]:
    subsets = {}
    for item in items:
        current = subsets.setdefault(item.subset, {"split": item.split, "scenes": []})
        if item.scene_name not in current["scenes"]:
            current["scenes"].append(item.scene_name)
    return subsets


def _write_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _output_root(config: Dict[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    return Path(experiment.get("output_root", "output/baseline_v2_pseudo3d_6cam_comparison"))

