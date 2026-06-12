"""Configuration and generated stage configs for Step 21B."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_io import write_yaml


STAGE_ORDER = [
    "precheck",
    "local_tracking",
    "tracklets",
    "candidates",
    "motion_filtering",
    "global_association",
    "final_export",
    "track1_export",
    "validation",
    "package",
    "comparison",
]


def load_bytetrack_pipeline_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate the Step 21B config."""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("ByteTrack local pipeline config must be a mapping")
    value["_config_path"] = str(path)
    return value


def selected_stages(stage: Optional[str]) -> List[str]:
    """Resolve one stage or the complete ordered pipeline."""
    if stage in (None, "", "all"):
        return list(STAGE_ORDER)
    if stage not in STAGE_ORDER:
        raise ValueError("Unknown stage %s. Valid stages: %s" % (stage, ", ".join(STAGE_ORDER)))
    return [str(stage)]


def configured_subsets(config: Dict[str, Any], section_name: str) -> List[Tuple[str, str, str]]:
    """Return pipeline subset, dataset split and scene triples."""
    section = config.get(section_name, {})
    groups = section.get("subsets", section) if section_name == "full_rerun" else section
    output = []
    if not isinstance(groups, dict):
        return output
    for subset_name, payload in groups.items():
        if subset_name in ("enabled", "require_pass_before_full_rerun", "run_if_precheck_passes"):
            continue
        if not isinstance(payload, dict):
            continue
        split = str(payload.get("split", ""))
        pipeline_subset = _pipeline_subset(split, str(subset_name))
        for scene_name in payload.get("scenes", []) or []:
            output.append((pipeline_subset, split, str(scene_name)))
    return sorted(set(output))


def prepare_generated_stage_configs(config: Dict[str, Any]) -> Dict[str, Path]:
    """Write isolated configs consumed by existing V2 stage CLIs."""
    paths = config.get("paths", {})
    output_root = Path(str(paths.get("output_comparison_root"))) / "generated_configs"
    output_root.mkdir(parents=True, exist_ok=True)
    subsets = ["official_val", "internal_holdout", "test"]
    values = {
        "tracklets": {
            "tracklets": {
                "tracking_root": paths.get("output_local_tracks_root"),
                "output_root": paths.get("output_tracklets_root"),
                "subsets": subsets,
                "scenes": None,
                "camera_ids": None,
                "min_length": 3,
                "min_mean_confidence": 0.01,
                "smooth_trajectory": True,
                "smoothing_window": 5,
                "progress": True,
            }
        },
        "candidates": {
            "candidates": {
                "tracklet_root": paths.get("output_tracklets_root"),
                "output_root": paths.get("output_candidates_root"),
                "subsets": subsets,
                "scenes": None,
                "camera_ids": None,
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
        },
        "motion_filtering": {
            "motion_filtering": {
                "candidate_root": paths.get("output_candidates_root"),
                "output_root": paths.get("output_motion_clean_root"),
                "motion_quality_config": "deep_oc_sort_3d/configs/mtmc_motion_quality_medium_conf001.yaml",
                "subsets": subsets,
                "scenes": None,
                "camera_ids": None,
                "require_3d_motion": True,
                "allow_suspicious_as_clean": False,
                "progress": True,
            }
        },
        "global_association": {
            "global_association": {
                "candidates_root": paths.get("output_motion_clean_root"),
                "output_root": paths.get("output_global_root"),
                "global_config": "deep_oc_sort_3d/configs/global_mtmc_transition_medium_conf001.yaml",
                "subsets": subsets,
                "scenes": None,
                "class_names": None,
                "max_candidates_per_scene": None,
                "progress": True,
            }
        },
        "final_export": {
            "final_export": {
                "root": paths.get("dataset_root"),
                "local_tracks_root": paths.get("output_local_tracks_root"),
                "global_mtmc_root": paths.get("output_global_root"),
                "output_root": paths.get("output_final_export_root"),
                "include_unassigned": True,
                "namespace_global_ids": True,
                "global_id_stride": 100000,
                "drop_invalid_bbox": True,
                "drop_unassigned_for_generic_export": True,
                "drop_invalid_bbox_for_generic_export": True,
                "progress": True,
                "subsets": _final_export_subsets(config),
            }
        },
        "track1_export": {
            "track1_export": {
                "generic_export_root": str(Path(str(paths.get("output_final_export_root"))) / "generic_tracking_export" / "test"),
                "output_root": paths.get("output_track1_root"),
                "schema_confirmed": True,
                "schema_yaml": "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml",
                "force_unconfirmed_preview": False,
                "subsets": ["test"],
                "scenes": ["Warehouse_023", "Warehouse_024", "Warehouse_025"],
                "progress": True,
            }
        },
    }
    output = {}
    for name, payload in values.items():
        path = output_root / ("%s.yaml" % name)
        write_yaml(path, payload)
        output[name] = path
    return output


def output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Return configured output paths as Path objects."""
    values = {}
    for key, value in config.get("paths", {}).items():
        if str(key).startswith("output_"):
            values[str(key)] = Path(str(value))
    return values


def _pipeline_subset(split: str, fallback: str) -> str:
    return {"train": "internal_holdout", "val": "official_val", "test": "test"}.get(split, fallback)


def _final_export_subsets(config: Dict[str, Any]) -> Dict[str, Any]:
    output = {}
    for subset, split, scene in configured_subsets(config, "full_rerun"):
        output.setdefault(subset, {"split": split, "scenes": []})
        output[subset]["scenes"].append(scene)
    return output
