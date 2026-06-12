"""Configuration resolution for the Step 21C ByteTrack sweep."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from deep_oc_sort_3d.bytetrack_tuning.tuning_io import write_yaml


def load_tuning_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate the tuning configuration."""
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("ByteTrack tuning config must be a mapping")
    if not isinstance(value.get("variants"), dict) or not value.get("variants"):
        raise ValueError("ByteTrack tuning config requires at least one variant")
    value["_config_path"] = str(path)
    return value


def tuning_output_root(config: Dict[str, Any]) -> Path:
    """Return the isolated Step 21C output root."""
    section = config.get("bytetrack_coverage_tuning", {})
    return Path(str(section.get("output_root", "output/bytetrack_tuning/baseline_v2_pseudo3d_fullcam")))


def subset_entries(config: Dict[str, Any], phase: str) -> List[Tuple[str, str, str]]:
    """Resolve pipeline subset, dataset split and scene tuples for a phase."""
    groups = config.get("subsets", {}).get(phase, {})
    output = []
    for subset_name, payload in (groups.items() if isinstance(groups, dict) else []):
        if not isinstance(payload, dict):
            continue
        split = str(payload.get("split", ""))
        pipeline_subset = {"train": "internal_holdout", "val": "official_val", "test": "test"}.get(
            split,
            str(subset_name),
        )
        for scene_name in payload.get("scenes", []) or []:
            output.append((pipeline_subset, split, str(scene_name)))
    return sorted(set(output))


def variant_root(config: Dict[str, Any], variant_name: str) -> Path:
    """Return the isolated output root for one variant."""
    return tuning_output_root(config) / "sweep_runs" / str(variant_name)


def build_variant_pipeline_config(
    config: Dict[str, Any],
    variant_name: str,
    include_full_test: bool,
) -> Dict[str, Any]:
    """Build a Step 21B-compatible config rooted inside one sweep variant."""
    variants = config.get("variants", {})
    if variant_name not in variants:
        raise ValueError("Unknown ByteTrack tuning variant: %s" % variant_name)
    root = variant_root(config, variant_name)
    paths = config.get("paths", {})
    entries = subset_entries(config, "sweep_eval")
    if not include_full_test and bool(config.get("sweep", {}).get("include_test_probe_in_phase_a", False)):
        entries.extend(subset_entries(config, "test_probe"))
    if include_full_test:
        entries.extend(subset_entries(config, "full_test"))
    grouped = {}
    for subset, split, scene_name in sorted(set(entries)):
        grouped.setdefault(subset, {"split": split, "scenes": []})
        grouped[subset]["scenes"].append(scene_name)

    variant_settings = deepcopy(variants.get(variant_name, {}))
    person_specific = variant_settings.pop("person_specific", None)
    tracking = config.get("tracking", {})
    variant_settings["class_agnostic_tracking"] = bool(tracking.get("class_agnostic_tracking", False))
    variant_settings["allow_cross_class_matching"] = bool(tracking.get("allow_cross_class_matching", False))
    resolved = {
        "baseline_v2_bytetrack_local": {
            "name": variant_name,
            "progress": bool(config.get("bytetrack_coverage_tuning", {}).get("progress", True)),
            "random_seed": int(config.get("bytetrack_coverage_tuning", {}).get("random_seed", 42)),
        },
        "paths": {
            "dataset_root": paths.get("dataset_root"),
            "yolo_pipeline_root": paths.get("yolo_pipeline_root"),
            "v2_observations_root": paths.get("v2_observations_root"),
            "current_local_tracks_root": paths.get("baseline_v2_local_tracks_root"),
            "output_precheck_root": str(root / "summaries" / "precheck"),
            "output_local_tracks_root": str(root / "local_tracks"),
            "output_tracklets_root": str(root / "tracklets"),
            "output_candidates_root": str(root / "candidates"),
            "output_motion_clean_root": str(root / "motion_clean"),
            "output_global_root": str(root / "global_mtmc"),
            "output_final_export_root": str(root / "final_export"),
            "output_track1_root": str(root / "track1_submission"),
            "output_package_root": str(root / "package"),
            "output_comparison_root": str(root / "summaries"),
            "baseline_v1_global_root": paths.get("baseline_v1_global_root"),
            "baseline_v1_final_export_root": paths.get("baseline_v1_final_export_root"),
            "baseline_v1_track1_root": paths.get("baseline_v1_track1_root"),
            "baseline_v2_tracklets_root": paths.get("baseline_v2_tracklets_root"),
            "baseline_v2_candidates_root": paths.get("baseline_v2_candidates_root"),
            "baseline_v2_motion_clean_root": paths.get("baseline_v2_motion_clean_root"),
            "baseline_v2_global_root": paths.get("baseline_v2_global_root"),
            "baseline_v2_final_export_root": paths.get("baseline_v2_final_export_root"),
            "baseline_v2_track1_root": paths.get("baseline_v2_track1_root"),
        },
        "precheck": {"enabled": False, "require_pass_before_full_rerun": False},
        "full_rerun": {"enabled": True, "run_if_precheck_passes": False, "subsets": grouped},
        "bytetrack_style": variant_settings,
        "person_specific_requested": person_specific,
        "person_specific_applied": False,
        "global_association": {"use_reid": False, "use_existing_association": True},
        "selection": deepcopy(config.get("selection", {})),
    }
    return resolved


def write_resolved_configs(config: Dict[str, Any]) -> Dict[str, Path]:
    """Write the global resolved config and per-variant configs."""
    root = tuning_output_root(config)
    paths = {}
    write_yaml(root / "configs" / "resolved_config.yaml", _without_private_keys(config))
    for variant_name in sorted(config.get("variants", {}).keys()):
        variant_config = build_variant_pipeline_config(config, variant_name, include_full_test=False)
        path = variant_root(config, variant_name) / "variant_config.yaml"
        write_yaml(path, variant_config)
        paths[str(variant_name)] = path
    return paths


def _without_private_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}
