"""Configuration loading for controlled global tuning sweeps."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.global_tuning.tuning_io import load_yaml, write_yaml
from deep_oc_sort_3d.global_tuning.tuning_run_spec import GlobalTuningRunSpec, list_value


DEFAULT_SUBSETS = {
    "official_val": {"split": "val", "scenes": ["Warehouse_020", "Warehouse_021", "Warehouse_022"]},
    "internal_holdout": {"split": "train", "scenes": ["Warehouse_014", "Warehouse_015", "Warehouse_016"]},
    "test": {"split": "test", "scenes": ["Warehouse_023", "Warehouse_024", "Warehouse_025"]},
}


def load_sweep_config(path: Path) -> Dict[str, Any]:
    """Load a sweep config."""
    return load_yaml(path)


def load_run_config(path: Path) -> Dict[str, Any]:
    """Load a single run config."""
    data = load_yaml(path)
    section = data.get("global_tuning_run", data)
    return section if isinstance(section, dict) else {}


def build_run_specs_from_sweep(
    sweep_config_path: Path,
    run_names: Optional[List[str]] = None,
) -> List[GlobalTuningRunSpec]:
    """Build resolved run specs from a sweep YAML."""
    sweep = load_sweep_config(sweep_config_path)
    global_section = sweep.get("global_tuning", {})
    paths = sweep.get("paths", {})
    output_root = Path(paths.get("output_root", global_section.get("output_root", "output/global_tuning/debug")))
    requested = None if run_names is None else set([str(name) for name in run_names])
    specs = []
    for run_item in sweep.get("runs", []):
        if not isinstance(run_item, dict):
            continue
        name = str(run_item.get("name", ""))
        if not name:
            continue
        if requested is not None and name not in requested:
            continue
        config_path = Path(str(run_item.get("config", "")))
        spec = build_run_spec_from_config(
            run_name=name,
            config_path=config_path,
            output_root=output_root / "runs" / name,
            sweep_paths=paths,
            sweep_defaults=sweep,
        )
        specs.append(spec)
    return specs


def build_run_spec_from_config(
    run_name: str,
    config_path: Path,
    output_root: Optional[Path] = None,
    sweep_paths: Optional[Dict[str, Any]] = None,
    sweep_defaults: Optional[Dict[str, Any]] = None,
) -> GlobalTuningRunSpec:
    """Build a single run spec from a run config."""
    config = load_run_config(config_path)
    paths = dict(sweep_paths or {})
    paths.update(config.get("paths", {}))
    final_export = dict(config.get("final_export", {}))
    track1_export = dict(config.get("track1_export", {}))
    compact_export = dict(config.get("compact_export", {}))
    if "subsets" not in final_export and sweep_defaults is not None:
        if isinstance(sweep_defaults.get("subsets"), dict):
            final_export["subsets"] = sweep_defaults.get("subsets")
    if output_root is None:
        output_root = Path(config.get("output_root", "output/global_tuning/debug/runs/%s" % run_name))
    subsets = config.get("subsets")
    if subsets is None and sweep_defaults is not None:
        subsets = sweep_defaults.get("subsets")
    spec = GlobalTuningRunSpec(
        name=run_name,
        output_root=Path(output_root),
        config_path=config_path,
        dataset_root=str(paths.get("dataset_root", "/path/to/MTMC_Tracking_2026")),
        input_motion_clean_root=Path(str(paths.get("input_motion_clean_root", "output/mtmc_candidates_motion_clean/baseline_v2_pseudo3d_fullcam"))),
        local_tracks_root=Path(str(paths.get("local_tracks_root", "output/local_tracks/baseline_v2_pseudo3d_fullcam"))),
        schema_yaml=Path(str(paths.get("schema_yaml", track1_export.get("schema_yaml", "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml")))),
        subsets=list_value(config.get("selected_subsets", paths.get("selected_subsets", None))),
        scenes=list_value(config.get("selected_scenes", paths.get("selected_scenes", None))),
        class_names=list_value(config.get("class_names", None)),
        max_candidates_per_scene=config.get("max_candidates_per_scene"),
        global_config=dict(config.get("global_mtmc", config.get("global_config", {}))),
        final_export_options=final_export,
        track1_export_options=track1_export,
        compact_export=compact_export,
        enabled=bool(config.get("enabled", True)),
    )
    return spec


def materialize_runtime_configs(spec: GlobalTuningRunSpec, progress: bool = True) -> Dict[str, Path]:
    """Write per-run runtime configs consumed by existing scripts."""
    root = spec.runtime_config_root
    root.mkdir(parents=True, exist_ok=True)
    global_config_path = root / "global_mtmc.yaml"
    final_export_path = root / "final_export.yaml"
    track1_export_path = root / "track1_export.yaml"

    write_yaml({"global_mtmc": spec.global_config}, global_config_path)
    write_yaml({"final_export": build_final_export_config(spec, progress)}, final_export_path)
    write_yaml({"track1_export": build_track1_export_config(spec, progress)}, track1_export_path)
    return {
        "global_config": global_config_path,
        "final_export": final_export_path,
        "track1_export": track1_export_path,
    }


def build_final_export_config(spec: GlobalTuningRunSpec, progress: bool = True) -> Dict[str, Any]:
    """Build final export config for one tuning run."""
    config = {
        "root": spec.dataset_root,
        "local_tracks_root": str(spec.local_tracks_root),
        "global_mtmc_root": str(spec.global_association_root),
        "output_root": str(spec.final_export_root),
        "include_unassigned": True,
        "namespace_global_ids": True,
        "global_id_stride": 100000,
        "drop_invalid_bbox": True,
        "drop_unassigned_for_generic_export": True,
        "drop_invalid_bbox_for_generic_export": True,
        "progress": bool(progress),
        "subsets": DEFAULT_SUBSETS,
    }
    config.update(spec.final_export_options)
    config["global_mtmc_root"] = str(spec.global_association_root)
    config["output_root"] = str(spec.final_export_root)
    config["local_tracks_root"] = str(spec.local_tracks_root)
    return config


def build_track1_export_config(spec: GlobalTuningRunSpec, progress: bool = True, generic_root: Optional[Path] = None) -> Dict[str, Any]:
    """Build Track 1 export config for one tuning run."""
    if generic_root is None:
        generic_root = spec.final_export_root / "generic_tracking_export" / "test"
    config = {
        "generic_export_root": str(generic_root),
        "output_root": str(spec.track1_root),
        "schema_yaml": str(spec.schema_yaml),
        "force_unconfirmed_preview": False,
        "subsets": ["test"],
        "scenes": ["Warehouse_023", "Warehouse_024", "Warehouse_025"],
        "progress": bool(progress),
    }
    config.update(spec.track1_export_options)
    config["generic_export_root"] = str(generic_root)
    config["output_root"] = str(spec.track1_root)
    return config
