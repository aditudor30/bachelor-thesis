"""Wrappers around existing downstream export stages for tuning runs."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.final_export.track1_export_types import load_track1_schema_yaml
from deep_oc_sort_3d.final_export.track1_validator import validate_track1_export, write_track1_validation_report
from deep_oc_sort_3d.global_tuning.compact_export_policy import apply_compact_export_policy
from deep_oc_sort_3d.global_tuning.tuning_config import (
    build_track1_export_config,
    materialize_runtime_configs,
)
from deep_oc_sort_3d.global_tuning.tuning_io import write_json, write_yaml
from deep_oc_sort_3d.global_tuning.tuning_run_spec import GlobalTuningRunSpec
from deep_oc_sort_3d.scripts.export_track1_submission import export_track1_submission
from deep_oc_sort_3d.scripts.run_batch_global_mtmc_with_transitions import run_batch_global_mtmc_with_transitions
from deep_oc_sort_3d.scripts.run_final_mvp_export import run_final_mvp_export


class Namespace:
    """Simple argparse-like namespace."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def run_global_association_stage(
    spec: GlobalTuningRunSpec,
    runtime_configs: Dict[str, Path],
    overwrite: bool,
    progress: bool,
) -> None:
    """Run transition-enabled global association for one tuning spec."""
    args = Namespace(
        candidates_root=spec.input_motion_clean_root,
        output_root=spec.global_association_root,
        subsets=spec.subsets,
        scenes=spec.scenes,
        config=runtime_configs["global_config"],
        class_names=spec.class_names,
        max_candidates_per_scene=spec.max_candidates_per_scene,
        overwrite=bool(overwrite),
        progress=bool(progress),
    )
    run_batch_global_mtmc_with_transitions(args)


def run_final_export_stage(
    spec: GlobalTuningRunSpec,
    runtime_configs: Dict[str, Path],
    overwrite: bool,
    progress: bool,
) -> None:
    """Run final ID propagation and generic export for one tuning spec."""
    args = Namespace(
        config=runtime_configs["final_export"],
        subsets=spec.subsets,
        scenes=spec.scenes,
        camera_ids=None,
        overwrite=bool(overwrite),
        include_unassigned=None,
        namespace_global_ids=None,
        global_id_stride=None,
        drop_invalid_bbox=None,
        drop_unassigned_for_generic_export=None,
        drop_invalid_bbox_for_generic_export=None,
        progress=bool(progress),
    )
    run_final_mvp_export(args)


def maybe_run_compact_export_stage(spec: GlobalTuningRunSpec, progress: bool) -> Path:
    """Run compact export if enabled and return generic root for Track1."""
    if not bool(spec.compact_export.get("enabled", False)):
        return spec.final_export_root / "generic_tracking_export" / "test"
    compact_root = spec.final_export_root / "generic_tracking_export_compact"
    apply_compact_export_policy(
        generic_export_root=spec.final_export_root / "generic_tracking_export",
        output_root=compact_root,
        policy=spec.compact_export,
        show_progress=progress,
    )
    return compact_root / "test"


def run_track1_export_stage(
    spec: GlobalTuningRunSpec,
    generic_root: Path,
    progress: bool,
) -> Path:
    """Run Track1 export for one tuning spec and return submission path."""
    track1_config = build_track1_export_config(spec, progress=progress, generic_root=generic_root)
    runtime_path = spec.runtime_config_root / "track1_export_runtime.yaml"
    write_yaml({"track1_export": track1_config}, runtime_path)
    args = Namespace(
        config=runtime_path,
        generic_export_root=None,
        schema_yaml=None,
        schema_report=None,
        output_root=None,
        subsets=None,
        scenes=None,
        force_unconfirmed_preview=None,
        progress=bool(progress),
    )
    export_track1_submission(args)
    return spec.track1_root / "track1.txt"


def validate_track1_stage(spec: GlobalTuningRunSpec, submission_path: Path, progress: bool) -> Dict[str, Any]:
    """Validate Track1 output for one tuning spec."""
    schema = load_track1_schema_yaml(spec.schema_yaml)
    report = validate_track1_export(submission_path, schema, show_progress=progress)
    output = spec.track1_root / "track1_validation_report.json"
    write_track1_validation_report(report, output)
    write_json(report, spec.validation_root / "track1_validation_report.json")
    return report


def materialize_configs_for_run(spec: GlobalTuningRunSpec, progress: bool) -> Dict[str, Path]:
    """Write runtime configs and a copy of the resolved run spec."""
    configs = materialize_runtime_configs(spec, progress=progress)
    write_json(
        {
            "name": spec.name,
            "input_motion_clean_root": str(spec.input_motion_clean_root),
            "local_tracks_root": str(spec.local_tracks_root),
            "output_root": str(spec.output_root),
            "global_association_root": str(spec.global_association_root),
            "final_export_root": str(spec.final_export_root),
            "track1_root": str(spec.track1_root),
            "compact_export": spec.compact_export,
        },
        spec.runtime_config_root / "resolved_run_spec.json",
    )
    return configs

