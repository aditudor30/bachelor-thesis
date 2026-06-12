"""Export learned association mappings without altering the main pipeline."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.person_association.person_association_io import TrackKey
from deep_oc_sort_3d.person_association.person_merge_policy import mapping_rows
from deep_oc_sort_3d.person_association.person_merge_runner import (
    _compact_config,
    _run_track1_export,
    _validate_track1,
    _write_mapped_final_export,
)
from deep_oc_sort_3d.person_cleanup.person_export_policy import apply_person_cleanup_export_policy
from deep_oc_sort_3d.learned_association_application.scorer_association_io import write_csv_rows, write_json


def export_variant(
    variant_name: str,
    mapping: Dict[TrackKey, str],
    config: Dict[str, Any],
    run_root: Path,
    progress: bool,
) -> Dict[str, Any]:
    """Write mapped records, optional compact export, Track1 and validation."""
    paths = config.get("paths", {})
    source_final = Path(str(paths.get("v2_final_export_root")))
    schema_yaml = Path(str(paths.get("schema_yaml", "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml")))
    run_config = {
        "apply_to_subsets": ["internal_holdout", "official_val", "test"],
        "apply_to_scenes": None,
        "export_compact": {
            "enabled": variant_name == "mlp_combined_export_compact",
            "classification": {"short_rows_threshold": 5, "low_mean_confidence_threshold": 0.02, "low_max_confidence_threshold": 0.06},
            "pruning": {"enabled": True, "class_id": 0, "mode": "compact", "max_rows_per_track": 5, "mean_confidence_threshold": 0.02},
        },
    }
    write_csv_rows(run_root / "global_mapping.csv", mapping_rows(mapping))
    write_json(run_root / "global_mapping.json", {"|".join(key): value for key, value in mapping.items()})
    mapped_root = run_root / "merged_final_export" if run_config["export_compact"]["enabled"] else run_root / "final_export"
    export_summary = _write_mapped_final_export(source_final, mapped_root, mapping, run_config, progress)
    if run_config["export_compact"]["enabled"]:
        apply_person_cleanup_export_policy(
            source_final_export_root=mapped_root,
            output_final_export_root=run_root / "final_export",
            config=_compact_config(run_config),
            show_progress=progress,
        )
    track1_path = _run_track1_export(run_root, schema_yaml, progress)
    validation = _validate_track1(track1_path, schema_yaml, run_root, progress)
    return {"export": export_summary, "track1_path": str(track1_path), "validation": validation}
