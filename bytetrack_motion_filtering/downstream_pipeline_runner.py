"""Run unchanged downstream stages for one isolated Step 21E variant."""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_config import source_local_tracks_root, subset_entries, variant_root
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_io import write_json, write_yaml


STAGES = ["global_association", "final_export", "track1_export", "validation"]


def run_downstream_pipeline(
    config: Dict[str, Any],
    variant_name: str,
    progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Run global association, final export, Track1 export and validation."""
    root = variant_root(config, variant_name)
    generated = prepare_downstream_configs(config, variant_name)
    rows = []
    started = time.time()
    for stage in STAGES:
        marker = _marker(root, stage)
        if skip_existing and not overwrite and marker.exists():
            rows.append({"stage": stage, "status": "skipped_existing", "command": ""})
            continue
        command = _command(stage, generated, root, progress, overwrite)
        print("[%s/%s] %s" % (variant_name, stage, " ".join(command)))
        stage_started = time.time()
        try:
            subprocess.check_call(command)
            rows.append({"stage": stage, "status": "ok", "runtime_seconds": time.time() - stage_started, "command": " ".join(command)})
        except subprocess.CalledProcessError as exc:
            rows.append({"stage": stage, "status": "error", "returncode": exc.returncode, "command": " ".join(command)})
            break
    _mirror_validation(root)
    summary = {
        "variant_name": variant_name,
        "status": "error" if any(row.get("status") == "error" for row in rows) else "ok",
        "runtime_seconds": time.time() - started,
        "stages": rows,
    }
    write_json(root / "summaries" / "downstream_run.json", summary)
    return summary


def prepare_downstream_configs(config: Dict[str, Any], variant_name: str) -> Dict[str, Path]:
    """Generate configs consumed by the existing downstream CLIs."""
    root = variant_root(config, variant_name)
    generated_root = root / "summaries" / "generated_configs"
    entries = subset_entries(config, include_test=True)
    grouped = {}
    for subset, split, scene_name in entries:
        grouped.setdefault(subset, {"split": split, "scenes": []})
        grouped[subset]["scenes"].append(scene_name)
    pipeline = config.get("pipeline", {})
    values = {
        "global_association": {
            "global_association": {
                "candidates_root": str(root / "motion_clean"),
                "output_root": str(root / "global_mtmc"),
                "global_config": pipeline.get("global_config", "deep_oc_sort_3d/configs/global_mtmc_transition_medium_conf001.yaml"),
                "subsets": list(grouped.keys()),
                "scenes": None,
                "class_names": None,
                "max_candidates_per_scene": None,
                "progress": True,
            }
        },
        "final_export": {
            "final_export": {
                "root": config.get("paths", {}).get("dataset_root"),
                "local_tracks_root": str(source_local_tracks_root(config)),
                "global_mtmc_root": str(root / "global_mtmc"),
                "output_root": str(root / "final_export"),
                "include_unassigned": True,
                "namespace_global_ids": True,
                "global_id_stride": 100000,
                "drop_invalid_bbox": True,
                "drop_unassigned_for_generic_export": True,
                "drop_invalid_bbox_for_generic_export": True,
                "progress": True,
                "subsets": grouped,
            }
        },
        "track1_export": {
            "track1_export": {
                "generic_export_root": str(root / "final_export" / "generic_tracking_export" / "test"),
                "output_root": str(root / "track1_submission"),
                "schema_confirmed": True,
                "schema_yaml": pipeline.get("track1_schema", "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml"),
                "force_unconfirmed_preview": False,
                "subsets": ["test"],
                "scenes": [scene for subset, split, scene in entries if split == "test"],
                "progress": True,
            }
        },
    }
    output = {}
    for name, payload in values.items():
        path = generated_root / (name + ".yaml")
        write_yaml(path, payload)
        output[name] = path
    return output


def _command(stage: str, generated: Dict[str, Path], root: Path, progress: bool, overwrite: bool) -> List[str]:
    modules = {
        "global_association": "deep_oc_sort_3d.scripts.run_baseline_v2_global_association",
        "final_export": "deep_oc_sort_3d.scripts.run_baseline_v2_final_export",
        "track1_export": "deep_oc_sort_3d.scripts.export_track1_submission",
        "validation": "deep_oc_sort_3d.scripts.validate_baseline_v2_bytetrack_local_track1",
    }
    command = [sys.executable, "-m", modules[stage]]
    if stage == "validation":
        command.extend(["--track1", str(root / "track1_submission" / "track1.txt"), "--output-root", str(root / "validation")])
    else:
        command.extend(["--config", str(generated[stage])])
    command.append("--progress" if progress else "--no-progress")
    if overwrite and stage != "track1_export":
        command.append("--overwrite")
    return command


def _marker(root: Path, stage: str) -> Path:
    markers = {
        "global_association": root / "global_mtmc" / "summaries" / "global_transition_summary.json",
        "final_export": root / "final_export" / "generic_tracking_export",
        "track1_export": root / "track1_submission" / "track1.txt",
        "validation": root / "validation" / "track1_validation_summary.json",
    }
    return markers[stage]


def _mirror_validation(root: Path) -> None:
    source = root / "validation"
    target = root / "track1_submission" / "validation"
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.json"):
        shutil.copy2(str(path), str(target / path.name))

