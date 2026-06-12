"""Run isolated Step 21C variant stages using the existing Step 21B pipeline."""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_pipeline_config import prepare_generated_stage_configs
from deep_oc_sort_3d.bytetrack_tuning.tuning_config import (
    build_variant_pipeline_config,
    variant_root,
)
from deep_oc_sort_3d.bytetrack_tuning.tuning_io import write_json, write_yaml


FULL_STAGE_ORDER = [
    "local_tracking",
    "tracklets",
    "candidates",
    "motion_filtering",
    "global_association",
    "final_export",
    "track1_export",
    "validation",
]


def run_tuning_variant(
    config: Dict[str, Any],
    variant_name: str,
    full_pipeline: bool,
    progress: bool,
    overwrite: bool,
    skip_existing: bool,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run local-only Phase A or full Phase B for one variant."""
    resolved = build_variant_pipeline_config(config, variant_name, include_full_test=full_pipeline)
    root = variant_root(config, variant_name)
    variant_config_path = root / "variant_config.yaml"
    write_yaml(variant_config_path, resolved)
    generated = prepare_generated_stage_configs(resolved)
    stages = FULL_STAGE_ORDER if full_pipeline else ["local_tracking"]
    rows = []
    run_started = time.time()
    for stage in stages:
        if skip_existing and not overwrite and _stage_complete(root, stage, full_pipeline):
            rows.append({"stage": stage, "status": "skipped_existing", "command": ""})
            continue
        command = _stage_command(stage, variant_config_path, generated, root, progress, overwrite)
        print("[%s/%s] %s" % (variant_name, stage, " ".join(command)))
        if dry_run:
            rows.append({"stage": stage, "status": "dry_run", "command": " ".join(command)})
            continue
        stage_started = time.time()
        try:
            subprocess.check_call(command)
            rows.append(
                {
                    "stage": stage,
                    "status": "ok",
                    "runtime_seconds": time.time() - stage_started,
                    "command": " ".join(command),
                }
            )
        except subprocess.CalledProcessError as exc:
            rows.append(
                {
                    "stage": stage,
                    "status": "error",
                    "returncode": exc.returncode,
                    "command": " ".join(command),
                }
            )
            break
    if full_pipeline and not dry_run:
        _mirror_validation(root)
    summary = {
        "variant": variant_name,
        "phase": "full" if full_pipeline else "local_sweep",
        "status": "error" if any(row.get("status") == "error" for row in rows) else "ok",
        "person_specific_requested": resolved.get("person_specific_requested"),
        "person_specific_applied": False,
        "runtime_seconds": time.time() - run_started,
        "stages": rows,
    }
    write_json(root / "summaries" / ("full_run.json" if full_pipeline else "local_sweep_run.json"), summary)
    return summary


def _stage_command(
    stage: str,
    variant_config_path: Path,
    generated: Dict[str, Path],
    root: Path,
    progress: bool,
    overwrite: bool,
) -> List[str]:
    modules = {
        "local_tracking": "deep_oc_sort_3d.scripts.run_baseline_v2_bytetrack_local_stage",
        "tracklets": "deep_oc_sort_3d.scripts.run_baseline_v2_tracklets",
        "candidates": "deep_oc_sort_3d.scripts.run_baseline_v2_candidates",
        "motion_filtering": "deep_oc_sort_3d.scripts.run_baseline_v2_motion_filtering",
        "global_association": "deep_oc_sort_3d.scripts.run_baseline_v2_global_association",
        "final_export": "deep_oc_sort_3d.scripts.run_baseline_v2_final_export",
        "track1_export": "deep_oc_sort_3d.scripts.export_track1_submission",
        "validation": "deep_oc_sort_3d.scripts.validate_baseline_v2_bytetrack_local_track1",
    }
    command = [sys.executable, "-m", modules[stage]]
    if stage == "local_tracking":
        command.extend(["--config", str(variant_config_path), "--stage", "local_tracking"])
    elif stage == "validation":
        command.extend(
            [
                "--track1",
                str(root / "track1_submission" / "track1.txt"),
                "--output-root",
                str(root / "validation"),
            ]
        )
    else:
        command.extend(["--config", str(generated[stage])])
    command.append("--progress" if progress else "--no-progress")
    if overwrite and stage != "track1_export":
        command.append("--overwrite")
    return command


def _stage_complete(root: Path, stage: str, full_pipeline: bool) -> bool:
    markers = {
        "local_tracking": root / "local_tracks" / "summaries" / "local_tracking_summary.json",
        "tracklets": root / "tracklets" / "summaries" / "tracklet_summary.json",
        "candidates": root / "candidates" / "summaries" / "candidate_summary.json",
        "motion_filtering": root / "motion_clean" / "summaries" / "motion_quality_summary.json",
        "global_association": root / "global_mtmc" / "summaries" / "global_transition_summary.json",
        "final_export": root / "final_export" / "generic_tracking_export",
        "track1_export": root / "track1_submission" / "track1.txt",
        "validation": root / "validation" / "track1_validation_summary.json",
    }
    marker = markers.get(stage)
    if marker is None or not marker.exists():
        return False
    if stage == "local_tracking" and full_pipeline:
        return (root / "local_tracks" / "test" / "Warehouse_023").exists()
    return True


def _mirror_validation(root: Path) -> None:
    """Keep validation available both at the requested root and beside Track1."""
    source = root / "validation"
    target = root / "track1_submission" / "validation"
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.json"):
        shutil.copy2(str(path), str(target / path.name))
