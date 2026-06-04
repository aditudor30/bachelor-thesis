"""Command runner/runbook for the experimental baseline_v2 pseudo-3D pipeline."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


STAGE_MODULES = [
    ("observations", "deep_oc_sort_3d.scripts.build_pseudo3d_observations", "observations_config"),
    ("local_tracking", "deep_oc_sort_3d.scripts.run_baseline_v2_local_tracking", "local_tracking_config"),
    ("tracklets", "deep_oc_sort_3d.scripts.run_baseline_v2_tracklets", "tracklets_config"),
    ("candidates", "deep_oc_sort_3d.scripts.run_baseline_v2_candidates", "candidates_config"),
    ("motion_filtering", "deep_oc_sort_3d.scripts.run_baseline_v2_motion_filtering", "motion_filtering_config"),
    ("global_association", "deep_oc_sort_3d.scripts.run_baseline_v2_global_association", "global_association_config"),
    ("final_export", "deep_oc_sort_3d.scripts.run_baseline_v2_final_export", "final_export_config"),
    ("track1_export", "deep_oc_sort_3d.scripts.export_track1_submission", "track1_export_config"),
]


def load_baseline_v2_config(path: Path) -> Dict[str, Any]:
    """Load a baseline_v2 YAML config."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_baseline_v2_commands(config: Dict[str, Any], overwrite: bool, progress: bool) -> List[Dict[str, Any]]:
    """Build command specs for all configured baseline_v2 stages."""
    section = config.get("baseline_v2_pipeline", config)
    commands = []
    selected = section.get("stages")
    selected_set = None if selected is None else set([str(item) for item in selected])
    for stage, module, config_key in STAGE_MODULES:
        if selected_set is not None and stage not in selected_set:
            continue
        config_path = section.get(config_key)
        if not config_path:
            commands.append({"stage": stage, "status": "missing_config", "command": []})
            continue
        command = [sys.executable, "-m", module, "--config", str(config_path)]
        if progress:
            command.append("--progress")
        else:
            command.append("--no-progress")
        if overwrite and stage not in ("track1_export",):
            command.append("--overwrite")
        commands.append({"stage": stage, "status": "ready", "command": command})
    validation = section.get("track1_validation")
    if isinstance(validation, dict) and validation.get("submission"):
        command = [
            sys.executable,
            "-m",
            "deep_oc_sort_3d.scripts.validate_track1_submission",
            "--submission",
            str(validation.get("submission")),
            "--schema-yaml",
            str(validation.get("schema_yaml", "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml")),
            "--output",
            str(validation.get("output", "output/track1_submission/baseline_v2_pseudo3d/track1_validation.json")),
        ]
        command.append("--progress" if progress else "--no-progress")
        commands.append({"stage": "validation", "status": "ready", "command": command})
    comparison_config = section.get("comparison_config")
    if comparison_config:
        command = [sys.executable, "-m", "deep_oc_sort_3d.scripts.compare_baseline_v1_v2", "--config", str(comparison_config)]
        command.append("--progress" if progress else "--no-progress")
        commands.append({"stage": "comparison", "status": "ready", "command": command})
    return commands


def run_baseline_v2_pipeline(config_path: Path, overwrite: bool, progress: bool, dry_run: bool = False) -> Dict[str, Any]:
    """Run or print the full baseline_v2 command sequence."""
    config = load_baseline_v2_config(config_path)
    commands = build_baseline_v2_commands(config, overwrite=overwrite, progress=progress)
    rows = []
    for item in commands:
        command = item.get("command", [])
        if item.get("status") != "ready":
            rows.append({"stage": item.get("stage"), "status": item.get("status"), "command": ""})
            continue
        printable = " ".join(command)
        print(printable)
        if dry_run:
            rows.append({"stage": item.get("stage"), "status": "dry_run", "command": printable})
            continue
        subprocess.check_call(command)
        rows.append({"stage": item.get("stage"), "status": "ok", "command": printable})
    summary = {"config": str(config_path), "dry_run": dry_run, "stages": rows}
    output_root = Path(config.get("baseline_v2_pipeline", config).get("runbook_output_root", "output/pipeline_runs/baseline_v2_pseudo3d/summaries"))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "baseline_v2_runbook.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

