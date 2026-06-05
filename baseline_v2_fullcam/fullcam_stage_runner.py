"""Stage runner for baseline_v2_pseudo3d_fullcam."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


STAGE_ORDER = [
    "observations",
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


STAGE_MODULES = {
    "observations": "deep_oc_sort_3d.scripts.build_fullcam_pseudo3d_observations",
    "local_tracking": "deep_oc_sort_3d.scripts.run_baseline_v2_local_tracking",
    "tracklets": "deep_oc_sort_3d.scripts.run_baseline_v2_tracklets",
    "candidates": "deep_oc_sort_3d.scripts.run_baseline_v2_candidates",
    "motion_filtering": "deep_oc_sort_3d.scripts.run_baseline_v2_motion_filtering",
    "global_association": "deep_oc_sort_3d.scripts.run_baseline_v2_global_association",
    "final_export": "deep_oc_sort_3d.scripts.run_baseline_v2_final_export",
    "track1_export": "deep_oc_sort_3d.scripts.export_track1_submission",
    "validation": "deep_oc_sort_3d.scripts.validate_baseline_v2_fullcam_track1",
    "package": "deep_oc_sort_3d.scripts.package_baseline_v2_fullcam_submission",
    "comparison": "deep_oc_sort_3d.scripts.compare_baseline_v1_v2_fullcam",
}


DEFAULT_STAGE_CONFIGS = {
    "observations": "deep_oc_sort_3d/configs/baseline_v2_fullcam_observations.yaml",
    "local_tracking": "deep_oc_sort_3d/configs/baseline_v2_fullcam_local_tracking.yaml",
    "tracklets": "deep_oc_sort_3d/configs/baseline_v2_fullcam_tracklets.yaml",
    "candidates": "deep_oc_sort_3d/configs/baseline_v2_fullcam_candidates.yaml",
    "motion_filtering": "deep_oc_sort_3d/configs/baseline_v2_fullcam_motion_filtering.yaml",
    "global_association": "deep_oc_sort_3d/configs/baseline_v2_fullcam_global_association.yaml",
    "final_export": "deep_oc_sort_3d/configs/baseline_v2_fullcam_final_export.yaml",
    "track1_export": "deep_oc_sort_3d/configs/baseline_v2_fullcam_track1_export.yaml",
}


def load_pipeline_config(path: Path) -> Dict[str, Any]:
    """Load a fullcam pipeline YAML config."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def selected_stages(stage: Optional[str]) -> List[str]:
    """Return stage list for a single stage or the full pipeline."""
    if stage in (None, "", "all"):
        return list(STAGE_ORDER)
    if stage not in STAGE_ORDER:
        raise ValueError("Unknown stage %s. Valid stages: %s" % (stage, ", ".join(STAGE_ORDER)))
    return [str(stage)]


def build_stage_commands(
    config: Dict[str, Any],
    stages: Optional[List[str]] = None,
    overwrite: bool = False,
    progress: bool = True,
    skip_existing: bool = False,
) -> List[Dict[str, Any]]:
    """Build command specs for selected fullcam stages."""
    stage_list = list(STAGE_ORDER) if stages is None else stages
    commands = []
    stage_configs = _stage_config_paths(config)
    for stage in stage_list:
        module = STAGE_MODULES[stage]
        command = [sys.executable, "-m", module]
        if stage in DEFAULT_STAGE_CONFIGS:
            command.extend(["--config", str(stage_configs[stage])])
        else:
            command.extend(["--config", str(_full_pipeline_config_path(config))])
        command.append("--progress" if progress else "--no-progress")
        if overwrite and not skip_existing:
            command.append("--overwrite")
        commands.append({"stage": stage, "command": command, "status": "ready"})
    return commands


def run_fullcam_stages(
    config_path: Path,
    stages: Optional[List[str]] = None,
    overwrite: bool = False,
    progress: bool = True,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Run or dry-run selected fullcam stages."""
    config = load_pipeline_config(config_path)
    config["_config_path"] = str(config_path)
    stage_list = list(STAGE_ORDER) if stages is None else stages
    commands = build_stage_commands(config, stage_list, overwrite=overwrite, progress=progress, skip_existing=skip_existing)
    rows = []
    for item in commands:
        command = item["command"]
        printable = " ".join(command)
        print("[%s] %s" % (item["stage"], printable))
        if dry_run:
            rows.append({"stage": item["stage"], "status": "dry_run", "command": printable})
            continue
        subprocess.check_call(command)
        rows.append({"stage": item["stage"], "status": "ok", "command": printable})
    summary = {
        "config": str(config_path),
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "skip_existing": bool(skip_existing),
        "stages": rows,
    }
    _write_runbook(summary, config)
    return summary


def _stage_config_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    stage_section = config.get("stage_configs", {})
    output = {}
    for stage, default_path in DEFAULT_STAGE_CONFIGS.items():
        value = stage_section.get(stage, default_path) if isinstance(stage_section, dict) else default_path
        output[stage] = Path(str(value))
    return output


def _full_pipeline_config_path(config: Dict[str, Any]) -> Path:
    value = config.get("_config_path")
    if value:
        return Path(str(value))
    return Path("deep_oc_sort_3d/configs/baseline_v2_fullcam_full_pipeline.yaml")


def _write_runbook(summary: Dict[str, Any], config: Dict[str, Any]) -> None:
    paths = config.get("paths", {})
    output_root = Path(paths.get("output_pipeline_root", "output/pipeline_runs/baseline_v2_pseudo3d_fullcam")) / "summaries"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "baseline_v2_fullcam_runbook.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
