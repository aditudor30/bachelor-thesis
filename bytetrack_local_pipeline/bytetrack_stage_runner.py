"""Stage orchestration for the isolated Step 21B pipeline."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_io import write_json
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_pipeline_config import (
    load_bytetrack_pipeline_config,
    prepare_generated_stage_configs,
    selected_stages,
)
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_precheck import precheck_allows_full_rerun


def build_stage_commands(
    config_path: Path,
    stages: Optional[List[str]] = None,
    progress: bool = True,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    """Build commands while reusing existing V2 stage implementations."""
    config = load_bytetrack_pipeline_config(config_path)
    generated = prepare_generated_stage_configs(config)
    stage_list = selected_stages(None) if stages is None else list(stages)
    modules = {
        "precheck": "deep_oc_sort_3d.scripts.run_baseline_v2_bytetrack_local_precheck",
        "local_tracking": "deep_oc_sort_3d.scripts.run_baseline_v2_bytetrack_local_stage",
        "tracklets": "deep_oc_sort_3d.scripts.run_baseline_v2_tracklets",
        "candidates": "deep_oc_sort_3d.scripts.run_baseline_v2_candidates",
        "motion_filtering": "deep_oc_sort_3d.scripts.run_baseline_v2_motion_filtering",
        "global_association": "deep_oc_sort_3d.scripts.run_baseline_v2_global_association",
        "final_export": "deep_oc_sort_3d.scripts.run_baseline_v2_final_export",
        "track1_export": "deep_oc_sort_3d.scripts.export_track1_submission",
        "validation": "deep_oc_sort_3d.scripts.validate_baseline_v2_bytetrack_local_track1",
        "package": "deep_oc_sort_3d.scripts.package_baseline_v2_bytetrack_local_submission",
        "comparison": "deep_oc_sort_3d.scripts.compare_baseline_v2_bytetrack_local",
    }
    generated_stage_names = set(generated.keys())
    output = []
    for stage in stage_list:
        command = [sys.executable, "-m", modules[stage]]
        if stage in generated_stage_names:
            command.extend(["--config", str(generated[stage])])
        else:
            command.extend(["--config", str(config_path)])
        if stage == "local_tracking":
            command.extend(["--stage", "local_tracking"])
        command.append("--progress" if progress else "--no-progress")
        if overwrite and stage != "track1_export":
            command.append("--overwrite")
        output.append({"stage": stage, "command": command})
    return output


def run_bytetrack_stages(
    config_path: Path,
    stages: Optional[List[str]] = None,
    progress: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run selected stages and stop before destructive downstream work on failed precheck."""
    config = load_bytetrack_pipeline_config(config_path)
    commands = build_stage_commands(config_path, stages=stages, progress=progress, overwrite=overwrite)
    rows = []
    for item in commands:
        stage = str(item["stage"])
        if not dry_run and stage != "precheck" and _precheck_gate_required(config) and not _precheck_passed(config):
            row = {"stage": stage, "status": "blocked_precheck", "command": " ".join(item["command"])}
            rows.append(row)
            print("[%s] blocked: precheck has not passed" % stage)
            break
        printable = " ".join(item["command"])
        print("[%s] %s" % (stage, printable))
        if dry_run:
            rows.append({"stage": stage, "status": "dry_run", "command": printable})
            continue
        subprocess.check_call(item["command"])
        rows.append({"stage": stage, "status": "ok", "command": printable})
    summary = {"config": str(config_path), "stages": rows, "dry_run": dry_run, "overwrite": overwrite}
    comparison_root = Path(str(config.get("paths", {}).get("output_comparison_root")))
    write_json(comparison_root / "pipeline_runbook.json", summary)
    return summary


def _precheck_gate_required(config: Dict[str, Any]) -> bool:
    return bool(config.get("precheck", {}).get("require_pass_before_full_rerun", True))


def _precheck_passed(config: Dict[str, Any]) -> bool:
    root = Path(str(config.get("paths", {}).get("output_precheck_root")))
    path = root / "precheck_verdict.json"
    if not path.exists():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return precheck_allows_full_rerun(value if isinstance(value, dict) else {})
