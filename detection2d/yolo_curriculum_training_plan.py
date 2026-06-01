"""Build manual command plans for YOLO curriculum training and evaluation."""

import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_curriculum_plan(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML plan for curriculum training."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {}


def build_curriculum_commands(plan: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build command records from a curriculum plan without executing them."""
    commands = []
    root = str(plan.get("dataset", {}).get("root", "/path/to/MTMC_Tracking_2026"))
    stages = list(plan.get("training", {}).get("stages", []))
    eval_stage_names = set(plan.get("evaluation", {}).get("evaluate_stages", []))
    for stage in stages:
        stage_name = str(stage.get("name"))
        commands.append(
            {
                "phase": "train",
                "name": stage_name,
                "command": shell_command(build_train_args(stage)),
            }
        )
        if stage_name in eval_stage_names:
            commands.extend(build_stage_eval_commands(plan, root, stage))
    commands.extend(build_compare_commands(plan))
    return commands


def build_train_args(stage: Dict[str, Any]) -> List[str]:
    """Build argv for one YOLO training stage."""
    args = [
        "python",
        "-m",
        "deep_oc_sort_3d.scripts.train_yolo",
        "--data",
        str(stage["data"]),
        "--model",
        str(stage["model"]),
        "--epochs",
        str(stage["epochs"]),
        "--imgsz",
        str(stage["imgsz"]),
        "--batch",
        str(stage["batch"]),
        "--device",
        str(stage["device"]),
        "--workers",
        str(stage["workers"]),
        "--project",
        str(stage["project"]),
        "--name",
        str(stage["name"]),
    ]
    if stage.get("patience") is not None:
        args.extend(["--patience", str(stage["patience"])])
    if bool(stage.get("resume", False)):
        args.append("--resume")
    return args


def build_stage_eval_commands(plan: Dict[str, Any], root: str, stage: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build inference/evaluation/sweep commands for one trained stage."""
    commands = []
    stage_name = str(stage["name"])
    checkpoint = str(stage.get("checkpoint", _checkpoint_path(stage)))
    eval_config = plan.get("evaluation", {})
    for eval_name in ["official_val", "internal_holdout"]:
        split_config = eval_config.get(eval_name)
        if not isinstance(split_config, dict):
            continue
        commands.extend(build_inference_commands(root, stage_name, checkpoint, eval_name, split_config, eval_config))
        commands.append(build_eval_command(root, stage_name, eval_name, split_config, eval_config))
        commands.append(build_threshold_sweep_command(root, stage_name, eval_name, split_config, eval_config))
    return commands


def build_inference_commands(
    root: str,
    stage_name: str,
    checkpoint: str,
    eval_name: str,
    split_config: Dict[str, Any],
    eval_config: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build one inference command per evaluation scene."""
    commands = []
    split = str(split_config.get("split"))
    scenes = list(split_config.get("scenes", []))
    detections_dir = _format_path(str(split_config.get("detections_dir")), stage_name)
    for scene in scenes:
        output = str(Path(detections_dir) / ("%s.csv" % scene))
        args = [
            "python",
            "-m",
            "deep_oc_sort_3d.scripts.run_yolo_inference",
            "--root",
            root,
            "--split",
            split,
            "--scene",
            str(scene),
            "--model",
            checkpoint,
            "--output",
            output,
            "--conf",
            str(eval_config.get("inference_conf", 0.01)),
            "--frame-stride",
            str(eval_config.get("frame_stride", 1)),
            "--imgsz",
            str(eval_config.get("imgsz", 1280)),
        ]
        if eval_config.get("max_frames_per_scene") is not None:
            args.extend(["--max-frames", str(eval_config["max_frames_per_scene"])])
        camera_id = str(split_config.get("camera_id", "all"))
        if camera_id.lower() != "all":
            args.extend(["--camera-id", camera_id])
        commands.append(
            {
                "phase": "inference",
                "name": "%s_%s_%s" % (stage_name, eval_name, scene),
                "command": shell_command(args),
            }
        )
    return commands


def build_eval_command(
    root: str,
    stage_name: str,
    eval_name: str,
    split_config: Dict[str, Any],
    eval_config: Dict[str, Any],
) -> Dict[str, str]:
    """Build an official-val or internal-holdout evaluation command."""
    script = "deep_oc_sort_3d.scripts.evaluate_yolo_official_val"
    if str(split_config.get("split")) == "train":
        script = "deep_oc_sort_3d.scripts.evaluate_yolo_internal_holdout"
    args = [
        "python",
        "-m",
        script,
        "--root",
        root,
        "--detections-dir",
        _format_path(str(split_config.get("detections_dir")), stage_name),
        "--scenes",
    ]
    args.extend([str(scene) for scene in split_config.get("scenes", [])])
    args.extend(
        [
            "--camera-id",
            str(split_config.get("camera_id", "all")),
            "--iou-threshold",
            str(eval_config.get("iou_threshold", 0.3)),
            "--conf-threshold",
            str(eval_config.get("eval_conf", 0.05)),
            "--frame-stride",
            str(eval_config.get("frame_stride", 1)),
            "--output",
            _format_path(str(split_config.get("metrics_output")), stage_name),
        ]
    )
    if eval_config.get("max_frames_per_scene") is not None:
        args.extend(["--max-frames-per-scene", str(eval_config["max_frames_per_scene"])])
    return {
        "phase": "eval",
        "name": "%s_%s" % (stage_name, eval_name),
        "command": shell_command(args),
    }


def build_threshold_sweep_command(
    root: str,
    stage_name: str,
    eval_name: str,
    split_config: Dict[str, Any],
    eval_config: Dict[str, Any],
) -> Dict[str, str]:
    """Build a split-wide confidence-threshold sweep command."""
    args = [
        "python",
        "-m",
        "deep_oc_sort_3d.scripts.compare_yolo_split_conf_thresholds",
        "--root",
        root,
        "--split",
        str(split_config.get("split")),
        "--detections-dir",
        _format_path(str(split_config.get("detections_dir")), stage_name),
        "--scenes",
    ]
    args.extend([str(scene) for scene in split_config.get("scenes", [])])
    args.extend(
        [
            "--camera-id",
            str(split_config.get("camera_id", "all")),
            "--thresholds",
        ]
    )
    args.extend([str(threshold) for threshold in eval_config.get("thresholds", [])])
    args.extend(
        [
            "--iou-threshold",
            str(eval_config.get("iou_threshold", 0.3)),
            "--frame-stride",
            str(eval_config.get("frame_stride", 1)),
            "--min-precision",
            str(eval_config.get("min_precision", 0.5)),
            "--output",
            _format_path(str(split_config.get("threshold_output")), stage_name),
        ]
    )
    if eval_config.get("max_frames_per_scene") is not None:
        args.extend(["--max-frames-per-scene", str(eval_config["max_frames_per_scene"])])
    return {
        "phase": "threshold_sweep",
        "name": "%s_%s" % (stage_name, eval_name),
        "command": shell_command(args),
    }


def build_compare_commands(plan: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build metric comparison commands for evaluated stages."""
    compare_config = plan.get("comparison", {})
    if not isinstance(compare_config, dict) or not compare_config.get("enabled", True):
        return []
    commands = []
    for item in compare_config.get("metrics", []):
        args = [
            "python",
            "-m",
            "deep_oc_sort_3d.scripts.compare_yolo_models",
            "--metrics",
        ]
        args.extend([str(path) for path in item.get("paths", [])])
        args.extend(["--names"])
        args.extend([str(name) for name in item.get("names", [])])
        args.extend(["--output", str(item.get("output"))])
        commands.append(
            {
                "phase": "compare",
                "name": str(item.get("name", "compare")),
                "command": shell_command(args),
            }
        )
    return commands


def render_commands_markdown(commands: List[Dict[str, str]]) -> str:
    """Render command records as Markdown."""
    lines = ["# YOLO Curriculum Step 9C Commands", ""]
    current_phase = None
    for command in commands:
        phase = str(command["phase"])
        if phase != current_phase:
            lines.append("## %s" % phase.replace("_", " ").title())
            lines.append("")
            current_phase = phase
        lines.append("### %s" % command["name"])
        lines.append("")
        lines.append("```bash")
        lines.append(command["command"])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def shell_command(args: List[str]) -> str:
    """Return a shell-safe command string."""
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _checkpoint_path(stage: Dict[str, Any]) -> str:
    return str(Path(str(stage["project"])) / str(stage["name"]) / "weights" / "best.pt")


def _format_path(template: str, stage_name: str) -> str:
    return template.replace("{stage}", stage_name)
