"""Configuration dataclasses and YAML I/O for pipeline runs."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


DEFAULT_SPLIT_BY_SUBSET = {
    "official_val": "val",
    "internal_holdout": "train",
    "test": "test",
}


@dataclass
class PipelineRunConfig:
    """Resolved configuration for a batch pipeline run."""

    root: Path
    output_root: Path
    run_name: str
    detector_model: Path
    conf_threshold: float
    imgsz: int
    device: str
    frame_stride: int
    max_frames: Optional[int]
    scenes_by_subset: Dict[str, List[str]]
    camera_ids: Optional[List[str]]
    split_by_subset: Dict[str, str]
    build_observations: bool
    export_mot_like: bool
    iou_threshold: float
    depth_sampling_method: str
    class_must_match: bool


def load_pipeline_config(path: Union[str, Path]) -> PipelineRunConfig:
    """Load a pipeline config YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Pipeline config must contain a mapping.")
    pipeline = data.get("pipeline", {})
    subsets = data.get("subsets", {})
    if not isinstance(pipeline, dict):
        raise ValueError("Config field 'pipeline' must be a mapping.")
    if not isinstance(subsets, dict):
        raise ValueError("Config field 'subsets' must be a mapping.")

    scenes_by_subset = {}
    split_by_subset = dict(DEFAULT_SPLIT_BY_SUBSET)
    for subset_name, subset_data in subsets.items():
        if not isinstance(subset_data, dict):
            continue
        scenes_by_subset[str(subset_name)] = [str(scene) for scene in subset_data.get("scenes", [])]
        if subset_data.get("split") is not None:
            split_by_subset[str(subset_name)] = str(subset_data.get("split"))

    return PipelineRunConfig(
        root=Path(_required(pipeline, "root")),
        output_root=Path(_required(pipeline, "output_root")),
        run_name=str(_required(pipeline, "run_name")),
        detector_model=Path(_required(pipeline, "detector_model")),
        conf_threshold=float(pipeline.get("conf_threshold", 0.01)),
        imgsz=int(pipeline.get("imgsz", 1280)),
        device=str(pipeline.get("device", "0")),
        frame_stride=int(pipeline.get("frame_stride", 1)),
        max_frames=_optional_int(pipeline.get("max_frames")),
        scenes_by_subset=scenes_by_subset,
        camera_ids=_optional_str_list(pipeline.get("camera_ids")),
        split_by_subset=split_by_subset,
        build_observations=bool(pipeline.get("build_observations", True)),
        export_mot_like=bool(pipeline.get("export_mot_like", True)),
        iou_threshold=float(pipeline.get("iou_threshold", 0.3)),
        depth_sampling_method=str(pipeline.get("depth_sampling_method", "center_median")),
        class_must_match=bool(pipeline.get("class_must_match", True)),
    )


def save_resolved_config(config: PipelineRunConfig, output_path: Path) -> None:
    """Save a resolved config as YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False), encoding="utf-8")


def validate_pipeline_config(config: PipelineRunConfig) -> List[str]:
    """Return a list of human-readable config warnings/errors."""
    messages = []
    if not str(config.run_name):
        messages.append("run_name is empty")
    if not config.scenes_by_subset:
        messages.append("no subsets/scenes configured")
    for subset_name, scenes in config.scenes_by_subset.items():
        if subset_name not in config.split_by_subset:
            messages.append("subset %s has no split mapping" % subset_name)
        if not scenes:
            messages.append("subset %s has no scenes" % subset_name)
    if config.frame_stride <= 0:
        messages.append("frame_stride must be positive")
    if config.imgsz <= 0:
        messages.append("imgsz must be positive")
    if config.conf_threshold < 0.0:
        messages.append("conf_threshold must be non-negative")
    if config.max_frames is not None and config.max_frames <= 0:
        messages.append("max_frames must be positive when provided")
    return messages


def config_to_dict(config: PipelineRunConfig) -> Dict[str, Any]:
    """Convert a config to a YAML/JSON-friendly dictionary."""
    data = asdict(config)
    for key in ["root", "output_root", "detector_model"]:
        data[key] = str(data[key])
    subsets = {}
    for subset_name, scenes in config.scenes_by_subset.items():
        subsets[subset_name] = {
            "split": config.split_by_subset.get(subset_name, DEFAULT_SPLIT_BY_SUBSET.get(subset_name, subset_name)),
            "scenes": list(scenes),
        }
    pipeline = dict(data)
    pipeline.pop("scenes_by_subset")
    pipeline.pop("split_by_subset")
    return {"pipeline": pipeline, "subsets": subsets}


def update_config(config: PipelineRunConfig, **kwargs: Any) -> PipelineRunConfig:
    """Return a copy of config with simple CLI overrides applied."""
    clean = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in ("root", "output_root", "detector_model"):
            clean[key] = Path(value)
        elif key == "camera_ids":
            clean[key] = _optional_str_list(value)
        elif key == "subsets":
            selected = {}
            selected_split = {}
            for subset in value:
                subset_name = str(subset)
                if subset_name in config.scenes_by_subset:
                    selected[subset_name] = config.scenes_by_subset[subset_name]
                if subset_name in config.split_by_subset:
                    selected_split[subset_name] = config.split_by_subset[subset_name]
            clean["scenes_by_subset"] = selected
            clean["split_by_subset"] = selected_split
        else:
            clean[key] = value
    return replace(config, **clean)


def _required(data: Dict[str, Any], key: str) -> Any:
    if key not in data or data.get(key) is None:
        raise ValueError("Missing required pipeline config field: %s" % key)
    return data[key]


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_str_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]
