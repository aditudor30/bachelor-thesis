"""Export easy/medium curriculum YOLO datasets from bbox audit CSVs."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deep_oc_sort_3d.detection2d.yolo_curriculum_exporter import YoloCurriculumExporter
from deep_oc_sort_3d.detection2d.yolo_curriculum_summary import print_curriculum_summary


def export_yolo_curriculum(args: Any) -> None:
    """Run a curriculum export without training a detector."""
    config = _load_config(args.config)
    curriculum_config = _config_section(config, "curriculum")
    priority_config = _config_section(config, "priorities")

    curriculum = _value(args.curriculum, curriculum_config.get("name"), "easy_allclass")
    root = _required_path(_value(args.root, _config_section(config, "dataset").get("root"), None), "root")
    audit_csv = _required_path(
        _value(args.audit_csv, _config_section(config, "dataset").get("audit_csv"), None),
        "audit_csv",
    )
    output = _required_path(_value(args.output, _config_section(config, "dataset").get("output"), None), "output")
    class_rich = _value(args.class_rich_frames, _config_section(config, "dataset").get("class_rich_frames"), None)
    allowed_difficulties = _list_value(args.allowed_difficulties, curriculum_config.get("allowed_difficulties"))
    exclude_scenes = _list_value(args.exclude_scenes, curriculum_config.get("exclude_scenes"))
    target_classes = _list_value(args.target_classes, curriculum_config.get("target_classes"))
    include_all_visible = args.include_all_visible_objects
    if include_all_visible is None:
        include_all_visible = bool(curriculum_config.get("include_all_visible_objects", True))

    exporter = YoloCurriculumExporter(
        root=root,
        output_dir=output,
        audit_csv=audit_csv,
        class_rich_frames_csv=class_rich,
        curriculum=curriculum,
        allowed_difficulties=allowed_difficulties,
        target_classes=target_classes,
        class_priority=priority_config.get("class_priority"),
        scene_priority=priority_config.get("scene_priority"),
        camera_priority=priority_config.get("camera_priority"),
        exclude_scenes=exclude_scenes,
        max_frames_total=_int_value(args.max_frames_total, curriculum_config.get("max_frames_total")),
        max_frames_per_class=_int_value(args.max_frames_per_class, curriculum_config.get("max_frames_per_class")),
        max_person_only_frames=_int_value(args.max_person_only_frames, curriculum_config.get("max_person_only_frames"), 500),
        min_area_norm=_float_value(args.min_area_norm, curriculum_config.get("min_area_norm")),
        include_all_visible_objects=include_all_visible,
        jpeg_quality=_int_value(args.jpeg_quality, curriculum_config.get("jpeg_quality"), 95),
    )
    records = exporter.export()
    summary = exporter.summarize(records)
    print("Wrote data.yaml: %s" % (output / "data.yaml"))
    print("Wrote manifest: %s" % (output / "curriculum_manifest.csv"))
    print("Wrote summary: %s" % (output / "curriculum_summary.json"))
    print_curriculum_summary(summary)


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {}


def _config_section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = config.get(name, {})
    if isinstance(section, dict):
        return section
    return {}


def _value(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _list_value(cli_value: Optional[List[str]], config_value: Any) -> Optional[List[str]]:
    if cli_value is not None:
        return list(cli_value)
    if config_value is None:
        return None
    return [str(item) for item in list(config_value)]


def _int_value(cli_value: Optional[int], config_value: Any, default: Optional[int] = None) -> Optional[int]:
    value = _value(cli_value, config_value, default)
    if value is None:
        return None
    return int(value)


def _float_value(cli_value: Optional[float], config_value: Any) -> Optional[float]:
    value = _value(cli_value, config_value, None)
    if value is None:
        return None
    return float(value)


def _required_path(value: Any, name: str) -> Path:
    if value is None:
        raise ValueError("Missing required argument or config value: %s" % name)
    return Path(value)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Export YOLO curriculum dataset from bbox audit CSV.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--class-rich-frames", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--curriculum", choices=["easy_allclass", "medium_allclass"], default=None)
    parser.add_argument("--exclude-scenes", nargs="*", default=None)
    parser.add_argument("--target-classes", nargs="+", default=None)
    parser.add_argument("--max-frames-total", type=int, default=None)
    parser.add_argument("--max-frames-per-class", type=int, default=None)
    parser.add_argument("--max-person-only-frames", type=int, default=None)
    parser.add_argument("--min-area-norm", type=float, default=None)
    parser.add_argument("--allowed-difficulties", nargs="+", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--include-all-visible-objects", dest="include_all_visible_objects", action="store_true", default=None)
    mode.add_argument("--only-allowed-difficulty-objects", dest="include_all_visible_objects", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    export_yolo_curriculum(args)


if __name__ == "__main__":
    main()
