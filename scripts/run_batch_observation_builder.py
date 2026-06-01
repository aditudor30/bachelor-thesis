"""Run batch Observation3D construction for a configured pipeline run."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.pipeline.batch_observation_builder import BatchObservationBuilder
from deep_oc_sort_3d.pipeline.pipeline_paths import get_summaries_dir, make_run_root
from deep_oc_sort_3d.pipeline.pipeline_summary import (
    aggregate_per_class_from_observations,
    aggregate_per_scene_camera_summary,
    print_pipeline_summary,
    write_observation_summary,
    write_per_class_summary,
    write_per_scene_camera_summary,
)
from deep_oc_sort_3d.pipeline.run_config import (
    load_pipeline_config,
    save_resolved_config,
    update_config,
    validate_pipeline_config,
)


def run_batch_observation_builder(args: Any) -> None:
    """Build observations from batch Detection2D CSV files."""
    config = _apply_overrides(load_pipeline_config(args.config), args)
    messages = validate_pipeline_config(config)
    if messages:
        raise ValueError("Invalid pipeline config: %s" % "; ".join(messages))
    run_root = make_run_root(config.output_root, config.run_name)
    summaries_dir = get_summaries_dir(run_root)
    save_resolved_config(config, summaries_dir / "run_config_resolved.yaml")
    rows = BatchObservationBuilder(config=config, overwrite=args.overwrite).run()
    write_observation_summary(rows, summaries_dir / "observations_summary.csv")
    per_scene_camera = aggregate_per_scene_camera_summary(rows)
    write_per_scene_camera_summary(per_scene_camera, summaries_dir / "per_scene_camera_summary.csv")
    observation_paths = [Path(row["observations_jsonl"]) for row in rows if row.get("status") != "error"]
    per_class = aggregate_per_class_from_observations(observation_paths)
    write_per_class_summary(
        per_class,
        summaries_dir / "per_class_summary.csv",
        summaries_dir / "per_class_summary.json",
    )
    print_pipeline_summary(observation_rows=rows, per_class_summary=per_class)
    print("Wrote %s" % (summaries_dir / "observations_summary.csv"))


def _apply_overrides(config: Any, args: Any) -> Any:
    return update_config(
        config,
        root=args.root,
        output_root=args.output_root,
        run_name=args.run_name,
        max_frames=args.max_frames,
        camera_ids=args.camera_ids,
        subsets=args.subsets,
        iou_threshold=args.iou_threshold,
        depth_sampling_method=args.depth_sampling_method,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run batch Observation3D construction.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-ids", nargs="+", default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--depth-sampling-method", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_batch_observation_builder(args)


if __name__ == "__main__":
    main()
