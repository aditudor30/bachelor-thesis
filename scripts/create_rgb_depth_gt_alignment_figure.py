"""Create a two-panel RGB-depth-GT alignment figure."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.visualization.rgb_depth_alignment_figure import create_alignment_panel_figure


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Create RGB-depth-GT alignment figure.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/figure_rgb_depth_gt_alignment.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--auto-select", dest="auto_select", action="store_true")
    parser.add_argument("--no-auto-select", dest="auto_select", action="store_false")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(progress=True, auto_select=None)
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    overrides: Dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "split": args.split,
        "scene_name": args.scene_name,
        "camera_id": args.camera_id,
        "frame_id": args.frame_id,
        "auto_select": args.auto_select,
        "output_dir": args.output_dir,
        "max_objects": args.max_objects,
        "no_labels": bool(args.no_labels),
    }
    result = create_alignment_panel_figure(
        config_path=Path(args.config),
        overrides=overrides,
        progress=bool(args.progress),
        overwrite=bool(args.overwrite),
    )
    print("output_png:", result.get("output_png"))
    print("output_pdf:", result.get("output_pdf"))
    print("metadata:", result.get("output_metadata", result.get("metadata_path")))
    print("selected objects:", result.get("num_objects_selected"))


if __name__ == "__main__":
    main()
