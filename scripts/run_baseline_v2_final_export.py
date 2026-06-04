"""Run final MVP export for baseline_v2 pseudo-3D."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.scripts.run_final_mvp_export import run_final_mvp_export


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline_v2 final export.")
    parser.add_argument("--config", type=Path, default=Path("deep_oc_sort_3d/configs/baseline_v2_final_export.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=None)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _ensure_attrs(
        args,
        [
            "subsets",
            "scenes",
            "camera_ids",
            "include_unassigned",
            "namespace_global_ids",
            "global_id_stride",
            "drop_invalid_bbox",
            "drop_unassigned_for_generic_export",
            "drop_invalid_bbox_for_generic_export",
        ],
    )
    run_final_mvp_export(args)


def _ensure_attrs(args, names) -> None:
    for name in names:
        if not hasattr(args, name):
            setattr(args, name, None)


if __name__ == "__main__":
    main()
