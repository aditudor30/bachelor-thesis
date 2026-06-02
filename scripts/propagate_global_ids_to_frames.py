"""Propagate global IDs to frame-level local track records."""

import argparse
import json
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.final_export.global_id_propagation import propagate_for_camera_file


def propagate_global_ids_to_frames(args: Any) -> None:
    """CLI wrapper for one camera file."""
    row = propagate_for_camera_file(
        local_tracks_csv=args.local_tracks,
        candidates_with_global_ids_path=args.candidates_with_global_ids,
        subset=args.subset,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
        include_unassigned=args.include_unassigned,
        show_progress=args.progress,
        namespace_global_ids=args.namespace_global_ids,
        global_id_stride=args.global_id_stride,
    )
    print(json.dumps(row, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Propagate global ids to local frame records.")
    parser.add_argument("--local-tracks", required=True, type=Path)
    parser.add_argument("--candidates-with-global-ids", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    assign_group = parser.add_mutually_exclusive_group()
    assign_group.add_argument("--include-unassigned", dest="include_unassigned", action="store_true")
    assign_group.add_argument("--drop-unassigned", dest="include_unassigned", action="store_false")
    namespace_group = parser.add_mutually_exclusive_group()
    namespace_group.add_argument("--namespace-global-ids", dest="namespace_global_ids", action="store_true")
    namespace_group.add_argument("--keep-local-global-ids", dest="namespace_global_ids", action="store_false")
    parser.add_argument("--global-id-stride", type=int, default=100000)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(include_unassigned=True, namespace_global_ids=True, progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    propagate_global_ids_to_frames(args)


if __name__ == "__main__":
    main()
