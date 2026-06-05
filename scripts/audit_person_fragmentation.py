"""CLI for Person fragmentation audit."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.person_cleanup.person_fragmentation_audit import run_person_fragmentation_audit


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Audit Person fragmentation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_person_fragmentation_audit(args.config, progress=args.progress, overwrite=args.overwrite)
    print("person_global_tracks: %s" % summary.get("person_global_tracks"))
    print("person_generic_rows: %s" % summary.get("person_generic_rows"))


if __name__ == "__main__":
    main()

