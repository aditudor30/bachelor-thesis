"""CLI for comparing Person cleanup runs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.person_cleanup.person_cleanup_comparison import compare_person_cleanup_runs
from deep_oc_sort_3d.person_cleanup.person_cleanup_report import write_person_cleanup_report
from deep_oc_sort_3d.person_cleanup.person_cleanup_io import load_yaml


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Compare Person cleanup runs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = compare_person_cleanup_runs(args.config, progress=args.progress)
    config = load_yaml(args.config)
    root = Path(config.get("person_cleanup", {}).get("output_root", "output/person_cleanup/baseline_v2_pseudo3d_fullcam"))
    write_person_cleanup_report(summary, root / "comparison" / "PERSON_CLEANUP_REPORT.md")
    print("runs: %d" % len(summary.get("runs", [])))
    print("best_run: %s" % summary.get("best_person_cleanup_recommendation", {}).get("best_run"))


if __name__ == "__main__":
    main()

