"""Summarize Step 15G full-camera pseudo-3D outputs."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import write_markdown
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_report import build_fullcam_generation_report
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_summary import summarize_step15g, write_step15g_summary


def run(args: Any) -> Dict[str, Any]:
    """Write Step 15G summary and report files."""
    summary = summarize_step15g(args.root)
    write_step15g_summary(summary, args.root)
    write_markdown(build_fullcam_generation_report(summary), args.root / "report" / "PSEUDO3D_FULLCAM_GENERATION_REPORT.md")
    print("Step 15G recommendation: %s" % summary.get("recommendation"))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Step 15G full-camera pseudo-3D outputs.")
    parser.add_argument("--root", required=True, type=Path)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
