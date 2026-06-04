"""Summarize isolated pseudo-3D outputs."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import read_json_if_exists, write_json, write_markdown
from deep_oc_sort_3d.pseudo3d.pseudo3d_report import build_pseudo3d_validation_report


def run(args: Any) -> Dict[str, Any]:
    extraction = read_json_if_exists(args.root / "summaries" / "pseudo3d_extraction_summary.json")
    evaluation = read_json_if_exists(args.root / "evaluation" / "summary_eval.json")
    projection = read_json_if_exists(args.root / "projection_checks" / "projection_summary.json")
    smoothness = read_json_if_exists(args.root / "smoothness" / "pseudo3d_smoothness_summary.json")
    summary = {
        "extraction": extraction,
        "evaluation": evaluation,
        "projection": projection,
        "smoothness": smoothness,
    }
    report_root = args.root / "report"
    write_json(summary, report_root / "PSEUDO3D_STEP15C_SUMMARY.json")
    write_markdown(build_pseudo3d_validation_report(extraction, evaluation, projection, smoothness), report_root / "PSEUDO3D_ISOLATED_VALIDATION_REPORT.md")
    print("Predictions: %s" % extraction.get("num_predictions"))
    print("Extraction success rate: %s" % extraction.get("success_rate"))
    print("Projection success rate: %s" % projection.get("projection_success_rate"))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize isolated pseudo-3D outputs.")
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

