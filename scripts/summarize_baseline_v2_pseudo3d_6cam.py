"""Summarize the baseline_v2_pseudo3d_6cam comparison outputs."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.audit3d.audit3d_io import read_json_if_exists, write_csv, write_json


def run(args: Any) -> Dict[str, Any]:
    root = Path(args.root)
    summary = {
        "subset": read_json_if_exists(root / "subset_definition" / "six_camera_subset_summary.json"),
        "comparison": read_json_if_exists(root / "comparison" / "v1_vs_v2_6cam_summary.json"),
        "verdict": read_json_if_exists(root / "comparison" / "verdict.json"),
        "pseudo3d_coverage": read_json_if_exists(root / "diagnostics" / "pseudo3d_coverage_6cam.json"),
        "source_metadata": read_json_if_exists(root / "diagnostics" / "source_metadata_completeness_6cam.json"),
    }
    write_json(summary, root / "comparison" / "sixcam_final_summary.json")
    write_csv(_summary_rows(summary), root / "comparison" / "sixcam_final_summary.csv")
    print("sixcam summary written: %s" % (root / "comparison" / "sixcam_final_summary.json"))
    return summary


def _summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    verdict = summary.get("verdict", {})
    coverage = summary.get("pseudo3d_coverage", {})
    rows.append({"metric": "verdict", "value": verdict.get("label")})
    rows.append({"metric": "verdict_reason", "value": verdict.get("reason")})
    rows.append({"metric": "pseudo3d_used_rate", "value": coverage.get("pseudo3d_used_rate")})
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize baseline_v2_pseudo3d_6cam outputs.")
    parser.add_argument("--root", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

