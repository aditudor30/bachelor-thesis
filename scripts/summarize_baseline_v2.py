"""Summarize baseline_v2 pseudo-3D run artifacts."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.audit3d.audit3d_io import read_json_if_exists, write_csv, write_json


def run(args: Any) -> Dict[str, Any]:
    root = Path(args.root)
    summary = {
        "observations": read_json_if_exists(root / "pipeline_runs" / "baseline_v2_pseudo3d" / "summaries" / "pseudo3d_observation_summary.json"),
        "local_tracking": _read_any(root / "local_tracks" / "baseline_v2_pseudo3d" / "summaries", ["local_tracking_summary.json"]),
        "tracklets": _read_any(root / "tracklets" / "baseline_v2_pseudo3d" / "summaries", ["tracklet_summary.json"]),
        "candidates": _read_any(root / "mtmc_candidates" / "baseline_v2_pseudo3d" / "summaries", ["candidate_summary.json"]),
        "motion_filtering": _read_any(root / "mtmc_candidates_motion_clean" / "baseline_v2_pseudo3d" / "summaries", ["motion_quality_summary.json"]),
        "global_association": _read_any(root / "global_mtmc_transition" / "baseline_v2_pseudo3d" / "summaries", ["global_transition_summary.json"]),
        "final_export": _read_any(root / "final_mvp_exports" / "baseline_v2_pseudo3d" / "summaries", ["export_summary.json"]),
        "track1": read_json_if_exists(root / "track1_submission" / "baseline_v2_pseudo3d" / "track1_export_summary.json"),
    }
    output_root = Path(args.output_root or (root / "pipeline_runs" / "baseline_v2_pseudo3d" / "summaries"))
    write_json(summary, output_root / "baseline_v2_summary.json")
    write_csv(_summary_rows(summary), output_root / "baseline_v2_summary.csv")
    print("baseline_v2 summary written: %s" % (output_root / "baseline_v2_summary.json"))
    return summary


def _read_any(root: Path, names: List[str]) -> Dict[str, Any]:
    for name in names:
        data = read_json_if_exists(root / name)
        if data:
            return data
    return {}


def _summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for section, data in summary.items():
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, (dict, list)):
                    rows.append({"section": section, "metric": key, "value": value})
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize baseline_v2 pseudo-3D artifacts.")
    parser.add_argument("--root", type=Path, default=Path("output"))
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

