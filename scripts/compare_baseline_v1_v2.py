"""Compare baseline_v1_geometry_only and baseline_v2_pseudo3d outputs."""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from deep_oc_sort_3d.pseudo3d_integration.baseline_v2_comparison import (
    compare_baseline_v1_v2,
    write_comparison_csv,
    write_comparison_json,
    write_comparison_report_md,
)


def run(args: Any) -> Dict[str, Any]:
    cfg = _load_config(args)
    paths = cfg.get("comparison", cfg)
    summary = compare_baseline_v1_v2(paths.get("baseline_v1", {}), paths.get("baseline_v2", {}))
    output_root = Path(paths.get("output_root", args.output_root or "output/baseline_v2_pseudo3d_comparison"))
    write_comparison_json(summary, output_root / "baseline_v1_vs_v2_summary.json")
    write_comparison_csv(summary, output_root / "baseline_v1_vs_v2_summary.csv")
    write_comparison_report_md(summary, output_root / "baseline_v1_vs_v2_report.md")
    print("v2 assessment: %s" % summary.get("assessment"))
    return summary


def _load_config(args: Any) -> Dict[str, Any]:
    if args.config is not None:
        data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    return {
        "comparison": {
            "output_root": str(args.output_root or "output/baseline_v2_pseudo3d_comparison"),
            "baseline_v1": {
                "global_mtmc_root": str(Path(args.baseline_v1_root) / "global_mtmc_transition" / "yolo11m_medium_conf001_transition"),
                "final_export_root": str(Path(args.baseline_v1_root) / "final_mvp_exports" / "yolo11m_medium_conf001_transition"),
                "track1_root": str(Path(args.baseline_v1_root) / "track1_submission" / "yolo11m_medium_conf001_transition"),
            },
            "baseline_v2": {
                "global_mtmc_root": str(Path(args.baseline_v2_root) / "global_mtmc_transition" / "baseline_v2_pseudo3d"),
                "final_export_root": str(Path(args.baseline_v2_root) / "final_mvp_exports" / "baseline_v2_pseudo3d"),
                "track1_root": str(Path(args.baseline_v2_root) / "track1_submission" / "baseline_v2_pseudo3d"),
                "observations_summary": str(Path(args.baseline_v2_root) / "pipeline_runs" / "baseline_v2_pseudo3d" / "summaries" / "pseudo3d_observation_summary.json"),
            },
        }
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline_v1 and baseline_v2 outputs.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--baseline-v1-root", type=Path, default=Path("output"))
    parser.add_argument("--baseline-v2-root", type=Path, default=Path("output"))
    parser.add_argument("--output-root", type=Path, default=None)
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

