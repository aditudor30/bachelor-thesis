"""Stabilize one pseudo-3D prediction file."""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilization_io import (
    read_pseudo3d_outputs,
    write_smoothing_report_csv,
    write_smoothing_report_json,
    write_stabilized_outputs_csv,
    write_stabilized_outputs_jsonl,
)
from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilizer import Pseudo3DStabilizer


def run(args: Any) -> Dict[str, Any]:
    """Run stabilization for one input file."""
    config = _load_yaml(args.config)
    outputs = read_pseudo3d_outputs(args.input)
    stabilizer = Pseudo3DStabilizer(config)
    stabilized, report = stabilizer.stabilize_batch(outputs)
    write_stabilized_outputs_jsonl(stabilized, args.output_jsonl)
    write_stabilized_outputs_csv(stabilized, args.output_csv)
    write_smoothing_report_json(_compact_report(report), args.report)
    write_smoothing_report_csv(list(report.get("track_reports", [])), Path(args.report).with_suffix(".csv"))
    print("Stabilized predictions: %s" % report.get("num_predictions"))
    print("Jump-corrected records: %s" % report.get("num_jump_corrected"))
    return report


def _compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(report)
    compact["track_report_count"] = len(compact.get("track_reports", []))
    return compact


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stabilize one pseudo-3D prediction file.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
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
