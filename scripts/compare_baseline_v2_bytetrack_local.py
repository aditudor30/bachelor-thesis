"""Compare the Step 21B baseline with V1 and V2 current."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_comparison import (
    compare_bytetrack_pipeline,
    write_comparison_outputs,
)
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_io import read_json
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_pipeline_config import load_bytetrack_pipeline_config
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_report import write_bytetrack_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline V2 ByteTrack-local outputs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_bytetrack_pipeline_config(args.config)
    summary = compare_bytetrack_pipeline(config)
    precheck_root = Path(str(config.get("paths", {}).get("output_precheck_root")))
    summary["precheck"] = read_json(precheck_root / "precheck_verdict.json")
    output_root = Path(str(config.get("paths", {}).get("output_comparison_root")))
    write_comparison_outputs(summary, output_root)
    write_bytetrack_report(summary, output_root / "BASELINE_V2_BYTETRACK_LOCAL_REPORT.md")
    print("verdict: %s" % summary.get("verdict", {}).get("label"))
    print("output_root: %s" % output_root)


if __name__ == "__main__":
    main()
