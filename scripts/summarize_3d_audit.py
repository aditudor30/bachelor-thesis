"""CLI for printing a compact 3D audit summary."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.audit3d.audit3d_io import read_json_if_exists


def run(args: Any) -> None:
    summary = read_json_if_exists(args.audit_root / "report" / "TRACK1_3D_AUDIT_SUMMARY.json")
    if not summary:
        print("No summary found under %s" % args.audit_root)
        return
    print("Track1 rows: %s" % summary.get("track1_rows"))
    print("Generic rows: %s" % summary.get("generic_rows"))
    print("Unknown source records: %s" % summary.get("unknown_source_records"))
    print("Smoothness status: %s" % summary.get("smoothness_status_distribution"))
    print("Projection success rate: %s" % summary.get("projection_success_rate"))
    print("MVP assessment: %s" % summary.get("mvp_assessment"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a completed 3D audit output root.")
    parser.add_argument("--audit-root", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

