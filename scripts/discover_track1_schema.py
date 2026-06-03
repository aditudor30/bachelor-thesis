"""CLI for local Track 1 schema discovery."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.final_export.track1_schema_discovery import (
    discover_track1_schema,
    write_schema_discovery_report,
)


def run_discovery(args: Any) -> None:
    """Run local schema discovery and write a Markdown report."""
    report = discover_track1_schema(
        repo_root=args.repo_root,
        max_file_size_mb=args.max_file_size_mb,
        show_progress=args.progress,
    )
    write_schema_discovery_report(report, args.output)
    print("schema_found_locally: %s" % report.get("found"))
    print("matches: %s" % len(report.get("matches", [])))
    print("candidate_schema_files: %s" % len(report.get("candidate_schema_files", [])))
    for item in report.get("matches", [])[:25]:
        print(
            "%s:%s [%s] %s"
            % (
                item.get("file_path"),
                item.get("line_number"),
                item.get("matched_term"),
                item.get("line_text"),
            )
        )
    if not report.get("found"):
        print(
            "Official Track 1 schema was not found in the local repository. "
            "Do not implement a final track1.txt writer until the schema is confirmed."
        )
    print("Wrote %s" % args.output)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Discover local Track 1 schema references.")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-file-size-mb", type=int, default=5)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_discovery(args)


if __name__ == "__main__":
    main()
