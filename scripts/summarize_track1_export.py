"""Summarize Track 1 export scaffold outputs."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def summarize_track1_export(args: Any) -> None:
    """Print Track 1 export summary and validation status."""
    summary = _read_json(args.root / "track1_export_summary.json")
    validation = _find_validation(args.root)
    print("Track 1 export root: %s" % args.root)
    if summary is None:
        print("summary: missing")
    else:
        print("schema_confirmed: %s" % summary.get("schema_confirmed"))
        print("official_export_created: %s" % summary.get("official_export_created"))
        print("rows_written: %s" % summary.get("rows_written"))
        print("output_path: %s" % summary.get("output_path"))
        for warning in summary.get("warnings", []):
            print("warning: %s" % warning)
    if validation is None:
        print("validation: missing")
    else:
        print("validation official: %s" % validation.get("official_validation"))
        print("validation status: %s" % validation.get("status"))
        print("validation errors: %s" % validation.get("num_errors"))
    if summary is None or not summary.get("official_export_created"):
        print("TODO: confirm official Track 1 schema before packaging final track1.txt")


def _find_validation(root: Path) -> Optional[Dict[str, Any]]:
    for name in ["validation_report.json", "track1_validation_report.json"]:
        data = _read_json(root / name)
        if data is not None:
            return data
    return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize Track 1 export scaffold outputs.")
    parser.add_argument("--root", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summarize_track1_export(args)


if __name__ == "__main__":
    main()
