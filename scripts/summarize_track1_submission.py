"""Summarize final Track 1 submission checks and package status."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def summarize_track1_submission(args: Any) -> None:
    """Print final Track 1 submission summary."""
    checks_root = args.root / "final_checks"
    validation = _read_json(checks_root / "final_validation_report.json")
    distribution = _read_json(checks_root / "final_distribution_report.json")
    dedup = _read_json(checks_root / "dedup_audit_report.json")
    package_manifest = _find_manifest(args.package_root)
    print("root: %s" % args.root)
    if validation:
        print("validation status: %s" % validation.get("status"))
        print("validation errors: %s" % validation.get("num_errors"))
        print("total rows: %s" % validation.get("total_rows"))
    else:
        print("validation: missing")
    if distribution:
        print("scene coverage: %s" % json.dumps(distribution.get("per_scene_rows", {}), sort_keys=True))
        print("class coverage: %s" % json.dumps(distribution.get("per_class_rows", {}), sort_keys=True))
    if dedup:
        print("generic rows: %s" % dedup.get("generic_rows_total"))
        print("official rows: %s" % dedup.get("official_rows_total"))
        print("duplicates removed: %s" % dedup.get("duplicate_rows_removed_estimated"))
    if package_manifest:
        print("package manifest: %s" % package_manifest.get("package_root"))
        print("package files: %d" % len(package_manifest.get("files", [])))
    else:
        print("package: not found")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _find_manifest(package_root: Optional[Path]) -> Optional[Dict[str, Any]]:
    if package_root is None:
        return None
    return _read_json(package_root / "manifest.json")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize final Track 1 submission outputs.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--package-root", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summarize_track1_submission(args)


if __name__ == "__main__":
    main()
