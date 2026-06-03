"""Package final Track 1 submission artifacts."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deep_oc_sort_3d.final_export.track1_packaging import package_track1_submission


def package_submission(args: Any) -> None:
    """Create submission package."""
    config = _load_package_config(args.package_config)
    track1_path = Path(_value(args.track1, config.get("track1_path"), ""))
    output_root = Path(_value(args.output_package_root, config.get("output_package_root"), ""))
    baseline_name = str(_value(args.baseline_name, config.get("baseline_name"), "baseline_v1_geometry_only"))
    config_paths = _list_value(args.config_paths, config.get("configs"))
    reports = _list_value(args.reports, config.get("reports"))
    make_zip = bool(_value(args.make_zip, config.get("make_zip"), False))
    summary = package_track1_submission(
        track1_path=track1_path,
        output_package_root=output_root,
        config_paths=config_paths,
        reports=reports,
        baseline_name=baseline_name,
        overwrite=args.overwrite,
        make_zip=make_zip,
        show_progress=args.progress,
    )
    print("package_root: %s" % summary.get("package_root"))
    print("manifest_path: %s" % summary.get("manifest_path"))
    print("checksums_path: %s" % summary.get("checksums_path"))
    print("zip_path: %s" % summary.get("zip_path"))


def _load_package_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    section = data.get("submission_package", data)
    return section if isinstance(section, dict) else {}


def _value(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _list_value(cli_value: Any, config_value: Any) -> List[str]:
    value = _value(cli_value, config_value, [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Package Track 1 submission artifacts.")
    parser.add_argument("--package-config", type=Path, default=None)
    parser.add_argument("--track1", type=Path, default=None)
    parser.add_argument("--output-package-root", type=Path, default=None)
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument("--config", dest="config_paths", action="append", default=None)
    parser.add_argument("--report", dest="reports", action="append", default=None)
    parser.add_argument("--make-zip", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    package_submission(args)


if __name__ == "__main__":
    main()
