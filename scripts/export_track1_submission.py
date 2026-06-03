"""Export official Track 1 submission or an unconfirmed preview."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from deep_oc_sort_3d.final_export.track1_export_types import (
    default_unconfirmed_track1_schema,
    load_track1_schema_yaml,
)
from deep_oc_sort_3d.final_export.track1_mapping import (
    build_track1_mapping,
    infer_available_generic_columns,
    load_generic_tracking_csv,
    write_track1_mapping_report,
)
from deep_oc_sort_3d.final_export.track1_writer import export_track1_from_generic


def export_track1_submission(args: Any) -> None:
    """Export Track 1 submission from generic MVP export."""
    config = _load_export_config(args.config)
    generic_root = Path(_value(args.generic_export_root, config.get("generic_export_root"), ""))
    output_root = Path(_value(args.output_root, config.get("output_root"), "output/track1_submission/debug"))
    schema_yaml = _value(args.schema_yaml, config.get("schema_yaml"), None)
    subsets = _list_value(args.subsets, config.get("subsets"))
    scenes = _list_value(args.scenes, config.get("scenes"))
    progress = bool(_value(args.progress, config.get("progress"), True))
    force_preview = bool(_value(args.force_unconfirmed_preview, config.get("force_unconfirmed_preview"), False))

    schema = load_track1_schema_yaml(schema_yaml) if schema_yaml else default_unconfirmed_track1_schema()
    if not schema.schema_confirmed and not force_preview:
        print("warning: schema is not confirmed; writing unconfirmed preview only")
    generic_columns = _load_first_generic_header(generic_root, subsets=subsets, scenes=scenes)
    mapping = build_track1_mapping(schema, generic_columns)
    output_path = output_root / ("track1.txt" if schema.schema_confirmed else "track1_unconfirmed_preview.csv")
    summary = export_track1_from_generic(
        generic_export_root=generic_root,
        output_path=output_path,
        schema=schema,
        mapping=mapping,
        subsets=subsets,
        scenes=scenes,
        show_progress=progress,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "track1_export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_track1_mapping_report(mapping, output_root / "track1_mapping_report.md", schema, generic_columns)
    print("schema_confirmed: %s" % schema.schema_confirmed)
    print("official_export_created: %s" % summary.get("official_export_created"))
    print("rows_written: %s" % summary.get("rows_written"))
    print("output_path: %s" % summary.get("output_path"))
    for warning in summary.get("warnings", []):
        print("warning: %s" % warning)


def _load_first_generic_header(
    generic_root: Path,
    subsets: Optional[Any],
    scenes: Optional[Any],
) -> Any:
    files = sorted(generic_root.glob("*.csv"))
    if not files:
        files = sorted(generic_root.rglob("*.csv"))
    if not files:
        return []
    rows = load_generic_tracking_csv(files[0])
    return infer_available_generic_columns(rows)


def _load_export_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    section = data.get("track1_export", data)
    return section if isinstance(section, dict) else {}


def _value(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _list_value(cli_value: Any, config_value: Any) -> Optional[Any]:
    value = _value(cli_value, config_value, None)
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Export Track 1 submission or preview from MVP generic export.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--generic-export-root", type=Path, default=None)
    parser.add_argument("--schema-yaml", type=Path, default=None)
    parser.add_argument("--schema-report", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--force-unconfirmed-preview", action="store_true", default=None)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=None)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    export_track1_submission(args)


if __name__ == "__main__":
    main()
