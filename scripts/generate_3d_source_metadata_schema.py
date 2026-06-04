"""Generate the Step 15B 3D source metadata schema."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.audit3d.audit3d_io import write_json, write_markdown
from deep_oc_sort_3d.priors3d.source_metadata_schema import (
    build_source_metadata_schema,
    build_source_metadata_schema_markdown,
)


def run(args: Any) -> Dict[str, Any]:
    schema = build_source_metadata_schema()
    write_json(schema, args.output_json)
    write_markdown(build_source_metadata_schema_markdown(schema), args.output_md)
    print("Wrote source metadata schema: %s" % args.output_json)
    return schema


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 3D source metadata schema.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
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
