"""Print or save manual commands for YOLO curriculum Step 9C."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.detection2d.yolo_curriculum_training_plan import (
    build_curriculum_commands,
    load_curriculum_plan,
    render_commands_markdown,
)


def prepare_yolo_curriculum_9c(args: Any) -> None:
    """Create a reproducible manual command plan without running YOLO."""
    plan = load_curriculum_plan(args.config)
    commands = build_curriculum_commands(plan)
    rendered = render_commands_markdown(commands)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print("Wrote %s" % args.output)
    if args.print_commands or args.output is None:
        print(rendered)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Prepare manual YOLO curriculum Step 9C commands.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--print-commands", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    prepare_yolo_curriculum_9c(args)


if __name__ == "__main__":
    main()
