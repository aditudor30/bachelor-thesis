"""Summarize selected MVP figures from a manifest."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.visualization3d.figure_export_manifest import read_figure_manifest, summarize_figure_manifest


def main() -> None:
    args = parse_args()
    records = read_figure_manifest(args.manifest)
    summary = summarize_figure_manifest(records)
    print(json.dumps(summary, indent=2, sort_keys=True))
    for record in records:
        print(
            "%s %s %s %s frame=%s score=%s"
            % (
                record.figure_name,
                record.figure_type,
                record.scene_name,
                str(record.camera_id),
                str(record.frame_id),
                str(record.score),
            )
        )
        print("  caption: %s" % record.caption_suggestion)
    if args.output_md is not None:
        write_markdown(records, summary, args.output_md)
        print("output_md: %s" % args.output_md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def write_markdown(records, summary, output_path: Path) -> None:
    """Write a compact Markdown figure summary."""
    lines = []
    lines.append("# MVP Figure Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    for record in records:
        lines.append("## %s" % record.figure_name)
        lines.append("")
        lines.append("- type: `%s`" % record.figure_type)
        lines.append("- scene: `%s`" % record.scene_name)
        lines.append("- camera: `%s`" % str(record.camera_id))
        lines.append("- frame: `%s`" % str(record.frame_id))
        lines.append("- score: `%s`" % str(record.score))
        lines.append("- output: `%s`" % record.output_path)
        lines.append("- caption: %s" % record.caption_suggestion)
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

