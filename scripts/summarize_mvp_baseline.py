"""Print a compact summary of the configured MVP baseline."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from deep_oc_sort_3d.scripts.check_mvp_outputs import load_mvp_baseline_config


def summarize_mvp_baseline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact MVP baseline summary."""
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    final_root = Path(str(paths.get("final_export_root", "")))
    summary = {
        "name": config.get("name"),
        "detector": config.get("detector", {}),
        "depth": config.get("depth", {}),
        "paths": paths,
        "pipeline": config.get("pipeline", {}).get("result_summary", {}),
        "local_tracking": config.get("local_tracking", {}).get("result_summary", {}),
        "candidates": config.get("tracklet_candidates", {}).get("result_summary", {}),
        "motion_filter": config.get("motion_filter", {}).get("result_summary", {}),
        "global_association": config.get("global_association", {}).get("result_summary", {}),
        "final_export_configured": config.get("final_export", {}).get("result_summary", {}),
        "final_export_on_disk": {
            "propagation": _read_json(final_root / "summaries" / "propagation_summary.json"),
            "generic_export": _read_json(final_root / "summaries" / "export_summary.json"),
            "validation": _read_json(final_root / "validation" / "global_validation_summary.json"),
            "eval": _read_json(final_root / "eval" / "global_eval.json"),
        },
    }
    return summary


def print_mvp_baseline_summary(summary: Dict[str, Any]) -> None:
    """Print the MVP baseline summary."""
    print("MVP baseline: %s" % summary.get("name"))
    detector = summary.get("detector", {})
    print("detector: %s %s" % (detector.get("family"), detector.get("training_stage")))
    print("detector model: %s" % detector.get("model_path"))
    print("conf/imgsz: %s / %s" % (detector.get("confidence_threshold"), detector.get("imgsz")))
    depth = summary.get("depth", {})
    print("depth: unit=%s sampling=%s test_depth=%s" % (
        depth.get("train_val_unit"),
        depth.get("default_sampling_method"),
        depth.get("test_depth_available"),
    ))
    _print_section("pipeline", summary.get("pipeline", {}))
    _print_section("local_tracking", summary.get("local_tracking", {}))
    _print_section("candidates", summary.get("candidates", {}))
    _print_section("motion_filter", summary.get("motion_filter", {}))
    _print_section("global_association", summary.get("global_association", {}))
    disk = summary.get("final_export_on_disk", {})
    print("final export on disk:")
    for key in ["propagation", "generic_export", "validation", "eval"]:
        value = disk.get(key)
        if value is None:
            print("  %s: missing" % key)
        else:
            _print_selected(key, value)
    print("track1.txt status: TODO until official schema is confirmed")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _print_section(name: str, value: Dict[str, Any]) -> None:
    print("%s:" % name)
    for key in sorted(value.keys()):
        print("  %s: %s" % (key, value.get(key)))


def _print_selected(name: str, value: Dict[str, Any]) -> None:
    keys = [
        "files",
        "input_records",
        "output_records",
        "assigned_records",
        "unassigned_records",
        "assignment_ratio",
        "rows_written",
        "num_errors",
        "num_warnings",
        "num_records",
        "unique_global_tracks",
        "global_id_purity_mean",
        "fragmentation_approx",
    ]
    print("  %s:" % name)
    for key in keys:
        if key in value:
            print("    %s: %s" % (key, value.get(key)))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize the current MVP baseline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-json", default=None, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_mvp_baseline_config(args.config)
    summary = summarize_mvp_baseline(config)
    print_mvp_baseline_summary(summary)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print("Wrote %s" % args.output_json)


if __name__ == "__main__":
    main()
