"""Print compact baseline_v2_pseudo3d_fullcam summary."""

import argparse
import json
from pathlib import Path


def summarize(root: Path) -> None:
    """Print the comparison summary and verdict."""
    summary_path = root / "baseline_v1_vs_v2_fullcam_summary.json"
    verdict_path = root / "verdict.json"
    summary = _read_json(summary_path)
    verdict = _read_json(verdict_path)
    print("root: %s" % root)
    print("verdict: %s" % verdict.get("label"))
    v2 = summary.get("baseline_v2_fullcam", {})
    observations = v2.get("observations", {})
    track1 = v2.get("track1", {})
    global_assoc = v2.get("global_association", {})
    print("pseudo3d_used_rate: %s" % observations.get("pseudo3d_used_rate"))
    print("fallback_original_used_rate: %s" % observations.get("fallback_original_used_rate"))
    print("track1_rows: %s" % track1.get("rows"))
    print("track1_validation_errors: %s" % track1.get("validation_errors"))
    print("global_tracks: %s" % global_assoc.get("global_tracks"))
    print("multi_camera_tracks: %s" % global_assoc.get("multi_camera_tracks"))


def _read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize baseline_v2_pseudo3d_fullcam comparison outputs.")
    parser.add_argument("--root", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summarize(args.root)


if __name__ == "__main__":
    main()
