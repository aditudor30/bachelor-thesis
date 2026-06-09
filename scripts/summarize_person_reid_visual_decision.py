"""Summarize Step 18D visual decision outputs."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import read_json


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    summary = read_json(root / "comparison" / "visual_decision_summary.json") or {}
    decision = read_json(root / "comparison" / "final_variant_decision.json") or {}
    merge_summary = read_json(root / "merge_audit" / "merge_event_summary.json") or {}
    print("output_root: %s" % root)
    print("total_merge_events: %s" % merge_summary.get("total_events"))
    print("review_events: %s" % summary.get("total_review_events"))
    print("auto_label_counts: %s" % summary.get("auto_label_counts"))
    print("mean_risk_score: %s" % summary.get("mean_risk_score"))
    print("final_verdict: %s" % decision.get("final_verdict"))
    print("reason: %s" % decision.get("reason"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize visual decision artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("output/person_reid_visual_decision/baseline_v2_pseudo3d_fullcam"))
    return parser.parse_args()


if __name__ == "__main__":
    main()

