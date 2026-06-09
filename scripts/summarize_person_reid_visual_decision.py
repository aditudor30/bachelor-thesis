"""Summarize Step 18D visual decision outputs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import read_csv_rows, read_json
from deep_oc_sort_3d.reid_visual_decision.visual_decision_metrics import decide_final_variant, summarize_visual_rows


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    summary = read_json(root / "comparison" / "visual_decision_summary.json") or {}
    recomputed = recompute_summary_from_review_csvs(root)
    if recomputed.get("total_review_events"):
        summary = recomputed
    decision = read_json(root / "comparison" / "final_variant_decision.json") or {}
    merge_summary = read_json(root / "merge_audit" / "merge_event_summary.json") or {}
    print("output_root: %s" % root)
    print("total_merge_events: %s" % merge_summary.get("total_events"))
    print("review_events: %s" % summary.get("total_review_events"))
    print("auto_label_counts: %s" % summary.get("auto_label_counts"))
    print("mean_risk_score: %s" % summary.get("mean_risk_score"))
    print("final_verdict: %s" % decision.get("final_verdict"))
    print("reason: %s" % decision.get("reason"))
    if summary.get("source") == "recomputed_from_visual_review_events_csv":
        selected = {"best_run": decision.get("selected_variant_from_step18c", "combined_safe_080")}
        recomputed_decision = decide_final_variant(summary, selected)
        print("recomputed_final_verdict: %s" % recomputed_decision.get("final_verdict"))
        print("recomputed_reason: %s" % recomputed_decision.get("reason"))


def recompute_summary_from_review_csvs(root: Path):
    """Recompute visual summary from latest per-variant review CSV files."""
    rows = []
    merge_dir = Path(root) / "merge_audit"
    if not merge_dir.exists():
        return {}
    for path in sorted(merge_dir.glob("*_visual_review_events.csv")):
        csv_rows, _fields = read_csv_rows(path)
        rows.extend(csv_rows)
    if not rows:
        return {}
    summary = summarize_visual_rows(rows)
    summary["source"] = "recomputed_from_visual_review_events_csv"
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize visual decision artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("output/person_reid_visual_decision/baseline_v2_pseudo3d_fullcam"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
