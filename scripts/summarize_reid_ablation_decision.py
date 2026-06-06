"""CLI for printing a compact Step 16C decision summary."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.reid_ablation.ablation_io import read_json


def main() -> None:
    """Print summary."""
    parser = argparse.ArgumentParser(description="Summarize ReID ablation decision outputs.")
    parser.add_argument("--root", type=Path, default=Path("output/reid_ablation_decision/baseline_v2_pseudo3d_fullcam"))
    args = parser.parse_args()
    decision = read_json(args.root / "comparison" / "final_variant_decision.json") or {}
    summary = read_json(args.root / "comparison" / "reid_ablation_comparison.json") or {}
    print("verdicts: %s" % ", ".join([str(item) for item in decision.get("verdicts", [])]))
    print("kept_variants:")
    for key, value in sorted(decision.get("kept_variants", {}).items()):
        print("  %s: %s" % (key, value))
    print("final_recommendation: %s" % decision.get("final_recommendation"))
    print("")
    for row in summary.get("variants", []):
        print(
            "%s type=%s safe=%s noop=%s source=%s track1_delta=%s person_frag_delta=%s reid_merges=%s export_drops=%s rec=%s"
            % (
                row.get("variant_name"),
                row.get("source_type"),
                row.get("is_safe"),
                row.get("is_noop"),
                row.get("improvement_source"),
                row.get("track1_rows_delta_vs_v2"),
                row.get("person_fragmentation_delta_vs_v2"),
                row.get("num_reid_merges"),
                row.get("num_export_dropped_rows"),
                row.get("recommendation"),
            )
        )


if __name__ == "__main__":
    main()
