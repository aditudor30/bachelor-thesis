"""Print the important Step 21D audit conclusions."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_audit.audit_io import read_csv, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ByteTrack Step 21D audit")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    verdict = read_json(args.root / "comparison" / "verdict.json")
    plan = read_json(args.root / "comparison" / "recommended_fix_plan.json")
    bottlenecks = read_csv(args.root / "comparison" / "bottleneck_ranking.csv")
    print("verdict: %s" % verdict.get("label"))
    print("recommended_step_21e: %s" % verdict.get("recommended_step_21e"))
    print("reasons: %s" % ", ".join(verdict.get("reasons", [])))
    print("instrumented_export_policy_evidence: %s" % plan.get("evidence", {}).get("instrumented_export_policy_evidence"))
    print("motion_gap_or_jump_evidence: %s" % plan.get("evidence", {}).get("motion_gap_or_jump_evidence"))
    print("top bottlenecks:")
    for row in bottlenecks[:5]:
        print(
            "  %s retention=%s type=%s score=%s"
            % (row.get("bottleneck"), row.get("retention"), row.get("unit_comparison_type"), row.get("ranking_score"))
        )


if __name__ == "__main__":
    main()

