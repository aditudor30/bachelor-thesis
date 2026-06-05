"""CLI for printing a compact Person-aware association summary."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.person_association.person_association_io import read_json


def main() -> None:
    """Print compact summary."""
    parser = argparse.ArgumentParser(description="Summarize Person-aware association comparison outputs.")
    parser.add_argument("--root", type=Path, default=Path("output/person_association/baseline_v2_pseudo3d_fullcam"))
    args = parser.parse_args()
    summary = read_json(args.root / "comparison" / "person_association_summary.json") or {}
    recommendation = summary.get("best_person_association_recommendation", {})
    print("verdict: %s" % recommendation.get("verdict"))
    print("best_run: %s" % recommendation.get("best_run"))
    for run in summary.get("runs", []):
        print(
            "%s track1_rows=%s person_frag=%s purity=%s false_merge=%s non_person_delta=%s merges=%s"
            % (
                run.get("run_name"),
                run.get("track1_rows"),
                run.get("person_fragmentation_approx"),
                run.get("global_purity_mean"),
                run.get("false_merge_rate"),
                run.get("vs_v2_non_person_rows_delta"),
                run.get("applied_merge_mapping_size"),
            )
        )


if __name__ == "__main__":
    main()

