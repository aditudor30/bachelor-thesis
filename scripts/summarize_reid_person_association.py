"""CLI for printing a compact ReID-guided Person association summary."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.person_reid_association.reid_association_io import read_json


def main() -> None:
    """Print compact summary."""
    parser = argparse.ArgumentParser(description="Summarize ReID-guided Person association outputs.")
    parser.add_argument("--root", type=Path, default=Path("output/person_reid_association/baseline_v2_pseudo3d_fullcam"))
    args = parser.parse_args()
    summary = read_json(args.root / "comparison" / "reid_person_association_summary.json") or {}
    recommendation = summary.get("best_reid_person_association_recommendation", {})
    print("verdict: %s" % recommendation.get("verdict"))
    print("best_run: %s" % recommendation.get("best_run"))
    for run in summary.get("runs", []):
        print(
            "%s status=%s merges=%s person_frag=%s purity=%s false_merge=%s non_person_delta=%s reid_passing=%s"
            % (
                run.get("run_name"),
                run.get("run_status", "ok"),
                run.get("merges_applied"),
                run.get("person_fragmentation_approx"),
                run.get("global_purity_mean"),
                run.get("false_merge_rate"),
                run.get("vs_v2_non_person_rows_delta"),
                run.get("pairs_passing_reid_threshold"),
            )
        )


if __name__ == "__main__":
    main()
