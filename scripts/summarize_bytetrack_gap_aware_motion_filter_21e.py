"""Print a compact summary from existing Step 21E outputs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_report import summarize_existing_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Step 21E motion-filter outputs")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = summarize_existing_output(args.root)
    selected = value.get("selected", {})
    verdict = value.get("verdict", {})
    print("verdict: %s" % verdict.get("label", selected.get("verdict")))
    print("selected_variant: %s" % selected.get("selected_variant"))
    print("recommended_step_21f: %s" % verdict.get("recommended_step_21f", selected.get("recommended_step_21f")))
    for row in value.get("rows", []):
        print(
            "%s retention=%s track1_rows=%s errors=%s purity=%s false_merge=%s fragmentation=%s"
            % (
                row.get("variant_name"),
                row.get("motion_clean_retention"),
                row.get("track1_rows"),
                row.get("track1_validation_errors"),
                row.get("global_purity_mean"),
                row.get("false_merge_rate"),
                row.get("fragmentation_approx"),
            )
        )


if __name__ == "__main__":
    main()

