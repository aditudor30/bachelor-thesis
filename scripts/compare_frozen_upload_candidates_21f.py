"""Compare existing frozen candidates and refresh the Step 21F report."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.freeze_upload.freeze_comparison import compare_frozen_candidates
from deep_oc_sort_3d.freeze_upload.freeze_config import load_freeze_config
from deep_oc_sort_3d.freeze_upload.freeze_report import write_freeze_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen Step 21F candidates")
    parser.add_argument("--config", type=Path, required=True)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for CLI consistency; comparison files are refreshed.")
    parser.add_argument("--skip-existing", action="store_true", help="Accepted for CLI consistency.")
    args = parser.parse_args()
    config = load_freeze_config(args.config)
    comparison = compare_frozen_candidates(config, progress=args.progress)
    report_path = write_freeze_report(config, comparison)
    print("verdict: %s" % comparison.get("verdict", {}).get("label"))
    print("report: %s" % report_path)


if __name__ == "__main__":
    main()
