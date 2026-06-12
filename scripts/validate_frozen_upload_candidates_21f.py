"""Revalidate frozen Step 21F candidates without rebuilding the pipeline."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.freeze_upload.freeze_config import load_freeze_config
from deep_oc_sort_3d.freeze_upload.freeze_manager import validate_frozen_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen Step 21F candidates")
    parser.add_argument("--config", type=Path, required=True)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for CLI consistency; validation summaries are refreshed.")
    parser.add_argument("--skip-existing", action="store_true", help="Accepted for CLI consistency.")
    args = parser.parse_args()
    config = load_freeze_config(args.config)
    result = validate_frozen_candidates(config, progress=args.progress)
    for row in result.get("candidates", []):
        validation = row.get("validation", {})
        print("%s status=%s errors=%s" % (row.get("candidate_name"), validation.get("status"), validation.get("num_errors")))


if __name__ == "__main__":
    main()
