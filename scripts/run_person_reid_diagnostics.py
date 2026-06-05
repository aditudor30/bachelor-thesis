"""CLI for Step 16A Person ReID diagnostics."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.person_reid.reid_config import load_person_reid_config
from deep_oc_sort_3d.person_reid.reid_diagnostics import run_person_reid_diagnostics_from_config
from deep_oc_sort_3d.person_reid.reid_report import write_person_reid_report


def main() -> None:
    """Run diagnostics."""
    parser = argparse.ArgumentParser(description="Run Person ReID diagnostics.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()
    config = load_person_reid_config(args.config)
    summary = run_person_reid_diagnostics_from_config(config, show_progress=args.progress, overwrite=args.overwrite)
    write_person_reid_report(Path(str(config.get("reid_person", {}).get("output_root", "output/reid_person/baseline_v2_pseudo3d_fullcam"))))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

