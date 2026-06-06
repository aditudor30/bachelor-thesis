"""CLI for Step 16C ReID ablation cleanup and final decision."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.reid_ablation.ablation_decision import run_reid_ablation_decision
from deep_oc_sort_3d.reid_ablation.ablation_report import write_reid_ablation_report


def main() -> None:
    """Run the full Step 16C workflow."""
    parser = argparse.ArgumentParser(description="Run ReID ablation cleanup and final decision report.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()
    result = run_reid_ablation_decision(args.config, progress=args.progress, overwrite=args.overwrite)
    write_reid_ablation_report(result, Path(str(result.get("output_root"))))
    compact = {
        "output_root": result.get("output_root"),
        "verdicts": result.get("decision", {}).get("verdicts", []),
        "kept_variants": result.get("decision", {}).get("kept_variants", {}),
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

