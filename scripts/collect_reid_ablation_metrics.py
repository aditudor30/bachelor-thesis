"""CLI for collecting Step 16C ReID ablation metrics."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.reid_ablation.ablation_metric_loader import collect_reid_ablation_metrics_from_config


def main() -> None:
    """Collect metrics only."""
    parser = argparse.ArgumentParser(description="Collect ReID ablation metrics.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()
    _unused_overwrite = args.overwrite
    result = collect_reid_ablation_metrics_from_config(args.config, progress=args.progress)
    print(json.dumps({"output_root": result.get("output_root"), "variants": len(result.get("variants", []))}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

