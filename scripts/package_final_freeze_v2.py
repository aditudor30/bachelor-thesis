"""Build final freeze v2 packages only."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_config import load_final_freeze_v2_config, prepare_output_root, save_resolved_config
from deep_oc_sort_3d.final_freeze_v2.final_package_builder import package_final_freeze_v2_from_config


def main() -> None:
    args = parse_args()
    config = load_final_freeze_v2_config(args.config)
    output_root = prepare_output_root(config, overwrite=False)
    save_resolved_config(config, args.config, output_root)
    summary = package_final_freeze_v2_from_config(args.config, show_progress=bool(args.progress), overwrite=bool(args.overwrite))
    print("packages: %d" % len(summary.get("packages", [])))
    print("output_root: %s" % output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package final freeze v2 artifacts.")
    parser.add_argument("--config", type=Path, default=Path("deep_oc_sort_3d/configs/final_freeze_v2.yaml"))
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()

