"""Package final freeze outputs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.final_freeze.final_package import package_final_outputs_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Package final freeze outputs.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/final_freeze.yaml")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    result = package_final_outputs_from_config(Path(args.config), show_progress=bool(args.progress), overwrite=bool(args.overwrite))
    print("packages:", len(result.get("packages", [])))
    for package in result.get("packages", []):
        print("  %s: %s" % (package.get("name"), package.get("root")))


if __name__ == "__main__":
    main()

