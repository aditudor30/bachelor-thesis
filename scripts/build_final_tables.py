"""Build final freeze report tables."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.final_freeze.table_builder import build_final_tables_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final freeze tables.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/final_freeze.yaml")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    result = build_final_tables_from_config(Path(args.config), show_progress=bool(args.progress))
    print("tables:")
    for key, value in sorted(result.get("tables", {}).items()):
        print("  %s: %s" % (key, value))


if __name__ == "__main__":
    main()

