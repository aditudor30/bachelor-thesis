"""Package official 023-027 V2/V3 candidates separately."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.official_023_027.official_config import load_official_config
from deep_oc_sort_3d.official_023_027.official_package_builder import package_official_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Package official 023-027 uploads")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    config = load_official_config(args.config)
    if args.no_zip:
        print("zip packaging disabled by --no-zip")
        return
    result = package_official_candidates(config, overwrite=args.overwrite, skip_existing=args.skip_existing)
    for row in result.get("packages", []):
        print("%s status=%s size_mb=%s zip=%s" % (row.get("variant"), row.get("status"), row.get("zip_size_mb"), row.get("zip_path")))
    if result.get("status") != "ok":
        raise RuntimeError("One or more official upload packages failed verification")


if __name__ == "__main__":
    main()
