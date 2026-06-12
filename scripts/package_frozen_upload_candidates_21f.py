"""Build separately uploadable zip files for Step 21F candidates."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.freeze_upload.freeze_config import load_freeze_config
from deep_oc_sort_3d.freeze_upload.package_builder import package_frozen_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Package frozen Step 21F candidates")
    parser.add_argument("--config", type=Path, required=True)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    config = load_freeze_config(args.config)
    result = package_frozen_candidates(
        config,
        progress=args.progress,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        create_zip=bool(config.get("freeze_upload_candidates", {}).get("create_zip_packages", True)) and not args.no_zip,
    )
    for row in result.get("packages", []):
        print("%s status=%s zip=%s" % (row.get("candidate_name"), row.get("status"), row.get("zip_path")))


if __name__ == "__main__":
    main()
