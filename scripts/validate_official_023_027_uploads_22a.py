"""Validate frozen official 023-027 candidates."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.official_023_027.official_config import VARIANT_KEYS, frozen_variant_root, load_official_config
from deep_oc_sort_3d.official_023_027.official_track1_validator import validate_and_write


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate official 023-027 uploads")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    config = load_official_config(args.config)
    failed = []
    for variant in VARIANT_KEYS:
        root = frozen_variant_root(config, variant)
        report = validate_and_write(root / "track1.txt", root / "validation_summary.json", config, progress=args.progress)
        print("%s status=%s rows=%s errors=%s" % (variant, report.get("status"), report.get("total_rows"), report.get("num_errors")))
        if report.get("status") != "ok":
            failed.append(variant)
    if failed:
        raise RuntimeError("Official validation failed for: %s" % ", ".join(failed))


if __name__ == "__main__":
    main()
