"""Merge, remap, round and freeze official 023-027 Track1 files."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.official_023_027.official_config import load_official_config
from deep_oc_sort_3d.official_023_027.official_merge_builder import build_official_track1_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Build official 023-027 Track1 candidates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["incremental", "rerun_all"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    config = load_official_config(args.config)
    mode = args.mode or str(config.get("official_023_027", {}).get("mode", "incremental"))
    result = build_official_track1_candidates(config, mode=mode, progress=args.progress, overwrite=args.overwrite, skip_existing=args.skip_existing)
    for row in result.get("variants", []):
        print("%s status=%s rows=%s errors=%s" % (row.get("candidate_name", row.get("variant")), row.get("status"), row.get("rows"), row.get("validation_errors")))
    if result.get("compliance", {}).get("status") != "ok":
        raise RuntimeError("Official Track1 build failed compliance checks")


if __name__ == "__main__":
    main()
