"""Run the isolated official 023-027 extension pipeline."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.official_023_027.class_mapping_audit import audit_class_mapping
from deep_oc_sort_3d.official_023_027.official_comparison import compare_official_candidates
from deep_oc_sort_3d.official_023_027.official_config import load_official_config
from deep_oc_sort_3d.official_023_027.official_extension_runner import run_official_extension
from deep_oc_sort_3d.official_023_027.official_merge_builder import build_official_track1_candidates
from deep_oc_sort_3d.official_023_027.official_package_builder import package_official_candidates
from deep_oc_sort_3d.official_023_027.official_report import write_official_reports
from deep_oc_sort_3d.official_023_027.test_scene_audit import audit_test_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official 023-027 V2/V3 extension")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["incremental", "rerun_all"], default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    config = load_official_config(args.config)
    mode = args.mode or str(config.get("official_023_027", {}).get("mode", "incremental"))
    scene_audit = audit_test_scenes(config)
    mapping_audit = audit_class_mapping(config)
    if scene_audit.get("status") != "ok" or mapping_audit.get("status") != "ok":
        raise RuntimeError("Official 023-027 audit failed before extension")
    extension = run_official_extension(config, mode=mode, progress=args.progress, overwrite=args.overwrite, skip_existing=args.skip_existing)
    print("extension_status: %s" % extension.get("status"))
    if extension.get("status") != "ok":
        raise RuntimeError("Official 023-027 extension failed or produced missing outputs")
    if args.all:
        build = build_official_track1_candidates(config, mode=mode, progress=args.progress, overwrite=args.overwrite, skip_existing=args.skip_existing)
        packages = package_official_candidates(config, overwrite=args.overwrite, skip_existing=args.skip_existing)
        comparison = compare_official_candidates(config, progress=args.progress)
        reports = write_official_reports(config, comparison)
        print("compliance_status: %s" % build.get("compliance", {}).get("status"))
        print("packages: %s" % len(packages.get("packages", [])))
        print("verdict: %s" % comparison.get("verdict", {}).get("label"))
        print("report: %s" % reports.get("processing_report"))
        if build.get("compliance", {}).get("status") != "ok" or packages.get("status") != "ok":
            raise RuntimeError("Official 023-027 all-in-one run did not pass compliance")


if __name__ == "__main__":
    main()
