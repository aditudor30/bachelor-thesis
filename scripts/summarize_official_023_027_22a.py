"""Build and print the final official 023-027 summary."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.official_023_027.official_comparison import compare_official_candidates
from deep_oc_sort_3d.official_023_027.official_config import load_official_config
from deep_oc_sort_3d.official_023_027.official_report import write_official_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize official 023-027 candidates")
    parser.add_argument("--root", type=Path, default=Path("output/official_023_027"))
    parser.add_argument("--config", type=Path, default=Path("deep_oc_sort_3d/configs/official_023_027_v2_v3_22a.yaml"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    config = load_official_config(args.config)
    config.setdefault("paths", {})["output_root"] = str(args.root)
    comparison = compare_official_candidates(config, progress=args.progress)
    reports = write_official_reports(config, comparison)
    print("verdict: %s" % comparison.get("verdict", {}).get("label"))
    for name, row in comparison.get("upload_readiness", {}).items():
        if isinstance(row, dict):
            print("%s ready=%s errors=%s scenes=%s zip=%s" % (name, row.get("ready"), row.get("validation_errors"), row.get("scene_ids"), row.get("zip_path")))
    print("report: %s" % reports.get("processing_report"))


if __name__ == "__main__":
    main()
