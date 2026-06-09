"""Print a compact summary of final freeze v2 outputs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_io import read_json


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summary = read_json(root / "summaries" / "final_freeze_v2_summary.json") or {}
    manifest = read_json(root / "manifests" / "final_outputs_manifest.json") or {}
    tables = read_json(root / "tables" / "final_metrics_bundle.json") or {}
    print("root: %s" % root)
    print("status: %s" % summary.get("status"))
    print("variant_rows: %s" % len(tables.get("variants", [])))
    print("manifest_artifacts: %s" % len(manifest.get("artifacts", [])))
    print("final_verdict: %s" % summary.get("final_verdict"))
    print("reports_dir: %s" % (root / "reports"))
    print("packages_dir: %s" % (root / "packages"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final freeze v2.")
    parser.add_argument("--root", type=Path, default=Path("output/final_freeze_v2"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
