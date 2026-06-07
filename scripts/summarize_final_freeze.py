"""Print a concise final freeze summary."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.final_freeze.freeze_io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize final freeze outputs.")
    parser.add_argument("--root", default="output/final_freeze")
    args = parser.parse_args()
    root = Path(args.root)
    summary = read_json(root / "summaries" / "final_freeze_summary.json") or {}
    metrics = read_json(root / "tables" / "final_metrics_bundle.json") or {}
    manifest = read_json(root / "manifests" / "final_outputs_manifest.json") or {}
    print("Final freeze root:", root)
    print("summary:", _compact(summary))
    print("manifest files:", len(manifest.get("files", [])))
    print("baseline rows:")
    for row in metrics.get("baseline_rows", []):
        print(
            "  %s valid=%s rows=%s purity=%s fragmentation=%s"
            % (
                row.get("variant_name"),
                row.get("track1_valid"),
                row.get("track1_rows"),
                row.get("global_purity"),
                row.get("fragmentation_approx"),
            )
        )


def _compact(data: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["output_root", "manifest_files", "baseline_rows", "figures", "packages"]
    return {key: data.get(key) for key in keys if key in data}


if __name__ == "__main__":
    main()

