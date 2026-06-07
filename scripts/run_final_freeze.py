"""Run the final baseline freeze collection pipeline."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.final_freeze.baseline_manifest import build_final_outputs_manifest
from deep_oc_sort_3d.final_freeze.figure_collector import collect_final_figures_from_config
from deep_oc_sort_3d.final_freeze.final_package import package_final_outputs_from_config
from deep_oc_sort_3d.final_freeze.freeze_config import output_root_from_config
from deep_oc_sort_3d.final_freeze.freeze_io import load_yaml, write_json
from deep_oc_sort_3d.final_freeze.freeze_report import write_final_reports_from_config
from deep_oc_sort_3d.final_freeze.metric_collector import collect_final_metrics_from_config
from deep_oc_sort_3d.final_freeze.table_builder import build_final_tables_from_config


def run_final_freeze(config_path: Path, progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Run all final freeze steps."""
    config = load_yaml(config_path)
    output_root = output_root_from_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_final_outputs_manifest(config, show_progress=progress)
    metrics = collect_final_metrics_from_config(config_path, show_progress=progress)
    tables = build_final_tables_from_config(config_path, show_progress=progress)
    figures = collect_final_figures_from_config(config_path, show_progress=progress)
    reports = write_final_reports_from_config(config_path, show_progress=progress)
    packages = package_final_outputs_from_config(config_path, show_progress=progress, overwrite=overwrite)
    summary = {
        "config_path": str(config_path),
        "output_root": str(output_root),
        "manifest_files": len(manifest.get("files", [])),
        "baseline_rows": len(metrics.get("baseline_rows", [])),
        "figures": figures.get("num_figures", 0),
        "packages": len(packages.get("packages", [])),
        "tables": tables.get("tables", {}),
        "reports": reports.get("reports", {}),
    }
    write_json(summary, output_root / "summaries" / "final_freeze_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final baseline freeze.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/final_freeze.yaml")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    summary = run_final_freeze(Path(args.config), progress=bool(args.progress), overwrite=bool(args.overwrite))
    print("Final freeze output:", summary.get("output_root"))
    print("manifest files:", summary.get("manifest_files"))
    print("baseline rows:", summary.get("baseline_rows"))
    print("figures:", summary.get("figures"))
    print("packages:", summary.get("packages"))


if __name__ == "__main__":
    main()

