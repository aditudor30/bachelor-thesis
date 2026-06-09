"""Run the full final freeze v2 packaging/reporting step."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.final_freeze_v2.final_figure_collector import collect_final_freeze_v2_figures_from_config
from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_config import load_final_freeze_v2_config, output_root_from_config, prepare_output_root, save_resolved_config
from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_io import write_json
from deep_oc_sort_3d.final_freeze_v2.final_manifest_builder import build_final_outputs_manifest_from_config
from deep_oc_sort_3d.final_freeze_v2.final_metric_loader import collect_final_freeze_v2_metrics_from_config
from deep_oc_sort_3d.final_freeze_v2.final_package_builder import package_final_freeze_v2_from_config
from deep_oc_sort_3d.final_freeze_v2.final_report_builder import write_final_freeze_v2_reports_from_config
from deep_oc_sort_3d.final_freeze_v2.final_table_builder import build_final_freeze_v2_tables_from_config


def run_final_freeze_v2(config_path: Path, progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Run Step 19 freeze without rerunning upstream pipelines."""
    config = load_final_freeze_v2_config(config_path)
    output_root = prepare_output_root(config, overwrite=overwrite)
    save_resolved_config(config, config_path, output_root)
    metrics = collect_final_freeze_v2_metrics_from_config(config_path, show_progress=progress)
    tables = build_final_freeze_v2_tables_from_config(config_path, show_progress=progress, overwrite=overwrite)
    reports = write_final_freeze_v2_reports_from_config(config_path, show_progress=progress)
    figures = collect_final_freeze_v2_figures_from_config(config_path, show_progress=progress, overwrite=overwrite)
    packages = package_final_freeze_v2_from_config(config_path, show_progress=progress, overwrite=overwrite)
    manifest = build_final_outputs_manifest_from_config(config_path, show_progress=progress)
    summary = {
        "status": "ok",
        "output_root": str(output_root),
        "variant_rows": len(metrics.get("variants", [])),
        "tables": tables,
        "reports": reports,
        "figures": {"num_figures": figures.get("num_figures")},
        "packages": {"num_packages": len(packages.get("packages", []))},
        "manifest_artifacts": len(manifest.get("artifacts", [])),
        "final_verdict": {
            "v1_geometry_only": "keep as submission-safe baseline",
            "v2_pseudo3d_fullcam_current": "keep as main 3D MVP",
            "v2_export_compact": "keep as safe compact variant",
            "threshold_080": "keep as ReID-only diagnostic",
            "combined_safe_080": "keep as experimental fine-tuned ReID final",
        },
    }
    write_json(summary, output_root / "summaries" / "final_freeze_v2_summary.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_final_freeze_v2(Path(args.config), progress=bool(args.progress), overwrite=bool(args.overwrite))
    print("status: %s" % summary.get("status"))
    print("output_root: %s" % summary.get("output_root"))
    print("variant_rows: %s" % summary.get("variant_rows"))
    print("manifest_artifacts: %s" % summary.get("manifest_artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final freeze v2.")
    parser.add_argument("--config", type=Path, default=Path("deep_oc_sort_3d/configs/final_freeze_v2.yaml"))
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()

