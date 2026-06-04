"""Run Step 15G full-camera pseudo-3D generation and audit."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deep_oc_sort_3d.audit3d.audit3d_io import write_csv, write_json, write_markdown
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_coverage_audit import audit_fullcam_coverage, write_coverage_reports
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_discovery import (
    audit_existing_predictions,
    discover_required_camera_files,
    filter_fullcam_items,
    missing_prediction_rows,
    write_fullcam_items_csv,
    write_fullcam_items_json,
)
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_raw_generation import generate_raw_for_items
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_report import build_fullcam_generation_report
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_stabilization import stabilize_items
from deep_oc_sort_3d.pseudo3d_fullcam.fullcam_summary import summarize_step15g, write_step15g_summary


def run(args: Any) -> Dict[str, Any]:
    """Run discovery, raw generation, stabilization, coverage audit, and summary."""
    cfg = _load_yaml(args.config)
    _apply_flags(cfg, args)
    output_root = _output_root(cfg)
    discovery_root = output_root / "discovery"

    items = discover_required_camera_files(cfg)
    items = filter_fullcam_items(items, _csv_arg(args.subsets), _csv_arg(args.scenes), _csv_arg(args.camera_ids), args.max_cameras)
    write_fullcam_items_json(items, discovery_root / "required_camera_files.json")
    write_fullcam_items_csv(items, discovery_root / "required_camera_files.csv")
    discovery_audit = audit_existing_predictions(items)
    write_json(discovery_audit, discovery_root / "existing_raw_predictions_audit.json")
    write_json(discovery_audit, discovery_root / "existing_stabilized_predictions_audit.json")
    write_csv(missing_prediction_rows(items), discovery_root / "missing_camera_files.csv")

    raw_summary = generate_raw_for_items(items, cfg, show_progress=args.progress)
    stabilization_summary = stabilize_items(items, cfg, show_progress=args.progress)
    coverage_summary = audit_fullcam_coverage(items, output_root, cfg, show_progress=args.progress)
    write_coverage_reports(coverage_summary, output_root)
    summary = summarize_step15g(output_root, cfg.get("coverage", {}))
    write_step15g_summary(summary, output_root)
    write_markdown(build_fullcam_generation_report(summary), output_root / "report" / "PSEUDO3D_FULLCAM_GENERATION_REPORT.md")
    write_json(
        {
            "discovery": discovery_audit,
            "raw_generation": raw_summary,
            "stabilization": stabilization_summary,
            "coverage": _compact_coverage(coverage_summary),
            "summary": summary,
        },
        output_root / "report" / "PSEUDO3D_STEP15G_RUNBOOK_SUMMARY.json",
    )
    print("Step 15G recommendation: %s" % summary.get("recommendation"))
    return summary


def _apply_flags(cfg: Dict[str, Any], args: Any) -> None:
    if args.overwrite:
        cfg.setdefault("generation", {})["overwrite"] = True
        cfg.setdefault("generation", {})["skip_existing"] = False
        cfg.setdefault("stabilization", {})["overwrite"] = True
        cfg.setdefault("stabilization", {})["skip_existing"] = False
    if args.skip_existing is not None:
        cfg.setdefault("generation", {})["skip_existing"] = bool(args.skip_existing)
        cfg.setdefault("stabilization", {})["skip_existing"] = bool(args.skip_existing)


def _compact_coverage(summary: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(summary)
    out.pop("per_camera", None)
    out.pop("projection_rows", None)
    out.pop("smoothness_rows", None)
    out.pop("worst_remaining_jumps", None)
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    if data.get("include_config"):
        return _load_yaml(Path(data["include_config"]))
    return data


def _output_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("step15g", {}).get("output_root", "output/pseudo3d/baseline_v2_pseudo3d_fullcam"))


def _csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 15G full-camera pseudo-3D generation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument("--subsets", default=None)
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--camera-ids", default=None)
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    skip_group.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True, skip_existing=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
