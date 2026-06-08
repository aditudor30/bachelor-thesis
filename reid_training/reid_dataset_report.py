"""Markdown report for the SmartSpaces Person ReID dataset."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.reid_training.reid_dataset_config import output_root_from_config
from deep_oc_sort_3d.reid_training.reid_dataset_io import read_json, write_json


def write_reid_dataset_report_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Write report and README files."""
    output_root = output_root_from_config(config)
    summary = read_json(output_root / "diagnostics" / "dataset_summary.json") or {}
    warnings = read_json(output_root / "diagnostics" / "warnings.json") or {}
    report = build_report_markdown(config, summary, warnings)
    readme = build_readme_markdown(config, summary)
    report_path = output_root / "reports" / "PERSON_REID_DATASET_REPORT.md"
    readme_path = output_root / "reports" / "README_PERSON_REID_DATASET.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    readme_path.write_text(readme, encoding="utf-8")
    result = {"report": str(report_path), "readme": str(readme_path), "verdict": summary.get("verdict")}
    write_json(result, output_root / "reports" / "report_summary.json")
    return result


def build_report_markdown(config: Dict[str, Any], summary: Dict[str, Any], warnings: Dict[str, Any]) -> str:
    """Build report Markdown."""
    lines = [
        "# SmartSpaces Person ReID Dataset Report",
        "",
        "## Executive Summary",
        "",
        "- Verdict: `%s`." % summary.get("verdict"),
        "- Total valid crops: `%s`." % summary.get("total_crops"),
        "- Train crops / identities: `%s / %s`." % (summary.get("train_crops"), summary.get("num_train_identities")),
        "- Val crops / identities: `%s / %s`." % (summary.get("val_crops"), summary.get("num_val_identities")),
        "- Train triplets: `%s`." % summary.get("triplets_train"),
        "",
        "## Settings",
        "",
        "- Dataset root: `%s`." % config.get("paths", {}).get("dataset_root"),
        "- Split strategy: `%s`." % config.get("split", {}).get("split_strategy"),
        "- Frame stride: `%s`." % config.get("crop_extraction", {}).get("frame_stride"),
        "- Max crops per identity: `%s`." % config.get("crop_extraction", {}).get("max_crops_per_identity"),
        "",
        "## Distributions",
        "",
        "- Crops per identity: `%s`." % summary.get("crops_per_identity"),
        "- Scene distribution: `%s`." % summary.get("scene_distribution"),
        "- Camera distribution: `%s`." % summary.get("camera_distribution"),
        "",
        "## Warnings",
        "",
        "- Rare identity count: `%s`." % warnings.get("rare_identity_count"),
        "- Train/val identity overlap: `%s`." % warnings.get("train_val_identity_overlap"),
        "- Invalid crop count: `%s`." % warnings.get("invalid_crop_count"),
        "- Invalid reasons: `%s`." % warnings.get("invalid_reasons"),
        "",
        "## Visual Checks",
        "",
        "Inspect the grids in `figures/` before using this dataset for ReID fine-tuning.",
        "",
        "## Recommended Next Step",
        "",
        "Proceed to Step 18B only if the verdict is `reid_dataset_ready` or `reid_dataset_usable_with_warnings`, the crop grids look correct, and train/val identity overlap is zero.",
    ]
    return "\n".join(lines) + "\n"


def build_readme_markdown(config: Dict[str, Any], summary: Dict[str, Any]) -> str:
    """Build dataset README."""
    _unused = config
    lines = [
        "# SmartSpaces Person ReID Dataset",
        "",
        "This directory contains a domain-specific Person ReID crop dataset generated from SmartSpaces train/val ground truth.",
        "",
        "Important files:",
        "",
        "- `metadata/all_crops.csv`",
        "- `metadata/train_split.csv`",
        "- `metadata/val_split.csv`",
        "- `pairs_triplets/triplets_train.csv`",
        "- `diagnostics/dataset_summary.json`",
        "- `figures/sample_crops_grid_train.png`",
        "",
        "Current verdict: `%s`." % summary.get("verdict"),
    ]
    return "\n".join(lines) + "\n"

