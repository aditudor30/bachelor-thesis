"""Build final freeze v2 Markdown reports."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_config import output_root_from_config
from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_io import load_yaml, read_json, write_json, write_text
from deep_oc_sort_3d.final_freeze_v2.final_metric_loader import FINAL_VARIANT_COLUMNS, collect_final_freeze_v2_metrics_from_config
from deep_oc_sort_3d.final_freeze_v2.final_table_builder import rows_to_markdown_table


def write_final_freeze_v2_reports_from_config(config_path: Path, show_progress: bool = True) -> Dict[str, Any]:
    """Write all final reports."""
    config = load_yaml(Path(config_path))
    output_root = output_root_from_config(config)
    bundle = read_json(output_root / "tables" / "final_metrics_bundle.json")
    if bundle is None:
        bundle = collect_final_freeze_v2_metrics_from_config(config_path, show_progress=show_progress)
    paths = write_final_reports(config, bundle)
    write_json({"reports": paths}, output_root / "reports" / "report_manifest.json")
    return {"reports": paths}


def write_final_reports(config: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, str]:
    """Write report files."""
    root = output_root_from_config(config) / "reports"
    paths = {
        "freeze": root / "FINAL_PROJECT_FREEZE_REPORT.md",
        "results": root / "FINAL_RESULTS_SUMMARY.md",
        "reid": root / "FINAL_REID_SUMMARY.md",
        "limitations": root / "FINAL_LIMITATIONS_AND_FUTURE_WORK.md",
        "reproducibility": root / "FINAL_REPRODUCIBILITY.md",
        "figures": root / "FINAL_THESIS_FIGURES_GUIDE.md",
    }
    write_text(build_project_freeze_report(bundle), paths["freeze"])
    write_text(build_results_summary(bundle), paths["results"])
    write_text(build_reid_summary(bundle), paths["reid"])
    write_text(build_limitations_report(), paths["limitations"])
    write_text(build_reproducibility_report(config), paths["reproducibility"])
    write_text(build_figures_guide(config), paths["figures"])
    return {key: str(value) for key, value in paths.items()}


def build_project_freeze_report(bundle: Dict[str, Any]) -> List[str]:
    """Build main freeze report."""
    variants = list(bundle.get("variants", []))
    return [
        "# Final Project Freeze Report",
        "",
        "## Final Verdict",
        "",
        "- `v1_geometry_only`: keep as submission-safe baseline.",
        "- `v2_pseudo3d_fullcam_current`: keep as main 3D MVP.",
        "- `v2_export_compact`: keep as safe compact variant.",
        "- `osnet_finetuned_threshold_080`: keep as ReID-only diagnostic.",
        "- `osnet_finetuned_combined_safe_080`: keep as experimental fine-tuned ReID final.",
        "",
        "## Interpretation",
        "",
        "V1 / V2 compact remain the safe fallback family. V2 pseudo3D fullcam is the main 3D contribution because it preserves explicit pseudo3D provenance. `combined_safe_080` is a ReID-enhanced experimental extension: promising, visually plausible, but not a complete replacement for the safe baseline.",
        "",
        "## Final Variant Comparison",
        "",
        rows_to_markdown_table(variants, FINAL_VARIANT_COLUMNS),
        "## Main Limitations",
        "",
        "- Person fragmentation remains the dominant limitation.",
        "- V2 offers 3D provenance and high pseudo3D coverage, but it has higher fragmentation than the compact baseline family.",
        "- Fine-tuned ReID produces controlled gains and visual signal, but final use should remain experimental.",
    ]


def build_results_summary(bundle: Dict[str, Any]) -> List[str]:
    """Build final results summary."""
    pseudo = bundle.get("pseudo3d", {})
    variants = list(bundle.get("variants", []))
    return [
        "# Final Results Summary",
        "",
        "## Pseudo3D MVP",
        "",
        "- pseudo3D used rate: `%s`" % pseudo.get("pseudo3d_used_rate"),
        "- fallback original rate: `%s`" % pseudo.get("fallback_original_used_rate"),
        "- Track1 validity: valid for the frozen V2 output.",
        "",
        "V2 offers 3D provenance and pseudo3D coverage ridicat, but has larger Person fragmentation than the compact submission-safe baseline.",
        "",
        "## Frozen Variants",
        "",
        rows_to_markdown_table(variants, ["variant_name", "role", "track1_valid", "track1_rows", "person_fragmentation", "global_purity", "false_merge_rate", "final_recommendation"]),
    ]


def build_reid_summary(bundle: Dict[str, Any]) -> List[str]:
    """Build final ReID summary."""
    training = list(bundle.get("reid_training", []))
    association = list(bundle.get("reid_association", []))
    visual = bundle.get("reid_visual", {})
    return [
        "# Final ReID Summary",
        "",
        "## ReID Dataset And Training",
        "",
        rows_to_markdown_table(training, ["stage", "verdict", "valid_crops", "train_crops", "val_crops", "finetuned_top1", "top1_gain", "notes"]),
        "## Association",
        "",
        rows_to_markdown_table(association, ["stage", "variant", "track1_valid", "fragment_coverage", "person_fragmentation_delta", "false_merge_delta", "purity_delta", "conclusion"]),
        "## Visual Decision",
        "",
        "- total merge events: `%s`" % visual.get("total_merge_events"),
        "- reviewed events: `%s`" % visual.get("review_events"),
        "- auto labels: `%s`" % visual.get("auto_label_counts"),
        "- final decision: `%s`" % visual.get("final_decision"),
        "",
        "OSNet off-the-shelf nu a produs un castig final clar, dar fine-tuning-ul pe crop-uri Person SmartSpaces a imbunatatit retrieval-ul si a produs castiguri controlate in association. Varianta combined_safe_080 este pastrata ca extensie experimentala ReID-enhanced, nu ca inlocuitor complet pentru baseline-ul sigur.",
    ]


def build_limitations_report() -> List[str]:
    """Build limitations and future work report."""
    return [
        "# Final Limitations And Future Work",
        "",
        "## Limitations",
        "",
        "- Person fragmentation remains high and dominates final error analysis.",
        "- Fine-tuned ReID improves retrieval but should be used conservatively in final association.",
        "- The BEV visualizations are coordinate-space diagnostics, not map-aligned final maps.",
        "- Test split does not use GT or depth maps; all test outputs must remain RGB/calibration/map-only.",
        "",
        "## Future Work",
        "",
        "- Train the crop-based 3D head and replace pseudo3D heuristics with learned 3D predictions where validated.",
        "- Add learned camera-pair transition priors and class-specific association thresholds.",
        "- Continue domain tuning of Person ReID with stronger hard-negative mining.",
        "- Improve compact export policies for short low-confidence Person fragments without harming non-Person recall.",
    ]


def build_reproducibility_report(config: Dict[str, Any]) -> List[str]:
    """Build reproducibility report."""
    paths = config.get("paths", {})
    lines = [
        "# Final Reproducibility",
        "",
        "The freeze step does not rerun detector inference, tracking, association, ReID extraction, fine-tuning, or Track1 export. It only reads existing outputs and writes final tables, reports, figures, manifests and compact packages.",
        "",
        "```bash",
        "python -m deep_oc_sort_3d.scripts.run_final_freeze_v2 \\",
        "  --config deep_oc_sort_3d/configs/final_freeze_v2.yaml \\",
        "  --progress \\",
        "  --overwrite",
        "```",
        "",
        "## Source Paths",
        "",
    ]
    for key in sorted(paths.keys()):
        lines.append("- `%s`: `%s`" % (key, paths.get(key)))
    return lines


def build_figures_guide(config: Dict[str, Any]) -> List[str]:
    """Build thesis figures guide."""
    _unused = config
    return [
        "# Final Thesis Figures Guide",
        "",
        "Use `output/final_freeze_v2/figures/captions/FIGURE_CAPTIONS.md` as the caption index.",
        "",
        "- Pipeline diagram: use for method overview.",
        "- RGB-Depth alignment: use to justify depth supervision and geometry diagnostics.",
        "- 2D tracking qualitative: use to show detector/local-tracking behavior.",
        "- Pseudo3D cuboids: use as diagnostic, not as perfect geometric truth.",
        "- BEV: Coordinate-space BEV visualization of estimated 3D trajectories. The plot is not map-aligned; percentile clipping is used only for visualization.",
        "- ReID panels: use to show plausible fine-tuned Person merges and ambiguous cases.",
        "- Final comparison chart: use to explain V1/V2/ReID trade-offs.",
    ]

