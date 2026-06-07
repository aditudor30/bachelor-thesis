"""Generate final freeze Markdown reports."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze.freeze_config import output_root_from_config
from deep_oc_sort_3d.final_freeze.freeze_io import load_yaml, read_json, write_json
from deep_oc_sort_3d.final_freeze.metric_collector import collect_final_metrics_from_config
from deep_oc_sort_3d.final_freeze.table_builder import format_metric, rows_to_markdown_table


def write_final_reports_from_config(config_path: Path, show_progress: bool = True) -> Dict[str, Any]:
    """Write final freeze Markdown reports."""
    config = load_yaml(config_path)
    output_root = output_root_from_config(config)
    bundle = read_json(output_root / "tables" / "final_metrics_bundle.json")
    if bundle is None:
        bundle = collect_final_metrics_from_config(config_path, show_progress=show_progress)
    paths = write_final_reports(config, bundle)
    write_json({"reports": paths}, output_root / "reports" / "report_manifest.json")
    return {"reports": paths}


def write_final_reports(config: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, str]:
    """Write all final report drafts."""
    output_root = output_root_from_config(config)
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "freeze_report": reports_root / "FINAL_BASELINE_FREEZE_REPORT.md",
        "results_summary": reports_root / "FINAL_RESULTS_SUMMARY.md",
        "limitations": reports_root / "LIMITATIONS_AND_FUTURE_WORK.md",
        "reproducibility": reports_root / "REPRODUCIBILITY.md",
    }
    _write_text(build_freeze_report(bundle), paths["freeze_report"])
    _write_text(build_results_summary(bundle), paths["results_summary"])
    _write_text(build_limitations_report(bundle), paths["limitations"])
    _write_text(build_reproducibility_report(config), paths["reproducibility"])
    return {key: str(value) for key, value in paths.items()}


def build_freeze_report(bundle: Dict[str, Any]) -> str:
    """Build the main final freeze report."""
    rows = list(bundle.get("baseline_rows", []))
    reid = bundle.get("reid", {})
    pseudo3d = bundle.get("pseudo3d", {})
    lines = [
        "# Final Baseline Freeze Report",
        "",
        "## Recommendation",
        "",
        "- Keep V1 geometry-only as the submission-safe baseline.",
        "- Keep V2 pseudo3D fullcam as the 3D provenance-backed MVP.",
        "- Keep V2 export_compact as a safe compact variant.",
        "- Keep OSNet ReID as validated infrastructure, but do not claim final tracking gain without domain tuning.",
        "",
        "## Baseline Comparison",
        rows_to_markdown_table(
            rows,
            [
                "variant_name",
                "role",
                "track1_valid",
                "track1_rows",
                "pseudo3d_used_rate",
                "global_purity",
                "false_merge_rate",
                "fragmentation_approx",
            ],
        ),
        "## Pseudo3D Summary",
        "",
        "- Pseudo3D used rate: `%s`." % format_metric(pseudo3d.get("pseudo3d_used_rate")),
        "- Fallback original rate: `%s`." % format_metric(pseudo3d.get("fallback_original_used_rate")),
        "",
        "## ReID Summary",
        "",
        "- Backend/model: `%s`." % format_metric(reid.get("model")),
        "- Embedding dim: `%s`." % format_metric(reid.get("embedding_dim")),
        "- Retrieval top1/top5: `%s` / `%s`." % (format_metric(reid.get("top1_retrieval")), format_metric(reid.get("top5_retrieval"))),
        "- Verdict: `%s`." % format_metric(reid.get("verdict")),
    ]
    return "\n".join(lines) + "\n"


def build_results_summary(bundle: Dict[str, Any]) -> str:
    """Build a concise thesis-ready results summary."""
    rows = list(bundle.get("baseline_rows", []))
    lines = [
        "# Final Results Summary",
        "",
        "This summary freezes the final set of baselines and diagnostic variants for Track 1 reporting.",
        "",
        rows_to_markdown_table(rows, ["variant_name", "role", "track1_valid", "track1_rows", "multi_camera_tracks", "global_purity", "fragmentation_approx"]),
        "The main engineering outcome is a validated Track1 submission path plus a 3D provenance-backed MVP that preserves pseudo3D metadata and can be extended with stronger learned association later.",
    ]
    return "\n".join(lines) + "\n"


def build_limitations_report(bundle: Dict[str, Any]) -> str:
    """Build limitations and future work notes."""
    _unused = bundle
    lines = [
        "# Limitations And Future Work",
        "",
        "## Current Limitations",
        "",
        "- Person fragmentation remains the dominant failure mode.",
        "- ReID infrastructure is connected, but current OSNet embeddings do not yet provide reliable final association gains without domain tuning.",
        "- Pseudo3D fullcam coverage is strong, but the MVP still relies on detector and tracking quality upstream.",
        "- Non-person classes are usable in the MVP, but rare class behavior should remain part of future validation.",
        "",
        "## Future Work",
        "",
        "- Fine-tune ReID on SmartSpaces crops, especially for cross-camera person association.",
        "- Train the crop-based 3D head for class, depth, center, dimensions, yaw, and embedding prediction.",
        "- Add camera-pair-specific transition priors and learned edge scoring.",
        "- Improve export policies for short low-confidence tracks without harming non-person classes.",
        "- Replace diagnostic pseudo3D heuristics with model-predicted 3D outputs where validation supports it.",
    ]
    return "\n".join(lines) + "\n"


def build_reproducibility_report(config: Dict[str, Any]) -> str:
    """Build reproducibility notes with frozen config paths."""
    paths = config.get("paths", {})
    lines = [
        "# Reproducibility Notes",
        "",
        "Run the final freeze entrypoint after the upstream outputs already exist:",
        "",
        "```bash",
        "python -m deep_oc_sort_3d.scripts.run_final_freeze \\",
        "  --config deep_oc_sort_3d/configs/final_freeze.yaml \\",
        "  --progress \\",
        "  --overwrite",
        "```",
        "",
        "Frozen input roots:",
        "",
    ]
    for key in sorted(paths.keys()):
        lines.append("- `%s`: `%s`" % (key, paths.get(key)))
    lines.extend(
        [
            "",
            "The freeze step does not rerun training, detector inference, local tracking, global association, ReID extraction, or Track1 export. It only collects summaries, checks critical files, copies selected figures, writes reports, and creates compact packages.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

