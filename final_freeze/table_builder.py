"""Build final freeze tables for reports and thesis writing."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze.freeze_config import output_root_from_config
from deep_oc_sort_3d.final_freeze.freeze_io import (
    NOT_AVAILABLE,
    load_yaml,
    read_json,
    write_csv_rows,
    write_json,
)
from deep_oc_sort_3d.final_freeze.metric_collector import collect_final_metrics_from_config


FINAL_COLUMNS = [
    "variant_name",
    "role",
    "track1_valid",
    "track1_errors",
    "track1_rows",
    "pseudo3d_used_rate",
    "global_tracks",
    "multi_camera_tracks",
    "global_purity",
    "false_merge_rate",
    "fragmentation_approx",
    "reid_used",
]


def build_final_tables_from_config(config_path: Path, show_progress: bool = True) -> Dict[str, Any]:
    """Build CSV, Markdown, and LaTeX tables for the final freeze."""
    config = load_yaml(config_path)
    output_root = output_root_from_config(config)
    bundle = _load_or_collect_metrics(config_path, output_root, show_progress=show_progress)
    baseline_rows = list(bundle.get("baseline_rows", []))
    track1_rows = list(bundle.get("track1", []))
    pseudo3d_rows = [bundle.get("pseudo3d", {})]
    reid_rows = [bundle.get("reid", {})]

    tables_root = output_root / "tables"
    paths: Dict[str, str] = {}
    paths["final_baseline_comparison_csv"] = str(tables_root / "final_baseline_comparison.csv")
    paths["final_baseline_comparison_md"] = str(tables_root / "final_baseline_comparison.md")
    paths["final_baseline_comparison_tex"] = str(tables_root / "final_baseline_comparison.tex")
    paths["track1_validation_summary_tex"] = str(tables_root / "track1_validation_summary.tex")
    paths["pseudo3d_summary_tex"] = str(tables_root / "pseudo3d_summary.tex")
    paths["reid_summary_tex"] = str(tables_root / "reid_summary.tex")
    paths["report_tables_md"] = str(tables_root / "final_report_tables.md")

    write_csv_rows(baseline_rows, Path(paths["final_baseline_comparison_csv"]), fieldnames=FINAL_COLUMNS)
    _write_text(
        rows_to_markdown_table(baseline_rows, FINAL_COLUMNS),
        Path(paths["final_baseline_comparison_md"]),
    )
    _write_text(
        rows_to_latex_table(
            baseline_rows,
            FINAL_COLUMNS,
            caption="Final baseline comparison.",
            label="tab:final-baseline-comparison",
        ),
        Path(paths["final_baseline_comparison_tex"]),
    )
    _write_text(
        rows_to_latex_table(
            track1_rows,
            ["variant_name", "role", "track1_valid", "track1_errors", "track1_rows"],
            caption="Track1 submission validation summary.",
            label="tab:track1-validation-summary",
        ),
        Path(paths["track1_validation_summary_tex"]),
    )
    _write_text(
        rows_to_latex_table(
            pseudo3d_rows,
            ["pseudo3d_used_rate", "fallback_original_used_rate", "source_provenance"],
            caption="Pseudo3D provenance summary.",
            label="tab:pseudo3d-summary",
        ),
        Path(paths["pseudo3d_summary_tex"]),
    )
    _write_text(
        rows_to_latex_table(
            reid_rows,
            ["model", "embedding_dim", "crop_embeddings", "global_fragment_embeddings", "top1_retrieval", "top5_retrieval", "verdict"],
            caption="ReID diagnostic summary.",
            label="tab:reid-summary",
        ),
        Path(paths["reid_summary_tex"]),
    )
    _write_text(_build_report_tables_markdown(bundle), Path(paths["report_tables_md"]))
    write_json({"tables": paths}, tables_root / "final_table_manifest.json")
    return {"tables": paths, "baseline_rows": baseline_rows}


def rows_to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """Render rows as a Markdown table."""
    if not rows:
        return "_No rows available._\n"
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join([format_metric(row.get(column)) for column in columns]) + " |")
    return "\n".join(lines) + "\n"


def rows_to_latex_table(rows: List[Dict[str, Any]], columns: List[str], caption: str = "", label: str = "") -> str:
    """Render rows as a compact LaTeX tabular environment."""
    align = "l" * len(columns)
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small"]
    lines.append("\\begin{tabular}{%s}" % align)
    lines.append("\\hline")
    lines.append(" & ".join([_latex_escape(column) for column in columns]) + " \\\\")
    lines.append("\\hline")
    if rows:
        for row in rows:
            lines.append(" & ".join([_latex_escape(format_metric(row.get(column))) for column in columns]) + " \\\\")
    else:
        lines.append(" & ".join([NOT_AVAILABLE] * len(columns)) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    if caption:
        lines.append("\\caption{%s}" % _latex_escape(caption))
    if label:
        lines.append("\\label{%s}" % _latex_escape(label))
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def format_metric(value: Any) -> str:
    """Format a metric for human-readable tables."""
    if value in (None, "", NOT_AVAILABLE):
        return NOT_AVAILABLE
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return "%.0f" % value
        return "%.6f" % value
    return str(value)


def _load_or_collect_metrics(config_path: Path, output_root: Path, show_progress: bool = True) -> Dict[str, Any]:
    bundle_path = output_root / "tables" / "final_metrics_bundle.json"
    bundle = read_json(bundle_path)
    if bundle is None:
        bundle = collect_final_metrics_from_config(config_path, show_progress=show_progress)
    return bundle


def _build_report_tables_markdown(bundle: Dict[str, Any]) -> str:
    sections = [
        "# Final Report Tables",
        "",
        "## Baseline Comparison",
        rows_to_markdown_table(list(bundle.get("baseline_rows", [])), FINAL_COLUMNS),
        "## Track1 Validation",
        rows_to_markdown_table(list(bundle.get("track1", [])), ["variant_name", "role", "track1_valid", "track1_errors", "track1_rows"]),
        "## Pseudo3D Summary",
        rows_to_markdown_table([bundle.get("pseudo3d", {})], ["pseudo3d_used_rate", "fallback_original_used_rate", "source_provenance"]),
        "## ReID Summary",
        rows_to_markdown_table([bundle.get("reid", {})], ["model", "embedding_dim", "crop_embeddings", "global_fragment_embeddings", "top1_retrieval", "top5_retrieval", "verdict"]),
    ]
    return "\n".join(sections)


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text
