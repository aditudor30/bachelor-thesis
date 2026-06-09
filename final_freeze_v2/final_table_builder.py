"""Build final freeze v2 tables."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_config import output_root_from_config
from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_io import NOT_AVAILABLE, load_yaml, read_json, write_csv_rows, write_json
from deep_oc_sort_3d.final_freeze_v2.final_metric_loader import FINAL_VARIANT_COLUMNS, collect_final_freeze_v2_metrics_from_config


def build_final_freeze_v2_tables_from_config(config_path: Path, show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Build final CSV and LaTeX tables."""
    _unused_overwrite = overwrite
    config = load_yaml(Path(config_path))
    output_root = output_root_from_config(config)
    bundle = read_json(output_root / "tables" / "final_metrics_bundle.json")
    if bundle is None:
        bundle = collect_final_freeze_v2_metrics_from_config(config_path, show_progress=show_progress)
    variants = list(bundle.get("variants", []))
    tables_root = output_root / "tables"
    write_csv_rows(variants, tables_root / "final_variant_comparison.csv", FINAL_VARIANT_COLUMNS)
    latex = rows_to_latex_table(
        variants,
        FINAL_VARIANT_COLUMNS,
        caption="Final frozen variant comparison for Track 1.",
        label="tab:final-variant-comparison",
    )
    (tables_root / "final_variant_comparison_latex.tex").write_text(latex, encoding="utf-8")
    write_csv_rows(list(bundle.get("track1", [])), tables_root / "final_track1_validation_summary.csv")
    write_csv_rows([bundle.get("pseudo3d", {})], tables_root / "final_pseudo3d_summary.csv")
    write_csv_rows(list(bundle.get("reid_training", [])), tables_root / "final_reid_training_summary.csv")
    write_csv_rows(list(bundle.get("reid_association", [])), tables_root / "final_reid_association_summary.csv")
    write_csv_rows([bundle.get("reid_visual", {})], tables_root / "final_reid_visual_decision_summary.csv")
    write_csv_rows(list(bundle.get("deltas", [])), tables_root / "final_metric_deltas.csv")
    write_json({"tables": sorted([str(path) for path in tables_root.glob("*") if path.is_file()])}, tables_root / "final_table_manifest.json")
    return {"tables_root": str(tables_root), "variant_rows": len(variants)}


def rows_to_latex_table(rows: List[Dict[str, Any]], columns: List[str], caption: str = "", label: str = "") -> str:
    """Render rows as a simple LaTeX table."""
    align = "l" * len(columns)
    lines = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\begin{tabular}{%s}" % align, "\\hline"]
    lines.append(" & ".join([latex_escape(column) for column in columns]) + " \\\\")
    lines.append("\\hline")
    if rows:
        for row in rows:
            lines.append(" & ".join([latex_escape(format_metric(row.get(column))) for column in columns]) + " \\\\")
    else:
        lines.append(" & ".join([NOT_AVAILABLE] * len(columns)) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    if caption:
        lines.append("\\caption{%s}" % latex_escape(caption))
    if label:
        lines.append("\\label{%s}" % latex_escape(label))
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def rows_to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """Render rows as Markdown table."""
    if not rows:
        return "_No rows available._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join([format_metric(row.get(column)) for column in columns]) + " |")
    return "\n".join(lines) + "\n"


def format_metric(value: Any) -> str:
    """Format a metric for reports."""
    if value in (None, "", NOT_AVAILABLE):
        return NOT_AVAILABLE
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if abs(value) >= 1000.0:
            return "%.0f" % value
        return "%.6f" % value
    return str(value)


def latex_escape(value: Any) -> str:
    """Escape text for LaTeX."""
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
