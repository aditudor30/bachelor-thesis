"""Diagnostic figures for the Step 21D audit."""

from pathlib import Path
from typing import Any, Dict, List


def write_audit_figures(
    stage_rows: List[Dict[str, Any]],
    lifecycle: Dict[str, Any],
    gt_result: Dict[str, Any],
    motion_result: Dict[str, Any],
    output_root: Path,
) -> None:
    """Write requested figures when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    _stage_waterfall(plt, stage_rows, output_root / "stage_retention_waterfall.png")
    _lifecycle_states(plt, lifecycle, output_root / "lifecycle_state_distribution.png")
    _gt_duration(plt, gt_result, output_root / "gt_retention_by_duration_bucket.png")
    _motion_gap(plt, motion_result, output_root / "motion_rejection_by_gap_bucket.png")
    _jump_rejection(plt, motion_result, output_root / "pseudo3d_jump_vs_motion_rejection.png")
    _dimension_plot(plt, stage_rows, "top_affected_scenes", output_root / "per_scene_retention_comparison.png")
    _dimension_plot(plt, stage_rows, "top_affected_classes", output_root / "per_class_retention_comparison.png")


def _stage_waterfall(plt: Any, rows: List[Dict[str, Any]], path: Path) -> None:
    selected = [row for row in rows if row.get("variant_name") == "bytetrack_21c_best"]
    labels = ["%s\n-> %s" % (row.get("stage_from"), row.get("stage_to")) for row in selected]
    values = [float(row.get("retention") or 0.0) for row in selected]
    colors = ["#247ba0" if row.get("unit_comparison_type") == "consistent" else "#b8b8b8" for row in selected]
    _bar(plt, labels, values, colors, "21C stage ratios (gray = diagnostic-only)", "retention", path)


def _lifecycle_states(plt: Any, result: Dict[str, Any], path: Path) -> None:
    rows = result.get("rows", [])
    labels = [str(row.get("variant_name")) for row in rows]
    tentative = [float(row.get("tentative_tracks", 0) or 0) for row in rows]
    confirmed = [float(row.get("confirmed_tracks", 0) or 0) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = list(range(len(labels)))
    axis.bar(positions, tentative, label="tentative")
    axis.bar(positions, confirmed, bottom=tentative, label="confirmed")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_title("Exported lifecycle-state records")
    if labels:
        axis.legend()
    figure.tight_layout()
    figure.savefig(str(path), dpi=150)
    plt.close(figure)


def _gt_duration(plt: Any, result: Dict[str, Any], path: Path) -> None:
    rows = [row for row in result.get("duration_rows", []) if row.get("stage") == "matched_in_bytetrack_21c_local"]
    labels = [str(row.get("duration_bucket")) for row in rows]
    values = [float(row.get("gt_object_frame_retention") or 0.0) for row in rows]
    _bar(plt, labels, values, "#2a9d8f", "GT retention by duration bucket", "retention", path)


def _motion_gap(plt: Any, result: Dict[str, Any], path: Path) -> None:
    rows = [row for row in result.get("gap_rows", []) if row.get("variant_name") == "bytetrack_21c_best"]
    labels = [str(row.get("gap_bucket")) for row in rows]
    values = [float(row.get("rejection_rate") or 0.0) for row in rows]
    _bar(plt, labels, values, "#e76f51", "Motion rejection by gap bucket", "rejection rate", path)


def _jump_rejection(plt: Any, result: Dict[str, Any], path: Path) -> None:
    rows = [row for row in result.get("bbox_rows", []) if row.get("variant_name") == "bytetrack_21c_best"]
    labels = [str(row.get("bbox_height_delta_bucket")) for row in rows]
    values = [float(row.get("rejection_rate") or 0.0) for row in rows]
    _bar(plt, labels, values, "#f4a261", "BBox-height jump proxy vs motion rejection", "rejection rate", path)


def _dimension_plot(plt: Any, rows: List[Dict[str, Any]], field: str, path: Path) -> None:
    selected = [
        row for row in rows
        if row.get("variant_name") == "bytetrack_21c_best" and row.get("unit_comparison_type") == "consistent"
    ]
    labels = ["%s->%s" % (row.get("stage_from"), row.get("stage_to")) for row in selected]
    values = [float(row.get("drop_ratio") or 0.0) for row in selected]
    _bar(plt, labels, values, "#457b9d", field.replace("top_affected_", "Affected "), "drop ratio", path)


def _bar(plt: Any, labels: List[str], values: List[float], colors: Any, title: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(range(len(values)), values, color=colors)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(str(path), dpi=150)
    plt.close(figure)

