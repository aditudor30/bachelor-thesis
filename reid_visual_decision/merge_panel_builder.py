"""Build visual panels for selected Person ReID merge events."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deep_oc_sort_3d.reid_visual_decision.crop_context_loader import load_visual_evidence
from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import safe_float, write_json
from deep_oc_sort_3d.reid_visual_decision.visual_risk_classifier import classify_merge_event


def build_merge_panels(
    events: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_root: Path,
    variant: str,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Create visual panels and return per-event review rows."""
    _unused_progress = progress
    panel_rows = []
    panel_cfg = config.get("panels", {})
    max_crops = int(panel_cfg.get("max_crops_per_fragment", 4))
    for event in events:
        evidence = load_visual_evidence(event, config, max_crops=max_crops)
        label_info = classify_merge_event(event, evidence, config)
        label = str(label_info.get("auto_label", "ambiguous"))
        panel_path = Path(output_root) / "visual_panels" / variant / label / ("%s.png" % event.get("merge_event_id"))
        panel_summary = save_panel(event, evidence, label_info, panel_path)
        row = dict(event)
        row.update(label_info)
        row.update(
            {
                "num_rows_a": evidence.get("num_rows_a"),
                "num_rows_b": evidence.get("num_rows_b"),
                "num_crops_a": evidence.get("num_crops_a"),
                "num_crops_b": evidence.get("num_crops_b"),
                "panel_path": str(panel_path),
                "panel_status": panel_summary.get("status"),
                "available_visual_evidence": visual_evidence_status(evidence),
            }
        )
        panel_rows.append(row)
    return panel_rows


def visual_evidence_status(evidence: Dict[str, Any]) -> str:
    """Return compact visual-evidence status."""
    if int(evidence.get("num_crops_a", 0) or 0) > 0 and int(evidence.get("num_crops_b", 0) or 0) > 0:
        return "both_fragments"
    if int(evidence.get("num_crops_a", 0) or 0) > 0 or int(evidence.get("num_crops_b", 0) or 0) > 0:
        return "one_fragment"
    return "none"


def save_panel(event: Dict[str, Any], evidence: Dict[str, Any], label_info: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Save one visual panel with Fragment A/B crop rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        write_json({"status": "matplotlib_unavailable", "event": event}, path.with_suffix(".json"))
        return {"status": "matplotlib_unavailable"}
    crops_a = evidence.get("crops_a", [])
    crops_b = evidence.get("crops_b", [])
    cols = max(4, max(len(crops_a), len(crops_b), 1))
    fig, axes = plt.subplots(3, cols, figsize=(3.0 * cols, 8.0))
    if cols == 1:
        axes = np.asarray(axes).reshape(3, 1)
    title = "%s | sim=%s | label=%s" % (
        event.get("merge_event_id"),
        format_float(event.get("reid_similarity")),
        label_info.get("auto_label"),
    )
    fig.suptitle(title, fontsize=11)
    draw_text_row(axes[0], metadata_lines(event, label_info))
    draw_crop_row(axes[1], crops_a, "A")
    draw_crop_row(axes[2], crops_b, "B")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(str(path), dpi=140)
    plt.close(fig)
    return {"status": "ok", "path": str(path)}


def draw_text_row(axes: Any, lines: List[str]) -> None:
    """Draw metadata text across the first row."""
    for ax in list(axes):
        ax.axis("off")
    if len(axes) > 0:
        axes[0].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")


def draw_crop_row(axes: Any, crop_items: List[Dict[str, Any]], prefix: str) -> None:
    """Draw crop images for one fragment."""
    for index, ax in enumerate(list(axes)):
        ax.axis("off")
        if index >= len(crop_items):
            continue
        crop = crop_items[index].get("crop")
        row = crop_items[index].get("row", {})
        if crop is None:
            ax.text(0.5, 0.5, "%s missing" % prefix, ha="center", va="center")
            continue
        ax.imshow(crop)
        ax.set_title("%s f=%s cam=%s" % (prefix, row.get("frame_id"), row.get("camera_id")), fontsize=8)


def metadata_lines(event: Dict[str, Any], label_info: Dict[str, Any]) -> List[str]:
    """Build metadata text for the panel."""
    return [
        "variant: %s" % event.get("variant"),
        "fragment A: %s" % event.get("fragment_a_id"),
        "fragment B: %s" % event.get("fragment_b_id"),
        "after: %s" % event.get("global_track_after"),
        "cameras: %s -> %s" % (event.get("camera_a"), event.get("camera_b")),
        "frames: %s-%s / %s-%s" % (
            event.get("frame_start_a"),
            event.get("frame_end_a"),
            event.get("frame_start_b"),
            event.get("frame_end_b"),
        ),
        "spatial=%s temporal=%s combined=%s" % (
            format_float(event.get("spatial_distance")),
            format_float(event.get("temporal_gap")),
            format_float(event.get("combined_score")),
        ),
        "risk=%s reasons=%s" % (format_float(label_info.get("risk_score")), label_info.get("risk_reasons")),
    ]


def format_float(value: Any) -> str:
    """Format optional float values."""
    parsed = safe_float(value, None)
    if parsed is None:
        return "None"
    return "%.4f" % parsed

