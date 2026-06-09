"""Manual-review CSV and instruction writer for ReID visual decisions."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import write_csv_dicts, write_text


REVIEW_FIELDS = [
    "variant",
    "merge_event_id",
    "auto_label",
    "risk_score",
    "risk_reasons",
    "human_label",
    "human_decision",
    "reviewer_notes",
    "panel_path",
    "reid_similarity",
    "spatial_distance",
    "temporal_gap",
    "fragment_a_id",
    "fragment_b_id",
    "global_track_after",
    "available_visual_evidence",
]


def write_manual_review_sheet(rows: List[Dict[str, Any]], output_root: Path, variant: str) -> Path:
    """Write one manual review CSV for a variant."""
    out_rows = []
    for row in rows:
        out = {field: row.get(field, "") for field in REVIEW_FIELDS}
        out["human_label"] = ""
        out["human_decision"] = ""
        out["reviewer_notes"] = ""
        out_rows.append(out)
    path = Path(output_root) / "manual_review" / ("review_sheet_%s.csv" % variant)
    write_csv_dicts(out_rows, path, REVIEW_FIELDS)
    return path


def write_review_instructions(output_root: Path) -> Path:
    """Write concise manual review instructions."""
    lines = [
        "# Person ReID Visual Merge Review Instructions",
        "",
        "Review each PNG panel together with its row in `review_sheet_*.csv`.",
        "",
        "Suggested `human_label` values:",
        "- `good_merge`: the two fragments look like the same person.",
        "- `bad_merge`: the fragments clearly belong to different people.",
        "- `ambiguous`: visual evidence is not enough.",
        "- `no_evidence`: crops/frames are missing or too poor.",
        "",
        "Suggested `human_decision` values:",
        "- `accept`: safe to keep this merge type.",
        "- `reject`: the merge is unsafe.",
        "- `needs_more_data`: keep as diagnostic only.",
        "",
        "The auto labels are heuristics. The final decision should prefer visual clarity over small numeric gains.",
    ]
    path = Path(output_root) / "manual_review" / "review_instructions.md"
    write_text(lines, path)
    return path

