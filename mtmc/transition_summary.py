"""I/O and summary helpers for MTMC transition diagnostics."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.mtmc.transition_diagnostics import summarize_transition_pairs
from deep_oc_sort_3d.mtmc.transition_types import (
    TransitionCandidatePair,
    transition_pair_from_dict,
    transition_pair_to_dict,
)


CSV_FIELDS = [
    "scene_name",
    "subset",
    "split",
    "class_id",
    "class_name",
    "candidate_id_a",
    "candidate_id_b",
    "camera_id_a",
    "camera_id_b",
    "camera_pair",
    "start_frame_a",
    "end_frame_a",
    "start_frame_b",
    "end_frame_b",
    "temporal_relation",
    "temporal_gap",
    "entry_exit_distance",
    "normalized_entry_exit_distance",
    "velocity_angle_difference",
    "velocity_magnitude_difference",
    "expected_position_error",
    "reverse_expected_position_error",
    "confidence_pair_mean",
    "gt_id_a",
    "gt_id_b",
    "same_gt_object_id",
    "diagnostic_label",
    "transition_cost",
    "accepted_by_threshold",
    "reject_reason",
]


def write_transition_pairs_csv(pairs: List[TransitionCandidatePair], path: Path) -> None:
    """Write transition pairs as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(transition_pair_to_dict(pair))


def read_transition_pairs_csv(path: Path) -> List[TransitionCandidatePair]:
    """Read transition pairs from CSV."""
    if not path.exists():
        return []
    output = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.append(transition_pair_from_dict(row))
    return output


def write_transition_pairs_jsonl(pairs: List[TransitionCandidatePair], path: Path) -> None:
    """Write transition pairs as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(transition_pair_to_dict(pair), sort_keys=True) for pair in pairs]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_transition_pairs_jsonl(path: Path) -> List[TransitionCandidatePair]:
    """Read transition pairs from JSONL."""
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            output.append(transition_pair_from_dict(json.loads(line)))
    return output


def write_transition_summary_json(summary: Dict[str, Any], path: Path) -> None:
    """Write transition summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def write_transition_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    """Write transition summary as compact CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def print_transition_summary(summary: Dict[str, Any]) -> None:
    """Print a compact transition summary."""
    print("total_pairs: %s" % summary.get("total_pairs"))
    print("true_transition: %s" % summary.get("true_transition"))
    print("false_transition: %s" % summary.get("false_transition"))
    print("unknown_gt: %s" % summary.get("unknown_gt"))
    print("accepted_by_threshold: %s" % summary.get("accepted_by_threshold"))
    print("accepted_true: %s" % summary.get("accepted_true"))
    print("accepted_false: %s" % summary.get("accepted_false"))
    print("precision_diagnostic: %s" % summary.get("precision_diagnostic"))
    print("recall_proxy_diagnostic: %s" % summary.get("recall_proxy_diagnostic"))
    print("per_class_counts: %s" % json.dumps(summary.get("per_class_counts", {}), sort_keys=True))
    print("reject_reasons: %s" % json.dumps(summary.get("reject_reasons", {}), sort_keys=True))


def summarize_and_write_transition_pairs(
    pairs: List[TransitionCandidatePair],
    output_root: Path,
) -> Dict[str, Any]:
    """Write transition pairs and summary to an output folder."""
    summary = summarize_transition_pairs(pairs)
    write_transition_pairs_csv(pairs, output_root / "transition_pairs.csv")
    write_transition_pairs_jsonl(pairs, output_root / "transition_pairs.jsonl")
    write_transition_summary_json(summary, output_root / "transition_summary.json")
    write_transition_summary_csv(summary, output_root / "transition_summary.csv")
    return summary
