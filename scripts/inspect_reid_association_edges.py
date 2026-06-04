"""Inspect ReID fields on global association edges."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.mtmc.global_io import read_association_edges_file
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge, edge_to_dict


def inspect_reid_association_edges(args: Any) -> Dict[str, Any]:
    """Inspect accepted/rejected ReID edge distributions."""
    edges = read_association_edges_file(args.edges)
    summary = summarize_edges(edges, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("edges: %d" % summary.get("edges", 0))
    print("accepted: %d" % summary.get("accepted", 0))
    print("rejected: %d" % summary.get("rejected", 0))
    print("used_reid: %d" % summary.get("used_reid", 0))
    print("Wrote %s" % args.output)
    return summary


def summarize_edges(edges: List[GlobalAssociationEdge], top_k: int = 20) -> Dict[str, Any]:
    """Return edge-level ReID diagnostics."""
    accepted = [edge for edge in edges if edge.accepted]
    rejected = [edge for edge in edges if not edge.accepted]
    accepted_high = sorted(
        [edge for edge in accepted if edge.appearance_distance is not None],
        key=lambda edge: float(edge.appearance_distance),
        reverse=True,
    )[: int(top_k)]
    rejected_low = sorted(
        [edge for edge in rejected if edge.appearance_distance is not None],
        key=lambda edge: float(edge.appearance_distance),
    )[: int(top_k)]
    return {
        "edges": len(edges),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "used_reid": len([edge for edge in edges if edge.used_reid]),
        "missing_reid": len([edge for edge in edges if not edge.used_reid]),
        "appearance_distance_all": _stats([edge.appearance_distance for edge in edges if edge.appearance_distance is not None]),
        "appearance_distance_accepted": _stats(
            [edge.appearance_distance for edge in accepted if edge.appearance_distance is not None]
        ),
        "appearance_distance_rejected": _stats(
            [edge.appearance_distance for edge in rejected if edge.appearance_distance is not None]
        ),
        "cosine_similarity_accepted": _stats([edge.cosine_similarity for edge in accepted if edge.cosine_similarity is not None]),
        "cosine_similarity_rejected": _stats([edge.cosine_similarity for edge in rejected if edge.cosine_similarity is not None]),
        "reject_reasons": _count_by_reject(edges),
        "reid_missing_reasons": _count_by_missing(edges),
        "accepted_edges_with_high_appearance_distance": [edge_to_dict(edge) for edge in accepted_high],
        "rejected_edges_with_low_appearance_distance": [edge_to_dict(edge) for edge in rejected_low],
    }


def _stats(values: List[Any]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p75": None, "p90": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def _count_by_reject(edges: List[GlobalAssociationEdge]) -> Dict[str, int]:
    counts = {}
    for edge in edges:
        reason = "ok" if edge.accepted else str(edge.reject_reason)
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _count_by_missing(edges: List[GlobalAssociationEdge]) -> Dict[str, int]:
    counts = {}
    for edge in edges:
        reason = str(edge.reid_missing_reason or ("used_reid" if edge.used_reid else "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Inspect ReID-aware global association edges.")
    parser.add_argument("--edges", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    inspect_reid_association_edges(args)


if __name__ == "__main__":
    main()
