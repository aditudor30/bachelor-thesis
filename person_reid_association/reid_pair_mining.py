"""Mine Person candidate pairs and attach ReID embeddings."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from deep_oc_sort_3d.person_association.person_pair_mining import (
    PersonTrackFragment,
    load_person_fragments_from_final_export,
    mine_person_candidate_pairs_with_summary,
)
from deep_oc_sort_3d.person_reid.reid_utils import cosine_similarity
from deep_oc_sort_3d.person_reid_association.reid_association_io import (
    TrackKey,
    parse_track_key,
    progress_iter,
    read_embeddings_jsonl,
    safe_float,
    serialize_track_key,
    write_csv_rows,
    write_json,
)


def load_reid_global_embeddings(reid_global_embeddings_root: Path) -> Dict[TrackKey, Any]:
    """Load global-fragment ReID embeddings keyed by Person final-export track key."""
    path = reid_global_embeddings_root / "person_global_fragment_embeddings.jsonl"
    records = read_embeddings_jsonl(path)
    mapping: Dict[TrackKey, Any] = {}
    for record in records:
        if record.global_track_id is None:
            continue
        key = (str(record.subset), str(record.scene_name), str(record.class_id), str(record.global_track_id))
        mapping[key] = record
    return mapping


def mine_reid_person_pairs_from_config(config: Dict[str, Any], show_progress: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load fragments and ReID embeddings, then mine ReID-covered candidate pairs."""
    paths = config.get("paths", {})
    final_root = Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam")))
    reid_root = Path(str(paths.get("reid_global_embeddings_root", "output/reid_person/baseline_v2_pseudo3d_fullcam/embeddings_global_fragment")))
    fragment_config = {
        "class_id": config.get("class_id", config.get("person_reid_association", {}).get("class_id", 0)),
        "subsets": config.get("apply_to_subsets", config.get("diagnostics", {}).get("subsets")),
        "scenes": config.get("apply_to_scenes"),
    }
    fragments = load_person_fragments_from_final_export(final_root, fragment_config, show_progress=show_progress)
    embeddings = load_reid_global_embeddings(reid_root)
    geometry_rows, geometry_summary = mine_person_candidate_pairs_with_summary(
        fragments,
        config.get("pair_mining", {}),
        show_progress=show_progress,
    )
    rows, reid_summary = attach_reid_to_pairs(geometry_rows, embeddings)
    summary = dict(geometry_summary)
    summary.update(reid_summary)
    summary["fragments"] = len(fragments)
    summary["reid_embeddings"] = len(embeddings)
    return rows, summary


def attach_reid_to_pairs(rows: List[Dict[str, Any]], embeddings: Dict[TrackKey, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Attach ReID similarity to geometry-compatible candidate rows."""
    output = []
    missing = 0
    with_reid = 0
    same_gt = 0
    different_gt = 0
    unknown_gt = 0
    for row in rows:
        key_a = parse_track_key(row.get("track_a"))
        key_b = parse_track_key(row.get("track_b"))
        emb_a = embeddings.get(key_a)
        emb_b = embeddings.get(key_b)
        copied = dict(row)
        if emb_a is None or emb_b is None:
            copied["reid_status"] = "missing_reid"
            copied["reid_missing_a"] = emb_a is None
            copied["reid_missing_b"] = emb_b is None
            copied["reid_similarity"] = ""
            missing += 1
        else:
            similarity = cosine_similarity(emb_a.embedding, emb_b.embedding)
            copied["reid_status"] = "ok"
            copied["reid_missing_a"] = False
            copied["reid_missing_b"] = False
            copied["reid_similarity"] = similarity
            copied["embedding_id_a"] = emb_a.embedding_id
            copied["embedding_id_b"] = emb_b.embedding_id
            copied["gt_a"] = "" if emb_a.matched_gt_object_id is None else emb_a.matched_gt_object_id
            copied["gt_b"] = "" if emb_b.matched_gt_object_id is None else emb_b.matched_gt_object_id
            with_reid += 1
            label = _gt_label(emb_a, emb_b)
            copied["reid_gt_pair_label"] = label
            if label == "same_gt":
                same_gt += 1
            elif label == "different_gt":
                different_gt += 1
            else:
                unknown_gt += 1
        output.append(copied)
    return output, {
        "candidate_rows": len(rows),
        "pairs_with_both_reid": with_reid,
        "pairs_missing_reid": missing,
        "same_gt_reid_pairs": same_gt,
        "different_gt_reid_pairs": different_gt,
        "unknown_gt_reid_pairs": unknown_gt,
    }


def write_reid_candidate_pairs(rows: List[Dict[str, Any]], output_csv: Path, summary_json: Path, summary: Dict[str, Any]) -> None:
    """Write ReID candidate pairs and summary."""
    fields = [
        "subset",
        "scene_name",
        "class_id",
        "class_name",
        "track_a",
        "track_b",
        "global_track_id_a",
        "global_track_id_b",
        "cameras_a",
        "cameras_b",
        "start_a",
        "end_a",
        "start_b",
        "end_b",
        "rows_a",
        "rows_b",
        "temporal_gap",
        "temporal_overlap",
        "entry_exit_distance_3d",
        "expected_position_error",
        "velocity_angle",
        "min_mean_confidence",
        "same_gt_diagnostic",
        "candidate_status",
        "reject_reason",
        "reid_status",
        "reid_similarity",
        "embedding_id_a",
        "embedding_id_b",
        "gt_a",
        "gt_b",
        "reid_gt_pair_label",
        "reid_missing_a",
        "reid_missing_b",
    ]
    write_csv_rows(rows, output_csv, fields)
    write_json(summary, summary_json)


def _gt_label(left: Any, right: Any) -> str:
    if left.matched_gt_object_id is None or right.matched_gt_object_id is None:
        return "unknown_gt"
    if _normalize_gt_id(left.matched_gt_object_id) == _normalize_gt_id(right.matched_gt_object_id):
        return "same_gt"
    return "different_gt"


def _normalize_gt_id(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)
