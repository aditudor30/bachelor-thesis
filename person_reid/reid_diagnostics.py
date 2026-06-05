"""Diagnostic runner for Person ReID embeddings."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from deep_oc_sort_3d.person_reid.reid_embedding_io import read_embeddings_jsonl
from deep_oc_sort_3d.person_reid.reid_retrieval_eval import evaluate_topk_retrieval
from deep_oc_sort_3d.person_reid.reid_similarity import pair_similarity_rows, sample_similarity_pairs, summarize_similarity_rows, threshold_sweep
from deep_oc_sort_3d.person_reid.reid_utils import count_by, progress_iter, write_csv_rows, write_json


def run_person_reid_diagnostics_from_config(config: Dict[str, Any], show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Run similarity and retrieval diagnostics for aggregated Person embeddings."""
    root = Path(str(config.get("reid_person", {}).get("output_root", "output/reid_person/baseline_v2_pseudo3d_fullcam")))
    embedding_path = root / "embeddings_global_fragment" / "person_global_fragment_embeddings.jsonl"
    if not embedding_path.exists():
        summary = {"status": "missing_global_fragment_embeddings", "embedding_path": str(embedding_path)}
        write_json(summary, root / "summaries" / "reid_diagnostics_summary.json")
        return summary
    records = read_embeddings_jsonl(embedding_path)
    diagnostics = config.get("diagnostics", {})
    max_pairs = int(diagnostics.get("max_pairs_per_scene", 200000))
    thresholds = [float(item) for item in diagnostics.get("thresholds", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])]
    pair_rows: List[Dict[str, Any]] = []
    for subset in sorted(set([record.subset for record in records])):
        subset_records = [record for record in records if record.subset == subset]
        pairs = sample_similarity_pairs(subset_records, max_pairs=max_pairs)
        pair_rows.extend(pair_similarity_rows(subset_records, pairs))
    write_csv_rows(pair_rows, root / "similarity" / "person_similarity_pairs.csv")
    similarity_summary = summarize_similarity_rows(pair_rows)
    sweep_rows = threshold_sweep(pair_rows, thresholds)
    write_csv_rows(sweep_rows, root / "similarity" / "person_similarity_threshold_sweep.csv")
    retrieval = evaluate_topk_retrieval(records, [1, 5])
    write_json(retrieval, root / "retrieval" / "person_retrieval_topk.json")
    norms = [float(np.linalg.norm(record.embedding)) for record in records]
    summary = {
        "status": "ok",
        "num_embeddings": len(records),
        "embedding_dim": records[0].embedding_dim if records else None,
        "backend": records[0].backend if records else None,
        "embedding_norm_mean": float(np.mean(norms)) if norms else None,
        "embedding_norm_std": float(np.std(norms)) if norms else None,
        "per_subset": count_by([_meta(record) for record in records], "subset"),
        "per_scene": count_by([_meta(record) for record in records], "scene_name"),
        "per_camera": count_by([_meta(record) for record in records], "camera_id"),
        "similarity": similarity_summary,
        "threshold_sweep_rows": len(sweep_rows),
        "retrieval": retrieval,
        "verdict": verdict_for_reid(similarity_summary, sweep_rows, records),
    }
    write_json(summary, root / "summaries" / "reid_diagnostics_summary.json")
    write_json(summary, root / "report" / "PERSON_REID_STEP16A_SUMMARY.json")
    return summary


def verdict_for_reid(summary: Dict[str, Any], sweep_rows: List[Dict[str, Any]], records: List[Any]) -> str:
    """Choose Step 16A verdict."""
    if not records:
        return "reid_backend_unavailable"
    margin = summary.get("separation_margin")
    if margin is None:
        return "reid_not_discriminative_enough"
    good_rows = []
    for row in sweep_rows:
        precision = row.get("precision")
        recall = row.get("recall")
        fpr = row.get("fpr")
        if precision is not None and recall is not None and fpr is not None:
            if float(precision) >= 0.70 and float(recall) >= 0.20 and float(fpr) <= 0.10:
                good_rows.append(row)
    if float(margin) >= 0.10 and good_rows:
        return "reid_ready_for_person_association"
    if float(margin) >= 0.03:
        return "reid_promising_but_needs_threshold_tuning"
    return "reid_not_discriminative_enough"


def _meta(record: Any) -> Dict[str, Any]:
    return {"subset": record.subset, "scene_name": record.scene_name, "camera_id": record.camera_id}

