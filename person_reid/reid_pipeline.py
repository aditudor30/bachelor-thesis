"""Step 16A Person ReID pipeline orchestration."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.person_reid.crop_extraction import extract_person_reid_crops_from_config
from deep_oc_sort_3d.person_reid.reid_aggregation import (
    aggregate_person_embeddings_from_config,
    compute_person_crop_embeddings_from_config,
)
from deep_oc_sort_3d.person_reid.reid_config import load_person_reid_config, output_root
from deep_oc_sort_3d.person_reid.reid_diagnostics import run_person_reid_diagnostics_from_config
from deep_oc_sort_3d.person_reid.reid_report import write_person_reid_report
from deep_oc_sort_3d.person_reid.reid_utils import write_json


def run_step16a_person_reid(config_path: Path, show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Run crop extraction, embedding, aggregation, diagnostics, and report."""
    config = load_person_reid_config(config_path)
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    status = {"config_path": str(config_path), "output_root": str(root)}
    crop_summary = extract_person_reid_crops_from_config(config, show_progress=show_progress, overwrite=overwrite)
    status["crop_extraction"] = crop_summary
    embedding_summary = compute_person_crop_embeddings_from_config(config, show_progress=show_progress, overwrite=overwrite)
    status["crop_embedding"] = embedding_summary
    if embedding_summary.get("status") == "ok":
        aggregation_summary = aggregate_person_embeddings_from_config(config, show_progress=show_progress, overwrite=overwrite)
        diagnostics_summary = run_person_reid_diagnostics_from_config(config, show_progress=show_progress, overwrite=overwrite)
    else:
        aggregation_summary = {"status": "skipped_no_embeddings"}
        diagnostics_summary = {"status": "skipped_no_embeddings", "verdict": "reid_backend_unavailable"}
        write_json(aggregation_summary, root / "summaries" / "aggregation_summary.json")
        write_json(diagnostics_summary, root / "summaries" / "reid_diagnostics_summary.json")
        write_json(diagnostics_summary, root / "report" / "PERSON_REID_STEP16A_SUMMARY.json")
    status["aggregation"] = aggregation_summary
    status["diagnostics"] = diagnostics_summary
    report_summary = write_person_reid_report(root)
    status["verdict"] = report_summary.get("verdict")
    write_json(status, root / "summaries" / "step16a_pipeline_status.json")
    return status

