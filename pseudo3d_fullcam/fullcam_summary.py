"""Summary builder for Step 15G full-camera pseudo-3D outputs."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.audit3d.audit3d_io import read_json_if_exists, write_csv, write_json


def summarize_step15g(root: Path, criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a compact Step 15G summary from generated reports."""
    coverage = read_json_if_exists(root / "coverage_audit" / "pseudo3d_fullcam_coverage_summary.json")
    raw = read_json_if_exists(root / "raw_generation_reports" / "summary_raw_generation.json")
    stabilization = read_json_if_exists(root / "stabilization_reports" / "summary_stabilization.json")
    criteria = criteria or {}
    min_camera_coverage = float(criteria.get("min_camera_file_coverage", 0.95))
    min_success = float(criteria.get("min_prediction_success_rate", 0.90))
    stabilized_file_coverage = coverage.get("stabilized_file_coverage")
    success_rate = coverage.get("success_rate_stabilized")
    can_continue = _meets(stabilized_file_coverage, min_camera_coverage) and _meets(success_rate, min_success)
    return {
        "step": "15G",
        "name": "baseline_v2_pseudo3d_fullcam_generation",
        "output_root": str(root),
        "required_camera_files": coverage.get("required_camera_files"),
        "raw_files_existing": coverage.get("raw_files_existing"),
        "stabilized_files_existing": coverage.get("stabilized_files_existing"),
        "raw_file_coverage": coverage.get("raw_file_coverage"),
        "stabilized_file_coverage": stabilized_file_coverage,
        "total_records_expected": coverage.get("total_records_expected"),
        "total_raw_predictions": coverage.get("total_raw_predictions"),
        "total_stabilized_predictions": coverage.get("total_stabilized_predictions"),
        "raw_record_coverage": coverage.get("raw_record_coverage"),
        "stabilized_record_coverage": coverage.get("stabilized_record_coverage"),
        "success_rate_raw": coverage.get("success_rate_raw"),
        "success_rate_stabilized": success_rate,
        "raw_generation": raw,
        "stabilization": stabilization,
        "source_metadata_completeness": coverage.get("source_metadata_completeness", {}),
        "projection": coverage.get("projection", {}),
        "smoothness": coverage.get("smoothness", {}),
        "failed_camera_count": len(coverage.get("failed_cameras", [])) if isinstance(coverage.get("failed_cameras"), list) else coverage.get("stabilized_files_missing"),
        "can_continue_to_15h": can_continue,
        "recommendation": "continue_to_15h" if can_continue else "inspect_failed_cameras_before_15h",
        "criteria": {
            "min_camera_file_coverage": min_camera_coverage,
            "min_prediction_success_rate": min_success,
        },
    }


def write_step15g_summary(summary: Dict[str, Any], root: Path) -> None:
    """Write Step 15G summary JSON/CSV/report-compatible artifacts."""
    summaries_root = root / "summaries"
    report_root = root / "report"
    write_json(summary, summaries_root / "step15g_summary.json")
    write_csv(_summary_rows(summary), summaries_root / "step15g_summary.csv")
    write_json(summary, report_root / "PSEUDO3D_STEP15G_SUMMARY.json")


def _meets(value: Any, threshold: float) -> bool:
    try:
        return float(value) >= threshold
    except (TypeError, ValueError):
        return False


def _summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for key, value in sorted(summary.items()):
        if isinstance(value, (dict, list, tuple)):
            continue
        rows.append({"metric": key, "value": value})
    return rows
