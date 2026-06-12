"""Run one isolated Step 21E motion-filter variant."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from deep_oc_sort_3d.bytetrack_motion_filtering.gap_aware_motion_filter import classify_candidate_gap_aware
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_config import candidate_root, variant_root
from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_io import progress_iter, write_csv, write_json, write_yaml
from deep_oc_sort_3d.bytetrack_motion_filtering.velocity_prior_estimator import load_or_estimate_velocity_priors
from deep_oc_sort_3d.mtmc.candidate_io import read_candidates_file
from deep_oc_sort_3d.mtmc.candidate_motion_io import (
    write_candidates_with_motion_csv,
    write_candidates_with_motion_jsonl,
    write_motion_metrics_csv,
)
from deep_oc_sort_3d.mtmc.candidate_motion_quality import CandidateMotionMetrics


def run_motion_filter_variant(
    config: Dict[str, Any],
    variant_name: str,
    progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Filter all 21C candidates for one configured variant."""
    variants = config.get("variants", {})
    if variant_name not in variants:
        raise ValueError("Unknown motion-filter variant: %s" % variant_name)
    root = variant_root(config, variant_name)
    motion_root = root / "motion_clean"
    summary_path = root / "summaries" / "motion_quality_summary.json"
    if skip_existing and not overwrite and summary_path.exists():
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    variant_config = dict(variants.get(variant_name, {}))
    write_yaml(root / "variant_config.yaml", {"variant_name": variant_name, "motion_filter": variant_config})
    priors = load_or_estimate_velocity_priors(config, progress=progress, overwrite=False)
    current_config = _load_current_motion_config(config)
    source_root = candidate_root(config)
    files = _find_candidate_files(source_root)
    if not files:
        raise FileNotFoundError("No candidate files found under %s" % source_root)

    file_rows = []
    all_diagnostics = []
    totals = {"input": 0, "clean": 0, "suspicious": 0, "invalid": 0, "unknown": 0}
    for path in progress_iter(files, progress, "%s motion files" % variant_name):
        subset, scene_name, camera_id = _parse_candidate_path(source_root, path)
        output_dir = motion_root / subset / scene_name
        expected = output_dir / (camera_id + "_clean_candidates.jsonl")
        if expected.exists() and not overwrite:
            file_rows.append(_file_row(path, output_dir, 0, 0, 0, 0, 0, "skipped_existing", ""))
            continue
        try:
            candidates = read_candidates_file(path)
            buckets = {"clean": [], "suspicious": [], "invalid": [], "unknown": []}
            metrics = []
            diagnostics = []
            for candidate in progress_iter(candidates, progress, "%s %s" % (variant_name, camera_id)):
                decision = classify_candidate_gap_aware(candidate, variant_config, priors, current_config)
                metric = decision.metrics
                metrics.append(metric)
                detail = dict(decision.diagnostics)
                detail["variant_name"] = variant_name
                detail["input_path"] = str(path)
                detail["violations_json"] = json.dumps(detail.pop("violations", []), sort_keys=True)
                diagnostics.append(detail)
                if metric.is_motion_clean:
                    buckets["clean"].append(candidate)
                if metric.motion_quality_flag == "motion_suspicious":
                    buckets["suspicious"].append(candidate)
                elif metric.motion_quality_flag == "motion_invalid":
                    buckets["invalid"].append(candidate)
                elif metric.motion_quality_flag == "motion_unknown":
                    buckets["unknown"].append(candidate)
            metrics_by_id = {item.candidate_id: item for item in metrics}
            _write_bucket(output_dir, camera_id, "clean", buckets["clean"], metrics_by_id)
            _write_bucket(output_dir, camera_id, "suspicious", buckets["suspicious"], metrics_by_id)
            _write_bucket(output_dir, camera_id, "invalid", buckets["invalid"], metrics_by_id)
            if buckets["unknown"]:
                _write_bucket(output_dir, camera_id, "unknown", buckets["unknown"], metrics_by_id)
            write_motion_metrics_csv(metrics, output_dir / (camera_id + "_motion_metrics.csv"))
            write_csv(output_dir / (camera_id + "_gap_aware_diagnostics.csv"), diagnostics)
            row = _file_row(
                path,
                output_dir,
                len(candidates),
                len(buckets["clean"]),
                len(buckets["suspicious"]),
                len(buckets["invalid"]),
                len(buckets["unknown"]),
                "ok",
                "",
            )
            file_rows.append(row)
            all_diagnostics.extend(diagnostics)
            totals["input"] += len(candidates)
            for key in ("clean", "suspicious", "invalid", "unknown"):
                totals[key] += len(buckets[key])
        except Exception as exc:
            file_rows.append(_file_row(path, output_dir, 0, 0, 0, 0, 0, "error", str(exc)))

    summary = _summarize(variant_name, source_root, totals, file_rows, all_diagnostics)
    write_json(summary_path, summary)
    write_json(motion_root / "summaries" / "motion_quality_summary.json", summary)
    write_csv(root / "summaries" / "motion_quality_files.csv", file_rows)
    write_csv(motion_root / "summaries" / "motion_quality_files.csv", file_rows)
    write_csv(root / "summaries" / "candidate_diagnostics.csv", all_diagnostics)
    print("variant: %s" % variant_name)
    print("candidate_root: %s" % source_root)
    print("input_candidates: %d" % totals["input"])
    print("motion_clean_candidates: %d" % totals["clean"])
    print("motion_clean_retention: %s" % summary.get("motion_clean_retention"))
    return summary


def _find_candidate_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*_candidates.jsonl"))
    if not files:
        files = sorted(root.rglob("*_candidates.csv"))
    output = []
    for path in files:
        if "summaries" in set(path.parts):
            continue
        if any(token in path.stem for token in ("_clean_", "_invalid_", "_suspicious_", "_unknown_")):
            continue
        output.append(path)
    return output


def _parse_candidate_path(root: Path, path: Path) -> Tuple[str, str, str]:
    relative = path.relative_to(root)
    parts = list(relative.parts)
    subset = parts[0] if len(parts) >= 3 else "unknown"
    scene_name = parts[1] if len(parts) >= 3 else "unknown"
    camera_id = path.stem.replace("_candidates", "")
    return subset, scene_name, camera_id


def _write_bucket(
    output_dir: Path,
    camera_id: str,
    name: str,
    candidates: List[Any],
    metrics_by_id: Dict[str, CandidateMotionMetrics],
) -> None:
    write_candidates_with_motion_jsonl(candidates, metrics_by_id, output_dir / (camera_id + "_" + name + "_candidates.jsonl"))
    write_candidates_with_motion_csv(candidates, metrics_by_id, output_dir / (camera_id + "_" + name + "_candidates.csv"))


def _load_current_motion_config(config: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(str(config.get("current_motion_quality_config", "deep_oc_sort_3d/configs/mtmc_motion_quality_medium_conf001.yaml")))
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = value.get("motion_quality", value) if isinstance(value, dict) else {}
    return section if isinstance(section, dict) else {}


def _summarize(
    variant_name: str,
    source_root: Path,
    totals: Dict[str, int],
    files: List[Dict[str, Any]],
    diagnostics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    steps = [float(row["max_step_distance_3d"]) for row in diagnostics if row.get("max_step_distance_3d") not in (None, "")]
    jump_count = sum(int(row.get("jump_count", 0) or 0) for row in diagnostics)
    total_steps = sum(int(row.get("num_steps", 0) or 0) for row in diagnostics)
    return {
        "variant_name": variant_name,
        "status": "error" if any(row.get("status") == "error" for row in files) else "ok",
        "candidate_root_used": str(source_root),
        "files": files,
        "input_candidates": totals["input"],
        "motion_clean_candidates": totals["clean"],
        "clean_count": totals["clean"],
        "rejected_candidates": totals["input"] - totals["clean"],
        "motion_clean_retention": _ratio(totals["clean"], totals["input"]),
        "rejection_rate": _ratio(totals["input"] - totals["clean"], totals["input"]),
        "motion_good": sum(1 for row in diagnostics if row.get("motion_quality_flag") == "motion_good"),
        "motion_suspicious": totals["suspicious"],
        "motion_invalid": totals["invalid"],
        "motion_unknown": totals["unknown"],
        "step_p95": _percentile(steps, 95),
        "step_p99": _percentile(steps, 99),
        "step_max": max(steps) if steps else None,
        "jump_count": jump_count,
        "jump_ratio": _ratio(jump_count, total_steps),
    }


def _file_row(
    input_path: Path,
    output_dir: Path,
    input_count: int,
    clean: int,
    suspicious: int,
    invalid: int,
    unknown: int,
    status: str,
    error: str,
) -> Dict[str, Any]:
    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "input_candidates": input_count,
        "clean_count": clean,
        "suspicious_count": suspicious,
        "invalid_count": invalid,
        "unknown_count": unknown,
        "status": status,
        "error_message": error,
    }


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=float), percentile))
