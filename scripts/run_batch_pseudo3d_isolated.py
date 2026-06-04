"""Run isolated pseudo-3D over configured subsets/scenes/cameras."""

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List

import yaml

from deep_oc_sort_3d.audit3d.audit3d_io import ensure_clean_output_dir, progress_iter, write_csv, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_config import load_pseudo3d_config
from deep_oc_sort_3d.pseudo3d.pseudo3d_estimator import Pseudo3DEstimator
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import (
    load_scene_camera_calibration,
    prediction_summary,
    read_frame_record_inputs,
    write_pseudo3d_predictions_csv,
    write_pseudo3d_predictions_jsonl,
)
from deep_oc_sort_3d.pseudo3d.pseudo3d_priors import load_pseudo3d_priors


def run(args: Any) -> Dict[str, Any]:
    cfg = _load_yaml(args.config)
    output_root = Path(cfg.get("output", {}).get("root", "output/pseudo3d/baseline_v2_pseudo3d_isolated"))
    ensure_clean_output_dir(output_root, overwrite=args.overwrite)
    progress = bool(args.progress if args.progress is not None else cfg.get("progress", True))
    estimator_config = load_pseudo3d_config(cfg.get("paths", {}).get("pseudo3d_config", "deep_oc_sort_3d/configs/pseudo3d_isolated_debug.yaml"))
    priors = load_pseudo3d_priors(cfg.get("paths", {}).get("class_priors", ""))
    estimator = Pseudo3DEstimator(priors, estimator_config)
    summaries = []
    for item in progress_iter(_batch_items(cfg), progress, "batch pseudo3D cameras", "camera"):
        summaries.append(_run_one_item(item, cfg, output_root, estimator, progress))
    summary = _summarize_batch(summaries)
    write_json(summary, output_root / "summaries" / "pseudo3d_extraction_summary.json")
    write_csv(summaries, output_root / "summaries" / "pseudo3d_extraction_summary.csv")
    write_json(summary.get("source_metadata_completeness", {}), output_root / "summaries" / "source_metadata_completeness.json")
    write_csv(_dict_counts_to_rows(summary.get("per_class", {}), "class_id"), output_root / "summaries" / "per_class_summary.csv")
    write_csv(_dict_counts_to_rows(summary.get("per_subset", {}), "subset"), output_root / "summaries" / "per_subset_summary.csv")
    print("Batch pseudo3D cameras: %s" % len(summaries))
    return summary


def _run_one_item(item: Dict[str, Any], cfg: Dict[str, Any], output_root: Path, estimator: Pseudo3DEstimator, progress: bool) -> Dict[str, Any]:
    try:
        dataset_root = Path(cfg.get("paths", {}).get("dataset_root", "dataset/MTMC_Tracking_2026"))
        records = Path(cfg.get("paths", {}).get("frame_records_root", "")) / item["subset"] / item["scene"] / ("%s_global_records.csv" % item["camera_id"])
        calibration = load_scene_camera_calibration(dataset_root, item["split"], item["scene"], item["camera_id"])
        inputs = read_frame_record_inputs(records, item["subset"], item["split"], item["scene"], item["camera_id"], calibration, progress)
        outputs = [estimator.estimate(input_item) for input_item in inputs]
        pred_dir = output_root / "predictions" / item["subset"] / item["scene"]
        stem = "%s_pseudo3d_predictions" % item["camera_id"]
        write_pseudo3d_predictions_jsonl(outputs, pred_dir / ("%s.jsonl" % stem))
        write_pseudo3d_predictions_csv(outputs, pred_dir / ("%s.csv" % stem))
        summary = prediction_summary(outputs)
        summary.update(item)
        summary["status"] = "ok"
        summary["records_path"] = str(records)
        return summary
    except Exception as exc:
        out = dict(item)
        out.update({"status": "error", "error": str(exc), "traceback": traceback.format_exc()})
        return out


def _batch_items(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for subset_name, subset_cfg in cfg.get("subsets", {}).items():
        split = str(subset_cfg.get("split", subset_name))
        for scene in subset_cfg.get("scenes", []):
            for camera_id in subset_cfg.get("camera_ids", []):
                items.append({"subset": subset_name, "split": split, "scene": scene, "camera_id": camera_id})
    return items


def _summarize_batch(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_predictions = sum(int(row.get("num_predictions", 0) or 0) for row in rows)
    failed_predictions = sum(int(row.get("num_failed", 0) or 0) for row in rows)
    per_class = _merge_count_dicts([row.get("per_class", {}) for row in rows])
    per_subset = _merge_count_dicts([row.get("per_subset", {}) for row in rows])
    return {
        "camera_count": len(rows),
        "camera_errors": sum(1 for row in rows if row.get("status") == "error"),
        "num_predictions": total_predictions,
        "num_failed": failed_predictions,
        "success_rate": float(total_predictions - failed_predictions) / float(total_predictions) if total_predictions else None,
        "per_class": per_class,
        "per_subset": per_subset,
        "source_metadata_completeness": _aggregate_metadata(rows, total_predictions),
    }


def _merge_count_dicts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
    return counts


def _dict_counts_to_rows(counts: Dict[str, int], key_name: str) -> List[Dict[str, Any]]:
    return [{key_name: key, "count": value} for key, value in sorted(counts.items())]


def _aggregate_metadata(rows: List[Dict[str, Any]], total_predictions: int) -> Dict[str, Any]:
    totals = {}
    for row in rows:
        metadata = row.get("source_metadata_completeness", {})
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            if key.endswith("_complete") or key == "is_estimated_for_test_set":
                totals[key] = totals.get(key, 0) + int(value or 0)
    for key, value in list(totals.items()):
        totals["%s_rate" % key] = float(value) / float(total_predictions) if total_predictions else None
    totals["total"] = total_predictions
    return totals


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated pseudo-3D batch.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
