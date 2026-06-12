"""Orchestrate the isolated Step 21A local tracker benchmark."""

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from deep_oc_sort_3d.data.frame_io import list_video_files
from deep_oc_sort_3d.local_tracker_benchmark.benchmark_config import (
    enabled_tracker_names,
    load_benchmark_config,
    output_root_from_config,
    resolve_scene_selection,
)
from deep_oc_sort_3d.local_tracker_benchmark.benchmark_figures import create_benchmark_figures
from deep_oc_sort_3d.local_tracker_benchmark.benchmark_report import write_benchmark_report
from deep_oc_sort_3d.local_tracker_benchmark.benchmark_selector import select_local_tracker
from deep_oc_sort_3d.local_tracker_benchmark.botsort_style_tracker import BoTSORTStyleTracker
from deep_oc_sort_3d.local_tracker_benchmark.bytetrack_style_tracker import ByteTrackStyleTracker
from deep_oc_sort_3d.local_tracker_benchmark.current_tracker_loader import load_current_tracker_subset
from deep_oc_sort_3d.local_tracker_benchmark.detection_loader import (
    group_detections_by_frame,
    inventory_detection_files,
    load_camera_detections,
)
from deep_oc_sort_3d.local_tracker_benchmark.downstream_probe import compute_downstream_probe
from deep_oc_sort_3d.local_tracker_benchmark.gt_local_diagnostics import compute_gt_diagnostics
from deep_oc_sort_3d.local_tracker_benchmark.local_track_io import (
    progress_iter,
    read_track_rows,
    write_csv_rows,
    write_json,
    write_track_records,
)
from deep_oc_sort_3d.local_tracker_benchmark.local_track_metrics import (
    compute_person_vs_nonperson,
    compute_track_metrics,
)
from deep_oc_sort_3d.local_tracker_benchmark.reid_adapters import build_reid_adapter


def run_local_tracker_benchmark(
    config_path: Path,
    subset_name: str = "quick",
    requested_trackers: Optional[List[str]] = None,
    progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Run selected trackers camera-by-camera and produce comparison artifacts."""
    config = load_benchmark_config(config_path)
    root = output_root_from_config(config)
    if overwrite and root.exists():
        shutil.rmtree(str(root))
    _prepare_dirs(root)
    _write_yaml(root / "configs" / "resolved_config.yaml", config)
    scene_selection = resolve_scene_selection(config, subset_name)
    pipeline_root = _resolve_pipeline_root(config, scene_selection)
    inventory, warnings = inventory_detection_files(pipeline_root, scene_selection)
    warnings.append("current_local_tracker runtime measures existing-output loading/copying, not original tracking runtime")
    write_json(root / "inputs" / "detection_inventory.json", {"pipeline_root": str(pipeline_root), "files": inventory})
    write_csv_rows(root / "inputs" / "scene_camera_inventory.csv", inventory)
    write_csv_rows(root / "inputs" / "benchmark_subset.csv", [
        {"subset": item[0], "split": item[1], "scene_name": item[2]} for item in scene_selection
    ])
    tracker_names = enabled_tracker_names(config, requested_trackers)
    run_statuses = []
    for tracker_name in progress_iter(tracker_names, progress, "local tracker variants"):
        run_root = root / "tracker_runs" / tracker_name
        for directory in ("local_tracks", "summaries", "diagnostics"):
            (run_root / directory).mkdir(parents=True, exist_ok=True)
        status_path = run_root / "summaries" / "run_status.json"
        if skip_existing and status_path.exists():
            run_statuses.append({"tracker_name": tracker_name, "status": "skipped_existing"})
            continue
        start_time = time.time()
        try:
            if tracker_name == "current_local_tracker":
                status = load_current_tracker_subset(
                    Path(str(config.get("paths", {}).get("current_local_tracks_root"))), inventory, run_root
                )
            else:
                status = _run_tracker_variant(tracker_name, config, inventory, run_root, progress, warnings)
        except Exception as exc:
            status = {"status": "error", "reason": str(exc)}
        status["tracker_name"] = tracker_name
        status["runtime_seconds"] = time.time() - start_time
        write_json(run_root / "runtime.json", {"runtime_seconds": status["runtime_seconds"]})
        write_json(status_path, status)
        _write_yaml(run_root / "tracker_config_resolved.yaml", _tracker_config(tracker_name, config))
        run_statuses.append(status)
    comparison = summarize_benchmark_outputs(root, config, run_statuses, warnings)
    return {"output_root": str(root), "runs": run_statuses, "comparison": comparison}


def summarize_benchmark_outputs(
    root: Path,
    config: Optional[Dict[str, Any]] = None,
    run_statuses: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute all tables from completed tracker run folders."""
    if warnings is None:
        warning_payload = _read_json_dict(root / "diagnostics" / "warnings.json")
        warnings = list(warning_payload.get("warnings", []))
    rows = []
    per_scene = []
    per_camera = []
    per_class = []
    person_nonperson = []
    probes = []
    runtime_rows = []
    statuses = run_statuses or _load_run_statuses(root)
    for status in statuses:
        tracker_name = str(status.get("tracker_name", ""))
        if status.get("status") not in ("ok", "skipped_existing"):
            rows.append({"tracker_name": tracker_name, "status": status.get("status"), "skip_reason": status.get("reason", "")})
            continue
        track_rows = _load_tracker_rows(root / "tracker_runs" / tracker_name / "local_tracks")
        metric = compute_track_metrics(track_rows)
        metric.update(compute_gt_diagnostics(track_rows))
        groups = compute_person_vs_nonperson(track_rows)
        person = _group_row(groups, "Person")
        nonperson = _group_row(groups, "NonPerson")
        metric.update(
            {
                "tracker_name": tracker_name,
                "status": "ok",
                "runtime_seconds": status.get("runtime_seconds", _runtime(root, tracker_name)),
                "person_num_tracks": person.get("num_tracks"),
                "person_mean_track_length": person.get("mean_track_length"),
                "person_median_track_length": person.get("median_track_length"),
                "person_len1_tracks": person.get("num_length_1_tracks"),
                "person_short_ratio_le3": person.get("short_track_ratio_le3"),
                "person_short_ratio_le5": person.get("short_track_ratio_le5"),
                "nonperson_short_track_ratio_le3": nonperson.get("short_track_ratio_le3"),
            }
        )
        rows.append(metric)
        runtime_rows.append({"tracker_name": tracker_name, "runtime_seconds": metric.get("runtime_seconds")})
        for row in _metric_rows_with_gt(track_rows, ["subset", "scene_name"]):
            row["tracker_name"] = tracker_name
            per_scene.append(row)
        for row in _metric_rows_with_gt(track_rows, ["subset", "scene_name", "camera_id"]):
            row["tracker_name"] = tracker_name
            per_camera.append(row)
        for row in _metric_rows_with_gt(track_rows, ["class_id", "class_name"]):
            row["tracker_name"] = tracker_name
            per_class.append(row)
        for row in groups:
            row["tracker_name"] = tracker_name
            person_nonperson.append(row)
        probe = compute_downstream_probe(track_rows)
        probe["tracker_name"] = tracker_name
        probes.append(probe)
        write_json(
            root / "tracker_runs" / tracker_name / "summaries" / "tracker_metrics.json",
            metric,
        )
        write_csv_rows(
            root / "tracker_runs" / tracker_name / "diagnostics" / "person_vs_nonperson.csv",
            groups,
        )
    selected = select_local_tracker(rows)
    write_csv_rows(root / "comparison" / "local_tracker_benchmark_summary.csv", rows)
    write_json(root / "comparison" / "local_tracker_benchmark_summary.json", {"trackers": rows})
    write_json(root / "comparison" / "selected_local_tracker_candidate.json", selected)
    write_csv_rows(root / "diagnostics" / "local_track_length_summary.csv", rows)
    write_json(root / "diagnostics" / "local_track_length_summary.json", {"trackers": rows})
    write_csv_rows(root / "diagnostics" / "short_track_diagnostics.csv", _short_diagnostics(rows))
    write_csv_rows(root / "diagnostics" / "per_scene_metrics.csv", per_scene)
    write_csv_rows(root / "diagnostics" / "per_camera_metrics.csv", per_camera)
    write_csv_rows(root / "diagnostics" / "per_class_metrics.csv", per_class)
    write_csv_rows(root / "diagnostics" / "person_vs_nonperson_summary.csv", person_nonperson)
    write_csv_rows(root / "diagnostics" / "tracker_runtime_summary.csv", runtime_rows)
    write_json(root / "diagnostics" / "warnings.json", {"warnings": warnings or []})
    write_csv_rows(root / "downstream_probe" / "tracklet_count_probe.csv", probes)
    write_csv_rows(root / "downstream_probe" / "candidate_count_probe.csv", probes)
    write_csv_rows(root / "downstream_probe" / "motion_quality_probe.csv", probes)
    write_csv_rows(
        root / "downstream_probe" / "optional_mini_global_probe.csv",
        [],
        ("tracker_name", "status", "reason"),
    )
    write_benchmark_report(
        root / "comparison" / "LOCAL_TRACKER_BENCHMARK_REPORT.md",
        rows,
        selected,
        warnings or [],
        probes=probes,
    )
    if config is None or bool(config.get("figures", {}).get("enabled", True)):
        create_benchmark_figures(rows, root / "figures", per_class_rows=per_class)
    return {"rows": rows, "selected": selected}


def _run_tracker_variant(
    tracker_name: str,
    config: Dict[str, Any],
    inventory: Sequence[Dict[str, Any]],
    run_root: Path,
    progress: bool,
    warnings: List[str],
) -> Dict[str, Any]:
    if "sbs" in tracker_name:
        adapter_status = build_reid_adapter(tracker_name, config, None)
        reason = str(adapter_status.get("reason", "SBS adapter unavailable"))
        warnings.append("%s skipped: %s" % (tracker_name, reason))
        return {"status": "skipped", "reason": reason}
    files = 0
    records = 0
    camera_errors = []
    reid_embedded = 0
    reid_backend = None
    reid_backend_name = None
    reid_weights_loaded = None
    for item in progress_iter(list(inventory), progress, "%s cameras" % tracker_name):
        try:
            detections = load_camera_detections(
                item, float(config.get("detections", {}).get("min_confidence_for_input", 0.001))
            )
            use_reid = tracker_name not in ("bytetrack_style_yolo11m", "botsort_style_no_reid_yolo11m")
            if use_reid:
                video = _find_video(config, item)
                adapter_status = build_reid_adapter(
                    tracker_name,
                    config,
                    video,
                    backend_instance=reid_backend,
                )
                if adapter_status.get("status") != "ok":
                    warnings.append("%s %s/%s skipped ReID: %s" % (tracker_name, item["scene_name"], item["camera_id"], adapter_status.get("reason")))
                    continue
                reid_backend = adapter_status.get("backend_instance")
                reid_backend_name = adapter_status.get("backend")
                reid_weights_loaded = adapter_status.get("weights_loaded")
                attach_summary = adapter_status["adapter"].attach(detections)
                reid_embedded += int(attach_summary.get("person_embeddings", 0))
            tracker = _make_tracker(tracker_name, config)
            output = tracker.run(group_detections_by_frame(detections))
            target = run_root / "local_tracks" / str(item["subset"]) / str(item["scene_name"]) / (str(item["camera_id"]) + ".csv")
            write_track_records(target, output)
            files += 1
            records += len(output)
        except Exception as exc:
            camera_errors.append("%s/%s: %s" % (item.get("scene_name"), item.get("camera_id"), exc))
    return {
        "status": "ok" if files else "skipped",
        "reason": "no camera completed; inspect warnings and camera_errors" if not files else "",
        "files": files,
        "records": records,
        "camera_errors": camera_errors,
        "reid_embeddings": reid_embedded,
        "reid_backend": reid_backend_name,
        "reid_weights_loaded": reid_weights_loaded,
    }


def _make_tracker(name: str, config: Dict[str, Any]) -> Any:
    if name == "bytetrack_style_yolo11m":
        return ByteTrackStyleTracker(config.get("bytetrack_style", {}))
    return BoTSORTStyleTracker(config.get("botsort_style", {}), use_reid=name not in ("botsort_style_no_reid_yolo11m",))


def _resolve_pipeline_root(config: Dict[str, Any], selection: Sequence[Any]) -> Path:
    paths = config.get("paths", {})
    primary = Path(str(paths.get("yolo_pipeline_root", "")))
    fallback = Path(str(paths.get("v2_pipeline_root", "")))
    if primary.exists() and (primary / "detections2d").exists():
        return primary
    return fallback


def _find_video(config: Dict[str, Any], item: Dict[str, Any]) -> Optional[Path]:
    dataset_root = Path(str(config.get("paths", {}).get("dataset_root", "")))
    videos_dir = dataset_root / str(item.get("split", "")) / str(item.get("scene_name", "")) / "videos"
    for path in list_video_files(videos_dir):
        if path.stem == str(item.get("camera_id", "")):
            return path
    return None


def _load_tracker_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*.csv")) if root.exists() else []:
        rows.extend(read_track_rows(path))
    return rows


def _load_run_statuses(root: Path) -> List[Dict[str, Any]]:
    statuses = []
    for path in sorted((root / "tracker_runs").glob("*/summaries/run_status.json")):
        import json

        statuses.append(json.loads(path.read_text(encoding="utf-8")))
    return statuses


def _read_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _prepare_dirs(root: Path) -> None:
    for name in ("configs", "inputs", "tracker_runs", "diagnostics", "downstream_probe", "comparison", "figures"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _tracker_config(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    key = "bytetrack_style" if name == "bytetrack_style_yolo11m" else "botsort_style"
    return {"tracker_name": name, "settings": config.get(key, {})}


def _write_yaml(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _group_row(rows: List[Dict[str, Any]], group: str) -> Dict[str, Any]:
    for row in rows:
        if row.get("group") == group:
            return row
    return {}


def _runtime(root: Path, tracker_name: str) -> Any:
    path = root / "tracker_runs" / tracker_name / "runtime.json"
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8")).get("runtime_seconds")


def _short_diagnostics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "tracker_name": row.get("tracker_name"),
            "num_length_1_tracks": row.get("num_length_1_tracks"),
            "num_length_le_3_tracks": row.get("num_length_le_3_tracks"),
            "num_length_le_5_tracks": row.get("num_length_le_5_tracks"),
            "short_track_ratio_len1": row.get("short_track_ratio_len1"),
            "short_track_ratio_le3": row.get("short_track_ratio_le3"),
            "short_track_ratio_le5": row.get("short_track_ratio_le5"),
        }
        for row in rows
    ]


def _metric_rows_with_gt(
    rows: Sequence[Dict[str, Any]],
    fields: Sequence[str],
) -> List[Dict[str, Any]]:
    groups = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        metric = compute_track_metrics(values)
        metric.update(compute_gt_diagnostics(values))
        for field, value in zip(fields, key):
            metric[field] = value
        output.append(metric)
    return output
