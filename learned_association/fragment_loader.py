"""Automatic fragment-source discovery and loading for Step 20A."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.learned_association.fragment_feature_builder import (
    aggregate_observation_rows,
    normalize_fragment_record,
)
from deep_oc_sort_3d.learned_association.pair_dataset_config import configured_scenes
from deep_oc_sort_3d.learned_association.pair_dataset_io import progress_iter, safe_int


SOURCE_CONFIG_KEYS = {
    "motion_clean_candidates": "v2_motion_clean_root",
    "tracklets": "v2_tracklets_root",
    "final_frame_records": "v2_final_export_root",
    "local_tracks": "v2_local_tracks_root",
}


def choose_fragment_source(config: Dict[str, Any]) -> Tuple[str, Path]:
    """Choose the first existing configured fragment source."""
    source_config = config.get("fragment_source", {})
    preferred = str(source_config.get("preferred_source", "motion_clean_candidates"))
    ordered = [preferred] + list(source_config.get("fallback_sources", []))
    checked = []
    for source_name in ordered:
        if source_name in checked:
            continue
        checked.append(source_name)
        path_key = SOURCE_CONFIG_KEYS.get(source_name)
        path_value = config.get("paths", {}).get(path_key) if path_key else None
        if path_value and Path(path_value).is_dir():
            return source_name, Path(path_value)
    raise FileNotFoundError(
        "No configured fragment source exists. Checked: %s" % ", ".join(checked)
    )


def load_person_fragments(
    config: Dict[str, Any],
    debug_limit_scenes: Optional[int] = None,
    progress: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and normalize Person fragments from the selected source."""
    source_name, source_root = choose_fragment_source(config)
    scenes = configured_scenes(config, None)
    if debug_limit_scenes is not None:
        scenes = select_debug_scenes_with_files(
            scenes, source_root, source_name, debug_limit_scenes
        )
    fragments = []  # type: List[Dict[str, Any]]
    file_counts = {}  # type: Dict[str, int]
    warnings = []  # type: List[str]
    for scene in progress_iter(scenes, "loading fragments", progress, len(scenes)):
        split = scene["split"]
        scene_name = scene["scene_name"]
        files = discover_scene_files(source_root, split, scene_name, source_name)
        file_counts["%s/%s" % (split, scene_name)] = len(files)
        if not files:
            warnings.append("No %s files for %s/%s" % (source_name, split, scene_name))
            continue
        scene_fragments = load_scene_fragments(files, source_name, split, scene_name)
        fragments.extend(scene_fragments)

    source_settings = config.get("fragment_source", {})
    min_length = int(source_settings.get("min_fragment_length", 3))
    min_confidence = float(source_settings.get("min_mean_confidence", 0.05))
    filtered = []
    for fragment in fragments:
        if safe_int(fragment.get("class_id"), 0) != 0:
            continue
        if str(fragment.get("class_name", "Person")).lower() != "person":
            continue
        fragment["pre_gt_valid"] = bool(
            int(fragment.get("num_observations") or 0) >= min_length
            and float(fragment.get("mean_confidence") or 0.0) >= min_confidence
        )
        filtered.append(fragment)

    embedding_summary = attach_fragment_embeddings(filtered, config)
    summary = {
        "source_name": source_name,
        "source_root": str(source_root),
        "num_loaded": len(fragments),
        "num_person": len(filtered),
        "file_counts": file_counts,
        "warnings": warnings,
        "embedding_summary": embedding_summary,
    }
    return filtered, summary


def discover_scene_files(
    root: Path, split: str, scene_name: str, source_name: str
) -> List[Path]:
    """Discover source files for one scene, preferring JSONL over CSV twins."""
    if source_name == "motion_clean_candidates":
        jsonl_files = _matching_files(root, "*_clean_candidates.jsonl", scene_name)
        if jsonl_files:
            return jsonl_files
        return _matching_files(root, "*_clean_candidates.csv", scene_name)
    if source_name == "tracklets":
        jsonl_files = _matching_files(root, "*tracklet*.jsonl", scene_name)
        if jsonl_files:
            return jsonl_files
        return _matching_files(root, "*tracklet*.csv", scene_name)
    candidates = []  # type: List[Path]
    for suffix in ("*.jsonl", "*.csv"):
        for path in root.rglob(suffix):
            normalized = str(path).replace("\\", "/")
            if scene_name not in normalized:
                continue
            if split and split not in normalized and split not in ("train", "val"):
                continue
            if source_name == "tracklets" and "tracklet" not in path.name.lower():
                continue
            candidates.append(path)

    by_key = {}  # type: Dict[str, Path]
    for path in sorted(candidates):
        key = str(path.with_suffix(""))
        current = by_key.get(key)
        if current is None or path.suffix.lower() == ".jsonl":
            by_key[key] = path
    return sorted(by_key.values())


def select_debug_scenes_with_files(
    scenes: Sequence[Dict[str, str]],
    source_root: Path,
    source_name: str,
    limit_per_split: int,
) -> List[Dict[str, str]]:
    """Select the first available debug scenes instead of empty configured scenes."""
    selected = []  # type: List[Dict[str, str]]
    counts = {}  # type: Dict[str, int]
    for scene in scenes:
        split = scene["split"]
        if counts.get(split, 0) >= max(0, limit_per_split):
            continue
        files = discover_scene_files(
            source_root, split, scene["scene_name"], source_name
        )
        if not files:
            continue
        selected.append(scene)
        counts[split] = counts.get(split, 0) + 1
    return selected


def _matching_files(root: Path, pattern: str, scene_name: str) -> List[Path]:
    """Return files matching both the canonical filename and scene name."""
    return sorted(
        path
        for path in root.rglob(pattern)
        if scene_name in str(path).replace("\\", "/")
    )


def load_scene_fragments(
    files: Sequence[Path], source_name: str, split: str, scene_name: str
) -> List[Dict[str, Any]]:
    """Load one scene from candidate/tracklet or frame-level files."""
    raw_rows = []  # type: List[Dict[str, Any]]
    for path in files:
        raw_rows.extend(iter_records(path))
    if source_name in ("final_frame_records", "local_tracks"):
        return aggregate_observation_rows(raw_rows, source_name, split, scene_name)
    return [
        normalize_fragment_record(row, source_name, split, scene_name)
        for row in raw_rows
        if is_person_record(row)
    ]


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield dictionaries from JSONL or CSV."""
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    yield value
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield dict(row)


def is_person_record(record: Dict[str, Any]) -> bool:
    """Return whether a source row represents Person."""
    class_id = safe_int(record.get("class_id"))
    class_name = str(record.get("class_name") or record.get("object_type") or "")
    return class_id == 0 or class_name.lower() == "person"


def attach_fragment_embeddings(
    fragments: Sequence[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Attach memory-mapped embeddings using several compatible key forms."""
    paths = config.get("paths", {})
    npy_path = Path(str(paths.get("fragment_embeddings_npy", "")))
    index_path = Path(str(paths.get("fragment_embeddings_index_csv", "")))
    if not npy_path.is_file() or not index_path.is_file():
        return {
            "status": "missing",
            "npy_path": str(npy_path),
            "index_path": str(index_path),
            "attached": 0,
        }
    embeddings = np.load(str(npy_path), mmap_mode="r")
    index_rows = list(iter_records(index_path))
    lookup = build_embedding_lookup(index_rows)
    attached = 0
    for fragment in fragments:
        row = find_embedding_row(fragment, lookup)
        if row is None:
            continue
        embedding_index = safe_int(
            row.get("embedding_index") or row.get("row_index") or row.get("index")
        )
        if embedding_index is None or embedding_index < 0 or embedding_index >= len(embeddings):
            continue
        vector = np.asarray(embeddings[embedding_index], dtype=np.float32)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            continue
        fragment["_embedding"] = vector
        fragment["embedding_available"] = True
        fragment["embedding_index"] = embedding_index
        attached += 1
    return {
        "status": "ok",
        "embedding_rows": len(index_rows),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "attached": attached,
        "missing": len(fragments) - attached,
    }


def build_embedding_lookup(
    rows: Sequence[Dict[str, Any]]
) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    """Index embedding metadata by split, scene, id kind and id value."""
    lookup = {}  # type: Dict[Tuple[str, str, str, str], Dict[str, Any]]
    id_fields = (
        "fragment_id",
        "fragment_embedding_id",
        "candidate_id",
        "candidate_id_optional",
        "tracklet_id",
        "tracklet_id_optional",
        "global_track_id",
        "local_track_id",
        "local_track_id_optional",
        "embedding_id",
    )
    for row in rows:
        split = str(row.get("split") or row.get("subset") or "")
        scene = str(row.get("scene_name") or "")
        for field in id_fields:
            value = row.get(field)
            if value is not None and value != "":
                lookup[(split, scene, field, str(value))] = row
                lookup.setdefault(("", scene, field, str(value)), row)
                alias = field.replace("_optional", "")
                if alias != field:
                    lookup[(split, scene, alias, str(value))] = row
                    lookup.setdefault(("", scene, alias, str(value)), row)
    return lookup


def find_embedding_row(
    fragment: Dict[str, Any],
    lookup: Dict[Tuple[str, str, str, str], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve an embedding row for a fragment."""
    split = str(fragment.get("split") or "")
    scene = str(fragment.get("scene_name") or "")
    for field in (
        "fragment_id",
        "fragment_embedding_id",
        "candidate_id",
        "tracklet_id",
        "global_track_id",
        "local_track_id",
        "embedding_id",
    ):
        value = fragment.get(field)
        if value is None or value == "":
            continue
        for split_key in (split, ""):
            row = lookup.get((split_key, scene, field, str(value)))
            if row is not None:
                return row
    return None
