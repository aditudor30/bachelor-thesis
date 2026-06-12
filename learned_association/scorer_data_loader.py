"""Load Step 20A pair CSVs for scorer training and evaluation."""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.learned_association.pair_dataset_io import progress_iter, read_csv_rows, safe_int


METADATA_FIELDS = (
    "pair_id",
    "split",
    "scene_name",
    "scene_id",
    "camera_pair",
    "camera_a",
    "camera_b",
    "fragment_a_id",
    "fragment_b_id",
    "gt_identity_a",
    "gt_identity_b",
    "same_identity",
    "hard_negative",
    "pair_type",
    "reid_similarity",
    "temporal_gap",
    "center_mean_distance_3d",
)


def load_pair_splits(
    config: Dict[str, Any], progress: bool = True
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load balanced train and validation rows."""
    paths = config.get("paths", {})
    train_path = Path(str(paths.get("train_pairs_csv", "")))
    val_path = Path(str(paths.get("val_pairs_csv", "")))
    if not train_path.is_file():
        raise FileNotFoundError("Training pairs CSV not found: %s" % train_path)
    if not val_path.is_file():
        raise FileNotFoundError("Validation pairs CSV not found: %s" % val_path)
    files = [("train", train_path), ("val", val_path)]
    loaded = {}
    for split, path in progress_iter(files, "loading pair data", progress, len(files)):
        loaded[split] = read_csv_rows(path)
    if not loaded.get("train"):
        raise ValueError("Training pairs are empty: %s" % train_path)
    if not loaded.get("val"):
        raise ValueError("Validation pairs are empty: %s" % val_path)
    return loaded["train"], loaded["val"]


def labels_from_rows(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Extract binary labels."""
    labels = [safe_int(row.get("same_identity"), 0) or 0 for row in rows]
    array = np.asarray(labels, dtype=np.float32)
    unique = set(int(value) for value in array.tolist())
    if not unique.issubset({0, 1}):
        raise ValueError("same_identity must contain only 0/1 labels")
    return array


def metadata_from_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Retain provenance columns needed by grouped evaluation."""
    return [{field: row.get(field, "") for field in METADATA_FIELDS} for row in rows]


def validate_scene_disjointness(
    train_rows: Sequence[Dict[str, Any]], val_rows: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Check that scene-level train/val leakage is absent."""
    train_scenes = sorted({str(row.get("scene_name") or "") for row in train_rows})
    val_scenes = sorted({str(row.get("scene_name") or "") for row in val_rows})
    overlap = sorted(set(train_scenes).intersection(val_scenes))
    return {
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "overlap_scenes": overlap,
        "scene_disjoint": not overlap,
    }
