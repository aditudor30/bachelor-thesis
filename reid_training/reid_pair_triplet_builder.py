"""Build positive pairs, negative pairs, and triplets for Person ReID."""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.reid_training.reid_dataset_config import output_root_from_config
from deep_oc_sort_3d.reid_training.reid_dataset_io import group_by, read_csv_rows, write_csv_rows, write_json


PAIR_FIELDS = [
    "pair_id",
    "split",
    "pair_type",
    "anchor_path",
    "other_path",
    "anchor_identity_id",
    "other_identity_id",
    "anchor_scene",
    "other_scene",
    "anchor_camera",
    "other_camera",
    "anchor_frame",
    "other_frame",
    "same_identity",
    "same_camera",
    "same_scene",
]


TRIPLET_FIELDS = [
    "triplet_id",
    "split",
    "anchor_path",
    "positive_path",
    "negative_path",
    "anchor_identity_id",
    "positive_identity_id",
    "negative_identity_id",
    "anchor_scene",
    "positive_scene",
    "negative_scene",
    "anchor_camera",
    "positive_camera",
    "negative_camera",
    "anchor_frame",
    "positive_frame",
    "negative_frame",
    "positive_same_camera",
    "negative_same_camera",
    "negative_same_scene",
]


def build_pairs_triplets_from_config(config: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    """Build pair/triplet CSV files from crop metadata."""
    output_root = output_root_from_config(config)
    metadata_path = output_root / "metadata" / "all_crops.csv"
    rows, _fields = read_csv_rows(metadata_path)
    valid_rows = [row for row in rows if str(row.get("is_valid_crop", "")) == "1"]
    pair_cfg = config.get("pairs_triplets", {})
    if not bool(pair_cfg.get("enabled", True)):
        return {"status": "disabled"}
    root = output_root / "pairs_triplets"
    seed = int(pair_cfg.get("random_seed", 42))
    summary: Dict[str, Any] = {"status": "ok"}
    for split in ["train", "val"]:
        split_rows = [row for row in valid_rows if row.get("split") == split]
        positive = build_positive_pairs(split_rows, split, int(pair_cfg.get("max_positive_pairs_per_identity", 50)), seed)
        negative = build_negative_pairs(split_rows, split, int(pair_cfg.get("max_negative_pairs_per_identity", 100)), seed + 11)
        triplets = build_triplets(split_rows, split, int(pair_cfg.get("max_triplets_per_identity", 100)), seed + 23)
        write_csv_rows(positive, root / ("positive_pairs_%s.csv" % split), PAIR_FIELDS)
        write_csv_rows(negative, root / ("negative_pairs_%s.csv" % split), PAIR_FIELDS)
        write_csv_rows(triplets, root / ("triplets_%s.csv" % split), TRIPLET_FIELDS)
        summary["positive_pairs_%s" % split] = len(positive)
        summary["negative_pairs_%s" % split] = len(negative)
        summary["triplets_%s" % split] = len(triplets)
    write_json(summary, root / "pairs_triplets_summary.json")
    return summary


def build_positive_pairs(rows: List[Dict[str, Any]], split: str, max_pairs_per_identity: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Build positive pairs from same identity."""
    rng = random.Random(seed)
    output: List[Dict[str, Any]] = []
    groups = group_by(rows, "identity_id")
    for identity, group_rows in sorted(groups.items()):
        if len(group_rows) < 2:
            continue
        pairs = _sample_positive_pairs(group_rows, max_pairs_per_identity, rng)
        for anchor, other in pairs:
            output.append(pair_row(len(output), split, "positive", anchor, other))
    return output


def build_negative_pairs(rows: List[Dict[str, Any]], split: str, max_pairs_per_identity: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Build negative pairs from different identities."""
    rng = random.Random(seed)
    output: List[Dict[str, Any]] = []
    groups = group_by(rows, "identity_id")
    all_rows = list(rows)
    for identity, group_rows in sorted(groups.items()):
        negatives = [row for row in all_rows if str(row.get("identity_id", "")) != identity]
        if not negatives:
            continue
        anchors = _sample_rows(group_rows, max_pairs_per_identity, rng)
        for anchor in anchors:
            other = choose_negative(anchor, negatives, rng)
            if other is not None:
                output.append(pair_row(len(output), split, "negative", anchor, other))
    return output


def build_triplets(rows: List[Dict[str, Any]], split: str, max_triplets_per_identity: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Build triplets anchor-positive-negative."""
    rng = random.Random(seed)
    output: List[Dict[str, Any]] = []
    groups = group_by(rows, "identity_id")
    all_rows = list(rows)
    for identity, group_rows in sorted(groups.items()):
        if len(group_rows) < 2:
            continue
        negatives = [row for row in all_rows if str(row.get("identity_id", "")) != identity]
        if not negatives:
            continue
        positives = _sample_positive_pairs(group_rows, max_triplets_per_identity, rng)
        for anchor, positive in positives:
            negative = choose_negative(anchor, negatives, rng)
            if negative is not None:
                output.append(triplet_row(len(output), split, anchor, positive, negative))
    return output


def choose_negative(anchor: Dict[str, Any], negatives: List[Dict[str, Any]], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Choose a negative, preferring same scene/camera hard negatives when present."""
    same_camera = [row for row in negatives if row.get("camera_id") == anchor.get("camera_id")]
    same_scene = [row for row in negatives if row.get("scene_name") == anchor.get("scene_name")]
    diff_camera = [row for row in negatives if row.get("camera_id") != anchor.get("camera_id")]
    pools = [same_camera, same_scene, diff_camera, negatives]
    for pool in pools:
        if pool:
            return rng.choice(pool)
    return None


def pair_row(pair_id: int, split: str, pair_type: str, anchor: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """Return pair CSV row."""
    same_identity = str(anchor.get("identity_id", "")) == str(other.get("identity_id", ""))
    same_camera = str(anchor.get("camera_id", "")) == str(other.get("camera_id", ""))
    same_scene = str(anchor.get("scene_name", "")) == str(other.get("scene_name", ""))
    return {
        "pair_id": "%s_%s_%06d" % (split, pair_type, pair_id),
        "split": split,
        "pair_type": pair_type,
        "anchor_path": anchor.get("crop_path", ""),
        "other_path": other.get("crop_path", ""),
        "anchor_identity_id": anchor.get("identity_id", ""),
        "other_identity_id": other.get("identity_id", ""),
        "anchor_scene": anchor.get("scene_name", ""),
        "other_scene": other.get("scene_name", ""),
        "anchor_camera": anchor.get("camera_id", ""),
        "other_camera": other.get("camera_id", ""),
        "anchor_frame": anchor.get("frame_id", ""),
        "other_frame": other.get("frame_id", ""),
        "same_identity": int(same_identity),
        "same_camera": int(same_camera),
        "same_scene": int(same_scene),
    }


def triplet_row(triplet_id: int, split: str, anchor: Dict[str, Any], positive: Dict[str, Any], negative: Dict[str, Any]) -> Dict[str, Any]:
    """Return triplet CSV row."""
    return {
        "triplet_id": "%s_triplet_%06d" % (split, triplet_id),
        "split": split,
        "anchor_path": anchor.get("crop_path", ""),
        "positive_path": positive.get("crop_path", ""),
        "negative_path": negative.get("crop_path", ""),
        "anchor_identity_id": anchor.get("identity_id", ""),
        "positive_identity_id": positive.get("identity_id", ""),
        "negative_identity_id": negative.get("identity_id", ""),
        "anchor_scene": anchor.get("scene_name", ""),
        "positive_scene": positive.get("scene_name", ""),
        "negative_scene": negative.get("scene_name", ""),
        "anchor_camera": anchor.get("camera_id", ""),
        "positive_camera": positive.get("camera_id", ""),
        "negative_camera": negative.get("camera_id", ""),
        "anchor_frame": anchor.get("frame_id", ""),
        "positive_frame": positive.get("frame_id", ""),
        "negative_frame": negative.get("frame_id", ""),
        "positive_same_camera": int(str(anchor.get("camera_id", "")) == str(positive.get("camera_id", ""))),
        "negative_same_camera": int(str(anchor.get("camera_id", "")) == str(negative.get("camera_id", ""))),
        "negative_same_scene": int(str(anchor.get("scene_name", "")) == str(negative.get("scene_name", ""))),
    }


def _sample_positive_pairs(rows: List[Dict[str, Any]], max_pairs: int, rng: random.Random) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    ordered = sorted(rows, key=lambda item: (str(item.get("camera_id", "")), int(item.get("frame_id", -1))))
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    cross_camera = _all_pairs(ordered, require_cross_camera=True)
    same_camera = _all_pairs(ordered, require_cross_camera=False)
    candidates = cross_camera if cross_camera else same_camera
    rng.shuffle(candidates)
    for pair in candidates[: int(max_pairs)]:
        pairs.append(pair)
    return pairs


def _all_pairs(rows: List[Dict[str, Any]], require_cross_camera: bool) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    max_source = min(len(rows), 200)
    sampled = rows[:max_source]
    for left_index in range(len(sampled)):
        for right_index in range(left_index + 1, len(sampled)):
            left = sampled[left_index]
            right = sampled[right_index]
            if require_cross_camera and left.get("camera_id") == right.get("camera_id"):
                continue
            pairs.append((left, right))
    return pairs


def _sample_rows(rows: List[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    if len(rows) <= int(count):
        return list(rows)
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[: int(count)]

