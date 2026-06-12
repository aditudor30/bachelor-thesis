"""Controlled positive and negative fragment-pair generation."""

import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.learned_association.pair_dataset_io import progress_iter
from deep_oc_sort_3d.learned_association.pair_feature_builder import cosine_similarity


def build_candidate_pairs(
    fragments: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Build reproducible candidate pairs without enumerating every combination."""
    settings = config.get("pair_generation", {})
    seed = int(config.get("person_association_pair_dataset", {}).get("random_seed", 42))
    random_generator = random.Random(seed)
    by_scene = defaultdict(list)  # type: Dict[Tuple[str, str], List[Dict[str, Any]]]
    for fragment in fragments:
        if not fragment.get("valid_for_pairs"):
            continue
        if fragment.get("gt_identity_id") in (None, "", "unknown"):
            continue
        key = (str(fragment.get("split") or ""), str(fragment.get("scene_name") or ""))
        by_scene[key].append(fragment)

    pairs = []  # type: List[Dict[str, Any]]
    scene_items = sorted(by_scene.items())
    for (split, scene_name), scene_fragments in progress_iter(
        scene_items, "pair generation", progress, len(scene_items)
    ):
        scene_pairs = build_scene_candidate_pairs(
            scene_fragments, split, scene_name, settings, random_generator
        )
        max_scene = int(settings.get("max_pairs_per_scene", 200000))
        if len(scene_pairs) > max_scene:
            positives = [row for row in scene_pairs if int(row["same_identity"]) == 1]
            negatives = [row for row in scene_pairs if int(row["same_identity"]) == 0]
            random_generator.shuffle(negatives)
            scene_pairs = positives[:max_scene]
            remaining = max(0, max_scene - len(scene_pairs))
            scene_pairs.extend(negatives[:remaining])
        pairs.extend(scene_pairs)
    return pairs


def build_scene_candidate_pairs(
    fragments: Sequence[Dict[str, Any]],
    split: str,
    scene_name: str,
    settings: Dict[str, Any],
    random_generator: random.Random,
) -> List[Dict[str, Any]]:
    """Build bounded positive and negative pairs for one scene."""
    by_identity = defaultdict(list)  # type: Dict[str, List[Dict[str, Any]]]
    for fragment in fragments:
        by_identity[str(fragment["gt_identity_id"])].append(fragment)

    max_positive = int(settings.get("max_positive_pairs_per_identity", 100))
    max_negative = int(settings.get("max_negative_pairs_per_identity", 300))
    hard_top_k = int(settings.get("hard_negative_top_k", 100))
    rows = []  # type: List[Dict[str, Any]]
    seen = set()

    for identity in sorted(by_identity.keys()):
        identity_fragments = by_identity[identity]
        positive_options = _sample_positive_options(
            identity_fragments, max_positive, random_generator
        )
        positive_options.sort(key=_positive_priority)
        if len(positive_options) > max_positive:
            top_cross_camera = [pair for pair in positive_options if not _same_camera(pair[0], pair[1])]
            selected = top_cross_camera[:max_positive]
            if len(selected) < max_positive:
                remaining = [pair for pair in positive_options if pair not in selected]
                random_generator.shuffle(remaining)
                selected.extend(remaining[: max_positive - len(selected)])
            positive_options = selected
        for fragment_a, fragment_b in positive_options:
            row = make_candidate_pair(fragment_a, fragment_b, split, scene_name, 1, "positive")
            if row["pair_id"] not in seen:
                seen.add(row["pair_id"])
                rows.append(row)

        other_fragments = [
            fragment
            for other_identity, values in by_identity.items()
            if other_identity != identity
            for fragment in values
        ]
        negative_options = []  # type: List[Tuple[float, Dict[str, Any], Dict[str, Any]]]
        for fragment_a in identity_fragments:
            sampled_others = _bounded_sample(
                other_fragments,
                max(max_negative * 4, hard_top_k * 4),
                random_generator,
            )
            for fragment_b in sampled_others:
                similarity = cosine_similarity(fragment_a.get("_embedding"), fragment_b.get("_embedding"))
                score = similarity if similarity is not None else -1.0
                negative_options.append((score, fragment_a, fragment_b))

        negative_options.sort(key=lambda item: item[0], reverse=True)
        hard = negative_options[: min(hard_top_k, max_negative)] if settings.get("include_hard_negatives", True) else []
        remaining_options = negative_options[len(hard) :]
        random_generator.shuffle(remaining_options)
        selected_negatives = hard + remaining_options[: max(0, max_negative - len(hard))]
        for index, (_, fragment_a, fragment_b) in enumerate(selected_negatives):
            row = make_candidate_pair(
                fragment_a,
                fragment_b,
                split,
                scene_name,
                0,
                "hard_negative" if index < len(hard) else "random_negative",
            )
            if row["pair_id"] not in seen:
                seen.add(row["pair_id"])
                rows.append(row)
    return rows


def make_candidate_pair(
    fragment_a: Dict[str, Any],
    fragment_b: Dict[str, Any],
    split: str,
    scene_name: str,
    same_identity: int,
    pair_type: str,
) -> Dict[str, Any]:
    """Create an internal candidate-pair record with a stable unordered id."""
    id_a = str(fragment_a.get("fragment_id"))
    id_b = str(fragment_b.get("fragment_id"))
    ordered_ids = sorted((id_a, id_b))
    return {
        "pair_id": "%s__%s__%s" % (scene_name, ordered_ids[0], ordered_ids[1]),
        "split": split,
        "scene_name": scene_name,
        "scene_id": fragment_a.get("scene_id"),
        "class_id": 0,
        "class_name": "Person",
        "fragment_a_id": id_a,
        "fragment_b_id": id_b,
        "gt_identity_a": fragment_a.get("gt_identity_id"),
        "gt_identity_b": fragment_b.get("gt_identity_id"),
        "same_identity": int(same_identity),
        "label_source": "ground_truth",
        "pair_type": pair_type,
        "hard_negative": int(pair_type == "hard_negative"),
        "_fragment_a": fragment_a,
        "_fragment_b": fragment_b,
    }


def _bounded_sample(
    values: Sequence[Dict[str, Any]], limit: int, random_generator: random.Random
) -> List[Dict[str, Any]]:
    if len(values) <= limit:
        return list(values)
    return random_generator.sample(list(values), limit)


def _sample_positive_options(
    fragments: Sequence[Dict[str, Any]],
    max_positive: int,
    random_generator: random.Random,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Sample positive combinations without materializing a quadratic list."""
    count = len(fragments)
    if count < 2 or max_positive <= 0:
        return []
    total_combinations = count * (count - 1) // 2
    if total_combinations <= max_positive * 20:
        result = []
        for index_a in range(count):
            for index_b in range(index_a + 1, count):
                result.append((fragments[index_a], fragments[index_b]))
        return result

    selected_indices = set()
    ordered = sorted(fragments, key=lambda row: int(row.get("frame_start") or 0))
    object_indices = {id(fragment): index for index, fragment in enumerate(fragments)}
    for index in range(len(ordered) - 1):
        if not _same_camera(ordered[index], ordered[index + 1]):
            selected_indices.add(
                tuple(
                    sorted(
                        (
                            object_indices[id(ordered[index])],
                            object_indices[id(ordered[index + 1])],
                        )
                    )
                )
            )
            if len(selected_indices) >= max_positive * 4:
                break
    target = min(total_combinations, max_positive * 10)
    attempts = 0
    while len(selected_indices) < target and attempts < target * 20:
        index_a, index_b = random_generator.sample(range(count), 2)
        selected_indices.add(tuple(sorted((index_a, index_b))))
        attempts += 1
    return [(fragments[index_a], fragments[index_b]) for index_a, index_b in selected_indices]


def _same_camera(fragment_a: Dict[str, Any], fragment_b: Dict[str, Any]) -> bool:
    return str(fragment_a.get("camera_id")) == str(fragment_b.get("camera_id"))


def _positive_priority(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> Tuple[int, int]:
    fragment_a, fragment_b = pair
    cross_camera_rank = 0 if not _same_camera(fragment_a, fragment_b) else 1
    gap = abs(int(fragment_b.get("frame_start") or 0) - int(fragment_a.get("frame_end") or 0))
    return cross_camera_rank, gap
