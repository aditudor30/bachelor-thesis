"""Deterministic positive/negative balancing for association pairs."""

import random
from typing import Any, Dict, List, Sequence


def balance_pairs(
    pairs: Sequence[Dict[str, Any]],
    negative_to_positive_ratio: float = 3.0,
    random_seed: int = 42,
    require_valid_reid: bool = True,
) -> List[Dict[str, Any]]:
    """Keep positives and sample hard negatives before random negatives."""
    positives = [row for row in pairs if int(row.get("same_identity") or 0) == 1]
    negatives = [row for row in pairs if int(row.get("same_identity") or 0) == 0]
    if require_valid_reid:
        positives = [row for row in positives if int(row.get("embedding_valid_pair") or 0) == 1]
        negatives = [row for row in negatives if int(row.get("embedding_valid_pair") or 0) == 1]
    target_negative_count = int(round(len(positives) * float(negative_to_positive_ratio)))
    hard = [row for row in negatives if int(row.get("hard_negative") or 0) == 1]
    random_negatives = [row for row in negatives if int(row.get("hard_negative") or 0) != 1]
    hard.sort(key=lambda row: float(row.get("reid_similarity") or -1.0), reverse=True)
    generator = random.Random(random_seed)
    generator.shuffle(random_negatives)
    selected = hard[:target_negative_count]
    selected.extend(random_negatives[: max(0, target_negative_count - len(selected))])
    result = list(positives) + selected
    generator.shuffle(result)
    return result
