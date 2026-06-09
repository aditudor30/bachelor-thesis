"""Balanced identity sampler for Person ReID training."""

import random
from typing import Dict, Iterator, List


class BalancedIdentitySampler(object):
    """Sample batches with P identities and K images per identity."""

    def __init__(self, labels: List[int], identities_per_batch: int = 16, images_per_identity: int = 4, seed: int = 42) -> None:
        self.labels = [int(label) for label in labels]
        self.identities_per_batch = int(identities_per_batch)
        self.images_per_identity = int(images_per_identity)
        self.seed = int(seed)
        self.index_by_label: Dict[int, List[int]] = {}
        for index, label in enumerate(self.labels):
            self.index_by_label.setdefault(label, []).append(index)
        self.labels_unique = sorted(self.index_by_label.keys())

    def __iter__(self) -> Iterator[int]:
        """Yield sampled indices."""
        rng = random.Random(self.seed)
        labels = list(self.labels_unique)
        rng.shuffle(labels)
        output: List[int] = []
        for start in range(0, len(labels), self.identities_per_batch):
            batch_labels = labels[start : start + self.identities_per_batch]
            if len(batch_labels) < self.identities_per_batch:
                break
            for label in batch_labels:
                indices = list(self.index_by_label[label])
                if len(indices) >= self.images_per_identity:
                    output.extend(rng.sample(indices, self.images_per_identity))
                else:
                    output.extend([rng.choice(indices) for _ in range(self.images_per_identity)])
        return iter(output)

    def __len__(self) -> int:
        """Return number of yielded samples per epoch."""
        full_groups = int(len(self.labels_unique) / max(1, self.identities_per_batch))
        return full_groups * self.identities_per_batch * self.images_per_identity

