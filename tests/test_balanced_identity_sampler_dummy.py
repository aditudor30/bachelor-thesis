from deep_oc_sort_3d.reid_training.balanced_identity_sampler import BalancedIdentitySampler


def test_balanced_identity_sampler_yields_pk_batches():
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    sampler = BalancedIdentitySampler(labels, identities_per_batch=2, images_per_identity=2, seed=7)
    indices = list(iter(sampler))
    assert len(indices) == 4
    sampled_labels = [labels[index] for index in indices]
    assert len(set(sampled_labels)) == 2
    for label in set(sampled_labels):
        assert sampled_labels.count(label) == 2
