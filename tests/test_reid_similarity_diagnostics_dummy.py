import numpy as np

from deep_oc_sort_3d.reid_training.reid_similarity_diagnostics import compute_similarity_diagnostics, finetuning_verdict, metric_deltas


def test_reid_similarity_diagnostics_separates_same_and_different():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"identity_id": "a"},
        {"identity_id": "a"},
        {"identity_id": "b"},
        {"identity_id": "b"},
    ]
    result = compute_similarity_diagnostics(embeddings, metadata, max_pairs=6, thresholds=[0.8], seed=1)
    assert result["status"] == "ok"
    assert result["same_gt"]["mean"] > result["different_gt"]["mean"]
    assert result["threshold_0.80"]["same_recall"] == 1.0


def test_reid_metric_deltas_and_verdict_dummy():
    pretrained = {"top1_accuracy": 0.5, "top5_accuracy": 0.7, "top10_accuracy": 0.8, "mAP": 0.4}
    finetuned = {"top1_accuracy": 0.54, "top5_accuracy": 0.72, "top10_accuracy": 0.82, "mAP": 0.45}
    pretrained_sim = {"separation_margin": 0.05, "threshold_0.80": {"different_high_sim_risk": 0.1, "same_recall": 0.5}}
    finetuned_sim = {"separation_margin": 0.09, "threshold_0.80": {"different_high_sim_risk": 0.08, "same_recall": 0.6}}
    deltas = metric_deltas(pretrained, finetuned, pretrained_sim, finetuned_sim)
    verdict = finetuning_verdict(pretrained, finetuned, deltas)
    assert deltas["top1_delta"] > 0
    assert verdict["verdict"] == "finetuned_reid_ready_for_association"
