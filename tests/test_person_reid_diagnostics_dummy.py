from deep_oc_sort_3d.person_reid.reid_diagnostics import verdict_for_reid


def test_person_reid_diagnostics_verdict_ready_when_separated():
    summary = {"separation_margin": 0.2}
    sweep = [{"precision": 0.8, "recall": 0.3, "fpr": 0.05}]
    verdict = verdict_for_reid(summary, sweep, [object()])
    assert verdict == "reid_ready_for_person_association"

