from deep_oc_sort_3d.learned_association_application.scorer_association_metrics import metric_deltas


def test_metric_deltas_preserve_direction():
    run = {"track1_rows": 90, "person_fragmentation": 40, "non_person_rows": 20}
    baseline = {"track1_rows": 100, "person_fragmentation": 50, "non_person_rows": 20}
    delta = metric_deltas(run, baseline)
    assert delta["track1_rows_delta"] == -10
    assert delta["person_fragmentation_delta"] == -10
    assert delta["non_person_rows_delta"] == 0
