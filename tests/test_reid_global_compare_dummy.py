"""Dummy tests for comparing ReID global association runs."""

import json

from deep_oc_sort_3d.scripts.compare_reid_global_runs import compare_reid_global_runs


class _Args:
    def __init__(self, baseline, runs, names, output):
        self.baseline = baseline
        self.runs = runs
        self.names = names
        self.output = output


def _write_summary(path, multi_camera, purity, fragmentation):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "global_tracks": 10,
                "multi_camera_tracks": multi_camera,
                "singleton_tracks": 10 - multi_camera,
                "accepted_edges": 5,
                "accepted_edge_temporal_relations": {"overlap": 3, "a_before_b": 2},
                "per_class_tracks": {"Person": 10},
                "diagnostic_gt_metrics": {
                    "global_purity_mean": purity,
                    "false_merge_rate": 0.1,
                    "fragmentation_approx": fragmentation,
                },
            }
        ),
        encoding="utf-8",
    )


def test_compare_reid_global_runs_writes_csv(tmp_path):
    baseline = tmp_path / "baseline"
    run = tmp_path / "run"
    _write_summary(baseline / "official_val" / "Warehouse_020" / "summary.json", 2, 0.90, 5)
    _write_summary(run / "official_val" / "Warehouse_020" / "summary.json", 3, 0.92, 4)

    output = tmp_path / "comparison.csv"
    rows = compare_reid_global_runs(_Args(baseline, [run], ["w010"], output))

    assert output.exists()
    assert len(rows) == 2
    assert rows[1]["delta_multi_camera_tracks"] == 1.0
    assert rows[1]["purity_change"] == "better"
    assert rows[1]["fragmentation_change"] == "better"
