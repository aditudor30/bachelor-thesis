import json

from deep_oc_sort_3d.global_tuning.tuning_comparison import compare_global_tuning_runs


def test_tuning_comparison_writes_summary_files(tmp_path):
    output_root = tmp_path / "tuning"
    run_root = output_root / "runs" / "demo" / "global_association" / "official_val" / "Warehouse_020"
    run_root.mkdir(parents=True)
    (run_root / "summary.json").write_text(
        json.dumps(
            {
                "global_tracks": 10,
                "singleton_tracks": 7,
                "multi_camera_tracks": 3,
                "accepted_edges": 4,
                "rejected_edges": 5,
                "total_candidates": 12,
                "per_class_tracks": {"Person": 10},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "eval.json").write_text(
        json.dumps({"global_purity_mean": 0.97, "false_merge_rate": 0.01, "fragmentation_approx": 2}),
        encoding="utf-8",
    )
    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text("global_tuning_run:\n  name: demo\n", encoding="utf-8")
    sweep_cfg = tmp_path / "sweep.yaml"
    sweep_cfg.write_text(
        "\n".join(
            [
                "paths:",
                "  output_root: '%s'" % str(output_root).replace("\\", "/"),
                "  baseline_v2_current_global_root: '%s'" % str(tmp_path / "v2_global").replace("\\", "/"),
                "  baseline_v2_current_final_export_root: '%s'" % str(tmp_path / "v2_final").replace("\\", "/"),
                "  baseline_v2_current_track1_root: '%s'" % str(tmp_path / "v2_track1").replace("\\", "/"),
                "  baseline_v1_global_root: '%s'" % str(tmp_path / "v1_global").replace("\\", "/"),
                "  baseline_v1_final_export_root: '%s'" % str(tmp_path / "v1_final").replace("\\", "/"),
                "  baseline_v1_track1_root: '%s'" % str(tmp_path / "v1_track1").replace("\\", "/"),
                "runs:",
                "  - name: demo",
                "    config: '%s'" % str(run_cfg).replace("\\", "/"),
            ]
        ),
        encoding="utf-8",
    )

    summary = compare_global_tuning_runs(sweep_cfg)

    assert len(summary["runs"]) == 1
    assert (output_root / "comparison" / "global_tuning_summary.json").exists()
    assert (output_root / "diagnostics" / "fragmentation_vs_false_merge.csv").exists()

