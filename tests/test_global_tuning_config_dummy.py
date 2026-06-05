from pathlib import Path

from deep_oc_sort_3d.global_tuning.tuning_config import build_run_specs_from_sweep


def test_global_tuning_config_builds_run_specs(tmp_path):
    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text(
        "\n".join(
            [
                "global_tuning_run:",
                "  name: demo",
                "  paths:",
                "    input_motion_clean_root: input_candidates",
                "    local_tracks_root: local_tracks",
                "  global_mtmc:",
                "    max_temporal_gap: 123",
            ]
        ),
        encoding="utf-8",
    )
    sweep_cfg = tmp_path / "sweep.yaml"
    sweep_cfg.write_text(
        "\n".join(
            [
                "paths:",
                "  output_root: '%s'" % str(tmp_path / "out").replace("\\", "/"),
                "runs:",
                "  - name: demo",
                "    config: '%s'" % str(run_cfg).replace("\\", "/"),
            ]
        ),
        encoding="utf-8",
    )

    specs = build_run_specs_from_sweep(sweep_cfg)

    assert len(specs) == 1
    assert specs[0].name == "demo"
    assert specs[0].global_config["max_temporal_gap"] == 123
    assert specs[0].global_association_root == tmp_path / "out" / "runs" / "demo" / "global_association"

