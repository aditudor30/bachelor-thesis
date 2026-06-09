from pathlib import Path

from deep_oc_sort_3d.reid_training.osnet_finetune_config import load_osnet_finetune_config, output_root_from_config


def test_osnet_finetune_config_applies_overrides(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "osnet_person_smartspaces_finetune:",
                "  output_root: old_output",
                "paths:",
                "  pretrained_osnet_weights: old.pth",
                "training:",
                "  epochs: 20",
                "  device: cuda",
                "data:",
                "  batch_size: 64",
            ]
        ),
        encoding="utf-8",
    )
    config = load_osnet_finetune_config(
        config_path,
        overrides={
            "output_root": "new_output",
            "epochs": 3,
            "batch_size": 8,
            "device": "cpu",
            "weights": "weights.pth",
        },
    )
    assert output_root_from_config(config) == Path("new_output")
    assert config["training"]["epochs"] == 3
    assert config["training"]["device"] == "cpu"
    assert config["data"]["batch_size"] == 8
    assert config["paths"]["pretrained_osnet_weights"] == "weights.pth"
