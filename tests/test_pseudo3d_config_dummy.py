from pathlib import Path

from deep_oc_sort_3d.pseudo3d.pseudo3d_config import load_pseudo3d_config, validate_pseudo3d_config


def test_pseudo3d_baseline_config_is_disabled_for_step15b():
    path = Path("deep_oc_sort_3d/configs/pseudo3d_estimator_baseline.yaml")

    config = load_pseudo3d_config(path)
    validation = validate_pseudo3d_config(config)

    assert config["pseudo3d"]["enabled"] is False
    assert config["method"]["primary"] == "bbox_height_depth"
    assert "class_default" in config["method"]["fallback_order"]
    assert validation["status"] == "ok"

