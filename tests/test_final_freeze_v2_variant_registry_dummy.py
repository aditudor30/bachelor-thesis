from deep_oc_sort_3d.final_freeze_v2.final_variant_registry import final_variant_specs


def test_final_freeze_v2_registry_contains_required_variants():
    specs = final_variant_specs({"paths": {}})
    names = [spec["variant_name"] for spec in specs]
    assert "v1_geometry_only" in names
    assert "v2_pseudo3d_fullcam_current" in names
    assert "v2_export_compact" in names
    assert "osnet_finetuned_threshold_080" in names
    assert "osnet_finetuned_combined_safe_080" in names
    combined = [spec for spec in specs if spec["variant_name"] == "osnet_finetuned_combined_safe_080"][0]
    assert combined["role"] == "experimental fine-tuned ReID variant"

