from deep_oc_sort_3d.final_freeze_v2.final_metric_loader import collect_variant_rows


def test_final_freeze_v2_metric_loader_handles_missing_files():
    rows = collect_variant_rows({"paths": {}}, show_progress=False)
    assert len(rows) >= 5
    v2 = [row for row in rows if row["variant_name"] == "v2_pseudo3d_fullcam_current"][0]
    assert v2["pseudo3d_used_rate"] == 0.9807563276013348
    reid = [row for row in rows if row["variant_name"] == "osnet_finetuned_combined_safe_080"][0]
    assert reid["person_fragmentation_delta"] == -52

