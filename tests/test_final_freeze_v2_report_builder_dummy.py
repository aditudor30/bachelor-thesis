from deep_oc_sort_3d.final_freeze_v2.final_report_builder import build_project_freeze_report, build_reid_summary


def test_final_freeze_v2_report_builder_mentions_final_verdicts():
    bundle = {
        "variants": [{"variant_name": "v1_geometry_only", "role": "submission-safe baseline"}],
        "reid_training": [],
        "reid_association": [],
        "reid_visual": {},
    }
    report = "\n".join(build_project_freeze_report(bundle))
    reid = "\n".join(build_reid_summary(bundle))
    assert "submission-safe baseline" in report
    assert "combined_safe_080" in report
    assert "extensie experimentala ReID-enhanced" in reid
