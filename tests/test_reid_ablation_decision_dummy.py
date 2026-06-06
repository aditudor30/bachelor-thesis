from deep_oc_sort_3d.reid_ablation.ablation_decision import make_final_decision


def test_reid_ablation_decision_keeps_v1_v2_and_marks_reid_no_gain():
    comparison = {
        "variants": [
            {
                "variant_name": "v1_geometry_only",
                "source_type": "v1",
                "track1_valid": True,
                "is_safe": True,
                "improvement_source": "legacy_geometry_submission",
            },
            {
                "variant_name": "v2_pseudo3d_fullcam_current",
                "source_type": "v2_current",
                "track1_valid": True,
                "is_safe": True,
                "improvement_source": "current_3d_mvp",
            },
            {
                "variant_name": "v2_export_compact",
                "source_type": "export_compact",
                "track1_valid": True,
                "is_safe": True,
                "improvement_source": "export_compact",
            },
            {
                "variant_name": "reid_medium",
                "source_type": "reid_only",
                "track1_valid": True,
                "is_safe": True,
                "real_upgrade": False,
                "num_reid_merges": 2,
                "pairs_with_reid": 100,
                "improvement_source": "minor_reid_activity_no_measurable_gain",
            },
        ]
    }
    decision = make_final_decision(comparison)
    assert "keep_v1_for_submission" in decision["verdicts"]
    assert "keep_v2_current_as_3d_mvp" in decision["verdicts"]
    assert "reid_infrastructure_valid_but_no_gain" in decision["verdicts"]
