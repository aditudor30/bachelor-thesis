from deep_oc_sort_3d.reid_ablation.ablation_comparison import compare_reid_ablation_variants


def _row(name, source_type, track1_rows, person_frag, reid_merges, export_rows):
    return {
        "variant_name": name,
        "source_type": source_type,
        "track1_valid": True,
        "track1_errors": 0,
        "track1_rows": track1_rows,
        "person_rows": 1000,
        "non_person_rows": 500,
        "global_purity": 0.97,
        "false_merge_rate": 0.05,
        "fragmentation_approx": 100,
        "person_fragmentation": person_frag,
        "num_reid_merges": reid_merges,
        "num_geometry_merges": 0,
        "num_export_dropped_rows": export_rows,
    }


def test_reid_ablation_comparison_labels_export_compact_source():
    rows = [
        _row("v2", "v2_current", 1000, 80, 0, 0),
        _row("compact", "export_compact", 900, 75, 0, 100),
        _row("reid_plus", "reid_plus_compact", 900, 75, 2, 100),
    ]
    comparison = compare_reid_ablation_variants(rows)
    by_name = {row["variant_name"]: row for row in comparison["variants"]}
    assert by_name["compact"]["improvement_source"] == "export_compact"
    assert by_name["reid_plus"]["improvement_source"] == "export_compact_only"

