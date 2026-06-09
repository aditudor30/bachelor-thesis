from deep_oc_sort_3d.final_freeze_v2.final_table_builder import format_metric, rows_to_latex_table, rows_to_markdown_table


def test_final_freeze_v2_table_builder_formats_missing_and_latex():
    rows = [{"variant_name": "v1", "track1_valid": True, "metric": None}]
    md = rows_to_markdown_table(rows, ["variant_name", "track1_valid", "metric"])
    tex = rows_to_latex_table(rows, ["variant_name", "track1_valid"], caption="Cap", label="tab:test")
    assert "not_available" in md
    assert "\\begin{table}" in tex
    assert format_metric(True) == "yes"

