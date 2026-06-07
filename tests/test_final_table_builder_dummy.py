from deep_oc_sort_3d.final_freeze.table_builder import format_metric, rows_to_latex_table, rows_to_markdown_table


def test_final_table_builder_formats_markdown_and_latex():
    rows = [{"variant_name": "V1_geometry", "track1_valid": True, "track1_rows": 42, "global_purity": 0.95}]
    columns = ["variant_name", "track1_valid", "track1_rows", "global_purity"]

    markdown = rows_to_markdown_table(rows, columns)
    latex = rows_to_latex_table(rows, columns, caption="Dummy caption", label="tab:dummy")

    assert "| variant_name |" in markdown
    assert "yes" in markdown
    assert "\\begin{table}" in latex
    assert "V1\\_geometry" in latex
    assert "tab:dummy" in latex


def test_final_table_builder_format_metric_handles_missing_and_float():
    assert format_metric(None) == "not_available"
    assert format_metric(0.1234567) == "0.123457"

