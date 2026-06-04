"""I/O helpers for Step 15B 3D priors."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deep_oc_sort_3d.audit3d.audit3d_io import (
    progress_iter,
    read_csv_dicts,
    read_json_if_exists,
    write_csv,
    write_json,
    write_markdown,
)


def read_prior_csv(path: Union[str, Path], show_progress: bool = True) -> List[Dict[str, Any]]:
    """Read class prior rows from CSV."""
    rows = read_csv_dicts(path)
    return [row for row in progress_iter(rows, show_progress, "read class prior rows", "row")]


def read_prior_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read class prior JSON when available."""
    return read_json_if_exists(path)


def write_prior_outputs(
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    output_json: Union[str, Path],
    output_csv: Union[str, Path],
    report_text: Optional[str] = None,
    report_path: Optional[Union[str, Path]] = None,
) -> None:
    """Write final prior JSON, CSV, and optional Markdown report."""
    write_json(summary, output_json)
    write_csv(rows, output_csv)
    if report_text is not None and report_path is not None:
        write_markdown(report_text, report_path)

