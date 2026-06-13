"""Step 22B wrapper around the strict official Track1 validator."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.official_023_027.official_track1_validator import validate_official_track1
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import write_json


def validate_variant_track1(path: Path, output_path: Path, config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Validate and persist one V3.1 official Track1 candidate."""
    report = validate_official_track1(path, config, progress=progress)
    write_json(output_path, report)
    return report

