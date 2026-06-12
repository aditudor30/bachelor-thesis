"""Track1 validation wrapper for the ByteTrack-local baseline."""

from pathlib import Path
from typing import Any, Dict, Optional

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_validation import validate_fullcam_track1


def validate_bytetrack_track1(
    track1_path: Path,
    output_root: Optional[Path] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Validate official Track1 schema and expected test scenes/classes."""
    return validate_fullcam_track1(
        track1_path=Path(track1_path),
        output_root=output_root,
        expected_scene_ids=[23, 24, 25],
        valid_class_ids=[0, 1, 2, 3, 4, 5, 6],
        show_progress=progress,
    )
