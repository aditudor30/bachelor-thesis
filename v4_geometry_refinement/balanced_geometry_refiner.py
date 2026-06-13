"""Conservative composition of the four V4 geometry refinements."""

from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row
from deep_oc_sort_3d.v4_geometry_refinement.dimension_consistency import stabilize_track_dimensions
from deep_oc_sort_3d.v4_geometry_refinement.outlier_repair import repair_position_outliers
from deep_oc_sort_3d.v4_geometry_refinement.track_smoothing import smooth_track_positions
from deep_oc_sort_3d.v4_geometry_refinement.yaw_refinement import refine_track_yaw


def refine_geometry_balanced(
    rows: Sequence[OfficialTrack1Row],
    config: Dict[str, Any],
    progress: bool = True,
) -> Tuple[List[OfficialTrack1Row], List[Dict[str, Any]]]:
    """Repair isolated outliers, then smooth positions, dimensions and yaw."""
    current, repair_changes = repair_position_outliers(rows, config, progress=progress)
    current, smooth_changes = smooth_track_positions(current, config, progress=progress)
    current, dimension_changes = stabilize_track_dimensions(current, config, progress=progress)
    current, yaw_changes = refine_track_yaw(current, config, progress=progress)
    changes = []
    for stage, values in [
        ("outlier_repair", repair_changes),
        ("smoothing", smooth_changes),
        ("dimension_consistency", dimension_changes),
        ("yaw_refinement", yaw_changes),
    ]:
        for value in values:
            row = dict(value)
            row["stage"] = stage
            changes.append(row)
    return current, changes

