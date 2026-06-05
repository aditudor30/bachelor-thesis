"""Run specification dataclasses for global tuning."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GlobalTuningRunSpec:
    """Resolved configuration for one tuning run."""

    name: str
    output_root: Path
    config_path: Optional[Path] = None
    dataset_root: str = "/path/to/MTMC_Tracking_2026"
    input_motion_clean_root: Path = Path("output/mtmc_candidates_motion_clean/baseline_v2_pseudo3d_fullcam")
    local_tracks_root: Path = Path("output/local_tracks/baseline_v2_pseudo3d_fullcam")
    schema_yaml: Path = Path("deep_oc_sort_3d/configs/track1_schema_confirmed.yaml")
    subsets: Optional[List[str]] = field(default_factory=lambda: ["official_val", "internal_holdout", "test"])
    scenes: Optional[List[str]] = None
    class_names: Optional[List[str]] = None
    max_candidates_per_scene: Optional[int] = None
    global_config: Dict[str, Any] = field(default_factory=dict)
    final_export_options: Dict[str, Any] = field(default_factory=dict)
    track1_export_options: Dict[str, Any] = field(default_factory=dict)
    compact_export: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @property
    def global_association_root(self) -> Path:
        """Return per-run global association output root."""
        return self.output_root / "global_association"

    @property
    def final_export_root(self) -> Path:
        """Return per-run final export output root."""
        return self.output_root / "final_export"

    @property
    def track1_root(self) -> Path:
        """Return per-run Track 1 output root."""
        return self.output_root / "track1_submission"

    @property
    def validation_root(self) -> Path:
        """Return per-run validation output root."""
        return self.output_root / "validation"

    @property
    def summaries_root(self) -> Path:
        """Return per-run summaries root."""
        return self.output_root / "summaries"

    @property
    def runtime_config_root(self) -> Path:
        """Return per-run materialized config root."""
        return self.output_root / "configs"


def list_value(value: Any) -> Optional[List[str]]:
    """Normalize optional list-ish values."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]

