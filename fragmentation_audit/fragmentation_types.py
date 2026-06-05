"""Small dataclasses used by the fragmentation audit."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FragmentationThresholds:
    """Configurable thresholds for short/singleton track diagnostics."""

    singleton_length: int = 1
    short_track_length: int = 3
    very_short_track_length: int = 5
    long_track_length: int = 30
    high_fragmentation_ratio: float = 0.5
    high_singleton_ratio: float = 0.5
    high_rows_per_track_p95: float = 100.0
    motion_invalid_warning_ratio: float = 0.10


@dataclass
class AuditRunPaths:
    """Root paths for one run being audited."""

    name: str
    pipeline_root: str
    local_tracks_root: str
    tracklets_root: str
    candidates_root: str
    motion_clean_root: str
    global_root: str
    final_export_root: str
    track1_root: str


@dataclass
class StageAuditResult:
    """Normalized result for one stage audit."""

    stage: str
    run_name: str
    root: str
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FragmentationAuditConfig:
    """Top-level fragmentation audit config."""

    output_root: str
    thresholds: FragmentationThresholds
    v1: AuditRunPaths
    v2: AuditRunPaths
    subsets: List[str] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    progress: bool = True

