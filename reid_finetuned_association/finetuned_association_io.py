"""I/O helpers for fine-tuned Person ReID association experiments."""

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from deep_oc_sort_3d.person_association.person_association_io import (
    count_by,
    frame_record_csv_files,
    generic_csv_files,
    infer_fieldnames,
    load_yaml,
    mean,
    optional_list,
    percentile,
    progress_iter,
    read_csv_rows,
    read_json,
    safe_float,
    safe_int,
    write_csv_rows,
    write_json,
    write_yaml,
)


OUTPUT_SUBDIRS = [
    "embeddings",
    "embeddings/crops",
    "embeddings/fragment_embeddings",
    "diagnostics",
    "sweep_runs",
    "comparison",
    "figures",
    "configs",
]


def load_finetuned_association_config(config_path: Path) -> Dict[str, Any]:
    """Load the Step 18C config file."""
    config = load_yaml(Path(config_path))
    return config if isinstance(config, dict) else {}


def output_root_from_config(config: Dict[str, Any]) -> Path:
    """Return Step 18C output root."""
    section = config.get("person_reid_finetuned_association", {})
    return Path(str(section.get("output_root", "output/person_reid_finetuned_association/baseline_v2_pseudo3d_fullcam")))


def prepare_output_root(config: Dict[str, Any], overwrite: bool = False) -> Path:
    """Create the separate Step 18C output tree."""
    root = output_root_from_config(config)
    if overwrite and root.exists():
        shutil.rmtree(str(root))
    for name in OUTPUT_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def save_resolved_config(config: Dict[str, Any], config_path: Path, output_root: Path) -> None:
    """Store a copy of the resolved config used for the run."""
    write_yaml(config, output_root / "configs" / "person_reid_finetuned_association_resolved.yaml")
    if Path(config_path).exists():
        target = output_root / "configs" / Path(config_path).name
        target.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")


def threshold_to_name(value: float) -> str:
    """Convert a threshold into a stable run name suffix."""
    return "threshold_%03d" % int(round(float(value) * 100.0))


def row_scene_id(row: Dict[str, Any]) -> str:
    """Return scene id from a row, falling back to scene_name suffix."""
    value = row.get("scene_id")
    if value not in (None, ""):
        return str(value)
    scene = str(row.get("scene_name", ""))
    if "_" in scene:
        return scene.split("_")[-1]
    return ""


def bool_text(value: Any) -> str:
    """Serialize booleans as simple text."""
    return "1" if bool(value) else "0"


def ensure_parent(path: Path) -> None:
    """Create parent directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_text(lines: Iterable[str], path: Path) -> None:
    """Write text lines."""
    ensure_parent(path)
    Path(path).write_text("\n".join([str(line) for line in lines]) + "\n", encoding="utf-8")


__all__ = [
    "Any",
    "Dict",
    "Iterable",
    "List",
    "Optional",
    "Path",
    "bool_text",
    "count_by",
    "ensure_parent",
    "frame_record_csv_files",
    "generic_csv_files",
    "infer_fieldnames",
    "load_finetuned_association_config",
    "load_yaml",
    "mean",
    "optional_list",
    "output_root_from_config",
    "percentile",
    "prepare_output_root",
    "progress_iter",
    "read_csv_rows",
    "read_json",
    "row_scene_id",
    "safe_float",
    "safe_int",
    "save_resolved_config",
    "threshold_to_name",
    "write_csv_rows",
    "write_json",
    "write_text",
    "write_yaml",
]
