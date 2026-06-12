"""I/O helpers for Step 20B Person association scorers."""

import pickle
from pathlib import Path
from typing import Any, Dict

import yaml

from deep_oc_sort_3d.learned_association.pair_dataset_io import read_csv_rows, write_csv_rows, write_json


SCORER_OUTPUT_DIRS = (
    "configs",
    "data",
    "models",
    "evaluation",
    "figures",
    "reports",
)


def prepare_scorer_output(root: Path, overwrite: bool = False) -> Dict[str, Path]:
    """Create the isolated Step 20B output tree."""
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(
            "Scorer output is not empty: %s. Use --overwrite to replace artifacts." % root
        )
    result = {}
    for name in SCORER_OUTPUT_DIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        result[name] = path
    return result


def save_pickle(path: Path, value: Any) -> None:
    """Write a Python object with pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    """Read a pickled Python object."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_resolved_config(config: Dict[str, Any], output_root: Path) -> None:
    """Write the effective YAML configuration."""
    path = output_root / "configs" / "resolved_config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


__all__ = [
    "load_pickle",
    "prepare_scorer_output",
    "read_csv_rows",
    "save_pickle",
    "save_resolved_config",
    "write_csv_rows",
    "write_json",
]
