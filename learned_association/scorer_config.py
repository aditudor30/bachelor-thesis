"""Configuration helpers for Step 20B scorer training."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_scorer_config(path: Path) -> Dict[str, Any]:
    """Load and validate a scorer YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for section in ("person_association_scorer", "paths", "features", "models", "mlp", "evaluation"):
        if section not in config:
            raise ValueError("Missing scorer config section: %s" % section)
    config["_config_path"] = str(path)
    return config


def apply_scorer_overrides(
    config: Dict[str, Any],
    pair_dataset_root: Optional[str] = None,
    output_root: Optional[str] = None,
    device: Optional[str] = None,
    epochs: Optional[int] = None,
    progress: Optional[bool] = None,
) -> Dict[str, Any]:
    """Apply supported CLI overrides in-place."""
    if pair_dataset_root is not None:
        root = Path(pair_dataset_root)
        paths = config.setdefault("paths", {})
        paths["pair_dataset_root"] = str(root)
        paths["train_pairs_csv"] = str(root / "pairs" / "train_pairs_balanced.csv")
        paths["val_pairs_csv"] = str(root / "pairs" / "val_pairs_balanced.csv")
        paths["raw_train_pairs_csv"] = str(root / "pairs" / "train_pairs.csv")
        paths["raw_val_pairs_csv"] = str(root / "pairs" / "val_pairs.csv")
    if output_root is not None:
        config.setdefault("person_association_scorer", {})["output_root"] = output_root
    if device is not None:
        config.setdefault("mlp", {})["device"] = device
    if epochs is not None:
        config.setdefault("mlp", {})["epochs"] = epochs
    if progress is not None:
        config.setdefault("person_association_scorer", {})["progress"] = progress
    return config


def scorer_output_root(config: Dict[str, Any]) -> Path:
    """Return the configured scorer output root."""
    return Path(
        str(
            config.get("person_association_scorer", {}).get(
                "output_root", "output/learned_association/person_scorer_v1"
            )
        )
    )


def scorer_progress_enabled(config: Dict[str, Any]) -> bool:
    """Return whether progress output is enabled."""
    return bool(config.get("person_association_scorer", {}).get("progress", True))


def random_seed(config: Dict[str, Any]) -> int:
    """Return the global random seed."""
    return int(config.get("person_association_scorer", {}).get("random_seed", 42))
