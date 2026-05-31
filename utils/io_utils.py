"""I/O helpers used by dataset inspection scripts."""

import json
from pathlib import Path
from typing import Any, Optional


def load_json_if_exists(path: Optional[Path]) -> Optional[Any]:
    """Load JSON if the file exists, otherwise return None."""
    if path is None or not path.exists() or not path.is_file():
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def path_exists(path: Optional[Path]) -> bool:
    """Return True when a path is present on disk."""
    return path is not None and path.exists()

