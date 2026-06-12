"""Small I/O helpers shared by the learned-association dataset tools."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np


NOT_AVAILABLE = "not_available"


def read_json(path: Path, default: Optional[Any] = None) -> Any:
    """Read JSON, returning ``default`` when the file is absent."""
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    """Write formatted JSON and create parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read all rows from a CSV file."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """Write dictionaries to CSV using a stable union of row keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames) if fieldnames is not None else collect_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in names})


def collect_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Return keys in first-seen order."""
    names = []  # type: List[str]
    seen = set()
    for row in rows:
        for key in row.keys():
            if key.startswith("_") or key in seen:
                continue
            seen.add(key)
            names.append(key)
    return names


def csv_value(value: Any) -> Any:
    """Convert composite values to compact JSON for CSV storage."""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(to_serializable(value), separators=(",", ":"))
    if value is None:
        return ""
    return value


def to_serializable(value: Any) -> Any:
    """Convert NumPy values and non-finite floats into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Parse a finite float without raising."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Parse an integer without raising."""
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_list(value: Any) -> List[Any]:
    """Parse list-like JSON values found in CSV fields."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return list(parsed) if isinstance(parsed, (list, tuple)) else []
        except (TypeError, ValueError):
            return []
    return []


def progress_iter(
    values: Iterable[Any],
    description: str,
    enabled: bool = True,
    total: Optional[int] = None,
) -> Iterator[Any]:
    """Wrap an iterable in tqdm, with a periodic-print fallback."""
    if not enabled:
        for value in values:
            yield value
        return
    try:
        from tqdm import tqdm

        for value in tqdm(values, desc=description, total=total):
            yield value
        return
    except ImportError:
        pass
    for index, value in enumerate(values, start=1):
        if index == 1 or index % 1000 == 0:
            suffix = "" if total is None else "/%d" % total
            print("%s: %d%s" % (description, index, suffix))
        yield value


def prepare_output_tree(root: Path, overwrite: bool = False) -> Dict[str, Path]:
    """Create the isolated Step 20A output tree."""
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(
            "Output root is not empty: %s. Use --overwrite or --skip-existing." % root
        )
    directories = {}
    for name in ("metadata", "pairs", "features", "diagnostics", "figures", "reports"):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        directories[name] = path
    return directories
