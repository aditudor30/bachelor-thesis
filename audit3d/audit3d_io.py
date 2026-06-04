"""Shared I/O, statistics, and progress helpers for 3D audits."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


Number = Optional[float]


def optional_float(value: Any) -> Optional[float]:
    """Parse a float while preserving missing values as None."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: Any) -> Optional[int]:
    """Parse an int through float syntax, returning None when missing."""
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_finite_number(value: Any) -> bool:
    """Return True when value is a finite numeric value."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finite_float(value: Any) -> Optional[float]:
    """Return a float only when it is finite."""
    parsed = optional_float(value)
    if parsed is None or not math.isfinite(parsed):
        return None
    return parsed


def numeric_field_stats(rows: List[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Any]:
    """Compute robust numeric summaries for the selected fields."""
    return {field: numeric_stats([row.get(field) for row in rows]) for field in fields}


def numeric_stats(values: Sequence[Any]) -> Dict[str, Any]:
    """Compute count, missingness, finite stats, and robust percentiles."""
    parsed = [optional_float(value) for value in values]
    missing = sum(1 for value in values if value in (None, ""))
    nan_count = sum(1 for value in parsed if value is not None and math.isnan(value))
    inf_count = sum(1 for value in parsed if value is not None and math.isinf(value))
    finite_values = [float(value) for value in parsed if value is not None and math.isfinite(value)]
    finite_values.sort()
    result = {
        "count": len(values),
        "valid_count": len(finite_values),
        "missing": missing,
        "nan": nan_count,
        "inf": inf_count,
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "p01": None,
        "p05": None,
        "p95": None,
        "p99": None,
        "unique_count": 0,
        "constant_value": None,
        "constant_ratio": None,
    }
    if not finite_values:
        return result

    total = float(len(finite_values))
    mean = sum(finite_values) / total
    variance = sum((value - mean) ** 2.0 for value in finite_values) / total
    unique_values = sorted(set(finite_values))
    most_common_value, most_common_count = most_common(finite_values)
    result.update(
        {
            "min": finite_values[0],
            "max": finite_values[-1],
            "mean": mean,
            "median": percentile(finite_values, 0.50),
            "std": math.sqrt(variance),
            "p01": percentile(finite_values, 0.01),
            "p05": percentile(finite_values, 0.05),
            "p95": percentile(finite_values, 0.95),
            "p99": percentile(finite_values, 0.99),
            "unique_count": len(unique_values),
            "constant_value": most_common_value if len(unique_values) == 1 else None,
            "constant_ratio": float(most_common_count) / total,
        }
    )
    return result


def percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    """Compute an interpolated percentile from sorted finite values."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, float(q))) * float(len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - float(lower)
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def most_common(values: Sequence[Any]) -> Tuple[Any, int]:
    """Return the most common value and its count."""
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None, 0
    best_value = None
    best_count = -1
    for value, count in counts.items():
        if count > best_count:
            best_value = value
            best_count = count
    return best_value, best_count


def flatten_stats(prefix: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one numeric stats dictionary using a field prefix."""
    flattened = {}
    for key, value in stats.items():
        flattened["%s_%s" % (prefix, key)] = value
    return flattened


def group_rows(rows: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    """Group row dictionaries by a tuple of key values."""
    grouped = {}
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def read_csv_dicts(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a CSV file as dictionaries."""
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_dicts(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read JSONL dictionaries, skipping empty lines."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def iter_data_files(root_or_file: Union[str, Path], suffixes: Optional[Sequence[str]] = None) -> List[Path]:
    """Return data files from a file or recursively from a directory."""
    if root_or_file in (None, ""):
        return []
    path = Path(root_or_file)
    suffix_set = set([suffix.lower() for suffix in suffixes]) if suffixes is not None else None
    if path.is_file():
        if suffix_set is None or path.suffix.lower() in suffix_set:
            return [path]
        return []
    if not path.exists():
        return []
    files = []
    for child in path.rglob("*"):
        if child.is_file() and (suffix_set is None or child.suffix.lower() in suffix_set):
            files.append(child)
    return sorted(files)


def progress_iter(values: Iterable[Any], show_progress: bool, desc: str, unit: str = "item") -> Iterable[Any]:
    """Wrap an iterable with tqdm when available, otherwise print periodically."""
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def print_progress_iter(values: Iterable[Any], desc: str) -> Iterable[Any]:
    """Fallback progress printer for environments without tqdm."""
    sequence = list(values)
    total = len(sequence)
    for index, value in enumerate(sequence):
        if index == 0 or (index + 1) % 10000 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value


def write_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write JSON with stable ordering."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(text: str, path: Union[str, Path]) -> None:
    """Write a Markdown file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def write_csv(rows: List[Dict[str, Any]], path: Union[str, Path], fieldnames: Optional[List[str]] = None) -> None:
    """Write rows to CSV, inferring fields when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = infer_fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def infer_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Infer stable CSV fieldnames from row dictionaries."""
    names = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names


def read_json_if_exists(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON dictionary when it exists, otherwise return an empty dict."""
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def ensure_clean_output_dir(path: Union[str, Path], overwrite: bool) -> None:
    """Create an output directory and guard against accidental overwrite."""
    output_path = Path(path)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError("Output directory exists and is not empty: %s" % output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def scene_id_from_name(scene_name: Any) -> Optional[int]:
    """Extract scene id from names like Warehouse_023."""
    if scene_name in (None, ""):
        return None
    text = str(scene_name)
    digits = ""
    for char in reversed(text):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    if not digits:
        return None
    return optional_int(digits)


def finite_xyz(row: Dict[str, Any], x_key: str, y_key: str, z_key: str) -> Optional[Tuple[float, float, float]]:
    """Return finite xyz coordinates from a row."""
    x = finite_float(row.get(x_key))
    y = finite_float(row.get(y_key))
    z = finite_float(row.get(z_key))
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def finite_dimensions(row: Dict[str, Any], w_key: str, l_key: str, h_key: str) -> Optional[Tuple[float, float, float]]:
    """Return finite positive 3D dimensions from a row."""
    width = finite_float(row.get(w_key))
    length = finite_float(row.get(l_key))
    height = finite_float(row.get(h_key))
    if width is None or length is None or height is None:
        return None
    if width <= 0.0 or length <= 0.0 or height <= 0.0:
        return None
    return (width, length, height)
