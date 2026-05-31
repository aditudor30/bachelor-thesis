"""Lazy HDF5 depth-map I/O helpers.

Depth files can be many GB. This module only reads HDF5 metadata or a single
requested frame slice.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import h5py
import numpy as np


DEPTH_EXTENSIONS = (".h5", ".hdf5")
PREFERRED_DEPTH_DATASETS = ("depth", "depth_maps", "data")


def list_depth_files(depth_dir: Path) -> List[Path]:
    """List HDF5 depth files under a depth directory."""
    directory = Path(depth_dir)
    if not directory.exists() or not directory.is_dir():
        return []
    files = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in DEPTH_EXTENSIONS:
            files.append(path)
    return sorted(files)


def infer_camera_id_from_depth_path(path: Path) -> str:
    """Infer camera id from a depth filename stem."""
    return Path(path).stem


def inspect_h5_depth_file(path: Path) -> Dict[str, Any]:
    """Inspect HDF5 depth metadata without loading full arrays."""
    depth_path = Path(path)
    report = {
        "path": str(depth_path),
        "exists": depth_path.exists(),
        "dataset_candidates": [],
        "selected_dataset": None,
        "selected_layout": None,
        "shape": None,
        "dtype": None,
        "num_frames": None,
        "error": None,
    }
    if not depth_path.exists() or not depth_path.is_file():
        report["error"] = "File does not exist"
        return report

    try:
        with h5py.File(str(depth_path), "r") as h5_file:
            candidates = _collect_dataset_candidates(h5_file)
            frame_datasets = _find_indexed_frame_datasets(h5_file)
            selected = find_depth_dataset_name(h5_file)
            report["dataset_candidates"] = candidates
            if frame_datasets and _selected_is_single_frame_dataset(h5_file, selected):
                first_name = frame_datasets[0][1]
                dataset = h5_file[first_name]
                report["selected_dataset"] = first_name
                report["selected_layout"] = "per_frame_datasets"
                report["shape"] = tuple(int(dim) for dim in dataset.shape)
                report["dtype"] = str(dataset.dtype)
                report["num_frames"] = len(frame_datasets)
            elif selected is not None:
                dataset = h5_file[selected]
                report["selected_dataset"] = selected
                report["selected_layout"] = "array_dataset"
                report["shape"] = tuple(int(dim) for dim in dataset.shape)
                report["dtype"] = str(dataset.dtype)
                report["num_frames"] = _estimate_num_frames(dataset.shape)
    except Exception as exc:
        report["error"] = str(exc)

    return report


def find_depth_dataset_name(h5_file: Any) -> Optional[str]:
    """Find the most likely depth dataset in an open HDF5 file or group."""
    for name in PREFERRED_DEPTH_DATASETS:
        if name in h5_file and _is_numeric_dataset(h5_file[name]):
            return name

    found = []

    def visitor(name: str, obj: Any) -> None:
        if not found and _is_numeric_dataset(obj):
            found.append(name)

    h5_file.visititems(visitor)
    if found:
        return found[0]
    return None


def read_depth_frame_h5(
    depth_path: Path,
    frame_idx: int,
    dataset_name: Optional[str] = None,
) -> np.ndarray:
    """Read one 0-based depth frame from an HDF5 file."""
    if frame_idx < 0:
        raise IndexError("frame_idx must be non-negative")

    with h5py.File(str(depth_path), "r") as h5_file:
        selected = dataset_name
        exact_frame_dataset = False
        if selected is None:
            selected = find_depth_dataset_name(h5_file)
            if _selected_is_single_frame_dataset(h5_file, selected):
                frame_dataset = _find_dataset_name_for_frame(h5_file, frame_idx)
                if frame_dataset is not None:
                    selected = frame_dataset
                    exact_frame_dataset = True
        if selected is None:
            selected = _find_dataset_name_for_frame(h5_file, frame_idx)
            exact_frame_dataset = selected is not None
        if selected is None:
            raise KeyError("No numeric depth dataset found in %s" % depth_path)

        dataset = h5_file[selected]
        if isinstance(dataset, h5py.Group):
            selected_in_group = _find_dataset_name_for_frame(dataset, frame_idx)
            if selected_in_group is None:
                raise KeyError("No frame dataset %d found in group %s" % (frame_idx, selected))
            dataset = dataset[selected_in_group]
            exact_frame_dataset = True

        return _read_frame_from_dataset(dataset, frame_idx, exact_frame_dataset)


def safe_read_depth_frame_h5(
    depth_path: Path,
    frame_idx: int,
    dataset_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Read one depth frame, returning None instead of raising on failure."""
    try:
        return read_depth_frame_h5(depth_path, frame_idx, dataset_name)
    except Exception:
        return None


def _collect_dataset_candidates(h5_file: Any) -> List[Dict[str, Any]]:
    candidates = []

    def visitor(name: str, obj: Any) -> None:
        if _is_numeric_dataset(obj):
            candidates.append(
                {
                    "name": name,
                    "shape": tuple(int(dim) for dim in obj.shape),
                    "dtype": str(obj.dtype),
                    "num_frames": _estimate_num_frames(obj.shape),
                }
            )

    h5_file.visititems(visitor)
    return candidates


def _selected_is_single_frame_dataset(h5_file: Any, selected: Optional[str]) -> bool:
    if selected is None or selected not in h5_file:
        return False
    obj = h5_file[selected]
    return isinstance(obj, h5py.Dataset) and len(obj.shape) == 2


def _find_dataset_name_for_frame(h5_file: Any, frame_idx: int) -> Optional[str]:
    for index, name in _find_indexed_frame_datasets(h5_file):
        if index == frame_idx:
            return name
    return None


def _find_indexed_frame_datasets(h5_file: Any) -> List[Tuple[int, str]]:
    indexed = []

    def visitor(name: str, obj: Any) -> None:
        if not _is_numeric_dataset(obj):
            return
        if len(obj.shape) != 2:
            return
        frame_index = _extract_frame_index(name)
        if frame_index is None:
            return
        indexed.append((frame_index, name))

    h5_file.visititems(visitor)
    return sorted(indexed, key=lambda item: item[0])


def _extract_frame_index(name: str) -> Optional[int]:
    base_name = name.split("/")[-1]
    matches = re.findall(r"\d+", base_name)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _is_numeric_dataset(obj: Any) -> bool:
    if not isinstance(obj, h5py.Dataset):
        return False
    try:
        return bool(np.issubdtype(obj.dtype, np.number))
    except TypeError:
        return False


def _estimate_num_frames(shape: Any) -> Optional[int]:
    if shape is None:
        return None
    if len(shape) == 2:
        return 1
    if len(shape) in (3, 4):
        return int(shape[0])
    return None


def _read_frame_from_dataset(dataset: Any, frame_idx: int, exact_frame_dataset: bool) -> np.ndarray:
    shape = dataset.shape
    if len(shape) == 2:
        if frame_idx != 0 and not exact_frame_dataset:
            raise IndexError("Single-frame depth dataset only supports frame_idx=0")
        return np.asarray(dataset[...])

    if len(shape) == 3:
        if frame_idx >= shape[0]:
            raise IndexError("frame_idx %d out of range for %d frames" % (frame_idx, shape[0]))
        return np.asarray(dataset[frame_idx])

    if len(shape) == 4 and shape[3] == 1:
        if frame_idx >= shape[0]:
            raise IndexError("frame_idx %d out of range for %d frames" % (frame_idx, shape[0]))
        return np.asarray(dataset[frame_idx, :, :, 0])

    raise ValueError("Unsupported depth dataset shape: %s" % (shape,))
