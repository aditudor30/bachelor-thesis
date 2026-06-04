"""Lookup stabilized pseudo-3D predictions by frame, track id, or bbox IoU."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilization_io import read_pseudo3d_outputs_csv, read_pseudo3d_outputs_jsonl
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DOutput


class Pseudo3DPredictionLookup:
    """Camera-scoped lookup table for stabilized pseudo-3D predictions."""

    def __init__(
        self,
        predictions_root: Union[str, Path],
        prefer_jsonl: bool = True,
    ) -> None:
        self.predictions_root = Path(predictions_root)
        self.prefer_jsonl = bool(prefer_jsonl)
        self._loaded = {}
        self._local_index = {}
        self._global_index = {}
        self._frame_class_index = {}
        self._summary = {
            "loaded_cameras": 0,
            "loaded_predictions": 0,
            "missing_camera_files": 0,
            "exact_local_hits": 0,
            "exact_global_hits": 0,
            "bbox_iou_hits": 0,
            "misses": 0,
        }

    def load_for_camera(self, subset: str, scene_name: str, camera_id: str) -> List[Pseudo3DOutput]:
        """Load stabilized predictions for one subset/scene/camera."""
        camera_key = (str(subset), str(scene_name), str(camera_id))
        if camera_key in self._loaded:
            return self._loaded[camera_key]
        path = self._camera_path(*camera_key)
        if path is None:
            self._loaded[camera_key] = []
            self._summary["missing_camera_files"] += 1
            return []
        predictions = read_pseudo3d_outputs_jsonl(path) if path.suffix.lower() == ".jsonl" else read_pseudo3d_outputs_csv(path)
        self._loaded[camera_key] = predictions
        self._summary["loaded_cameras"] += 1
        self._summary["loaded_predictions"] += len(predictions)
        self._index_camera(camera_key, predictions)
        return predictions

    def get_by_exact_key(
        self,
        subset: str,
        scene_name: str,
        camera_id: str,
        frame_id: int,
        class_id: int,
        local_track_id: Optional[int] = None,
        global_track_id: Optional[int] = None,
    ) -> Optional[Pseudo3DOutput]:
        """Return a prediction by local-track key first, then global-track key."""
        self.load_for_camera(subset, scene_name, camera_id)
        if local_track_id is not None:
            key = (str(subset), str(scene_name), str(camera_id), int(frame_id), int(class_id), int(local_track_id))
            value = self._local_index.get(key)
            if value is not None:
                self._summary["exact_local_hits"] += 1
                return value
        if global_track_id is not None:
            key = (str(subset), str(scene_name), str(camera_id), int(frame_id), int(class_id), int(global_track_id))
            value = self._global_index.get(key)
            if value is not None:
                self._summary["exact_global_hits"] += 1
                return value
        return None

    def get_by_bbox_iou_fallback(self, record: Any, min_iou: float = 0.8) -> Optional[Pseudo3DOutput]:
        """Return best prediction for the record by frame/class bbox IoU."""
        subset = _record_value(record, "subset", "")
        scene_name = _record_value(record, "scene_name", "")
        camera_id = _record_value(record, "camera_id", "")
        frame_id = _optional_int(_record_value(record, "frame_id", None))
        class_id = _optional_int(_record_value(record, "class_id", None))
        if frame_id is None or class_id is None:
            self._summary["misses"] += 1
            return None
        self.load_for_camera(str(subset), str(scene_name), str(camera_id))
        candidates = self._frame_class_index.get((str(subset), str(scene_name), str(camera_id), int(frame_id), int(class_id)), [])
        record_bbox = _record_bbox(record)
        best = None
        best_iou = 0.0
        for prediction in candidates:
            iou = _bbox_iou(record_bbox, prediction.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best = prediction
        if best is not None and best_iou >= float(min_iou):
            self._summary["bbox_iou_hits"] += 1
            return best
        self._summary["misses"] += 1
        return None

    def summary(self) -> Dict[str, Any]:
        """Return lookup usage and loading summary."""
        return dict(self._summary)

    def _camera_path(self, subset: str, scene_name: str, camera_id: str) -> Optional[Path]:
        base = self.predictions_root / subset / scene_name
        candidates = []
        if self.prefer_jsonl:
            candidates.extend([base / ("%s_pseudo3d_stabilized.jsonl" % camera_id), base / ("%s_pseudo3d_predictions.jsonl" % camera_id)])
            candidates.extend([base / ("%s_pseudo3d_stabilized.csv" % camera_id), base / ("%s_pseudo3d_predictions.csv" % camera_id)])
        else:
            candidates.extend([base / ("%s_pseudo3d_stabilized.csv" % camera_id), base / ("%s_pseudo3d_predictions.csv" % camera_id)])
            candidates.extend([base / ("%s_pseudo3d_stabilized.jsonl" % camera_id), base / ("%s_pseudo3d_predictions.jsonl" % camera_id)])
        for path in candidates:
            if path.exists():
                return path
        return None

    def _index_camera(self, camera_key: Tuple[str, str, str], predictions: List[Pseudo3DOutput]) -> None:
        subset, scene_name, camera_id = camera_key
        for prediction in predictions:
            frame_id = int(prediction.frame_id)
            class_id = int(prediction.class_id)
            if prediction.local_track_id is not None:
                key = (subset, scene_name, camera_id, frame_id, class_id, int(prediction.local_track_id))
                self._local_index[key] = prediction
            if prediction.global_track_id is not None:
                key = (subset, scene_name, camera_id, frame_id, class_id, int(prediction.global_track_id))
                self._global_index[key] = prediction
            frame_key = (subset, scene_name, camera_id, frame_id, class_id)
            self._frame_class_index.setdefault(frame_key, []).append(prediction)


def _record_value(record: Any, key: str, default: Any) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _record_bbox(record: Any) -> Tuple[float, float, float, float]:
    bbox = _record_value(record, "bbox_xyxy", None)
    if isinstance(bbox, str):
        bbox = _parse_bbox_string(bbox)
    if bbox is not None:
        try:
            values = list(bbox)
            if len(values) >= 4:
                return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
        except (TypeError, ValueError):
            pass
    return (
        float(_record_value(record, "x1", 0.0) or 0.0),
        float(_record_value(record, "y1", 0.0) or 0.0),
        float(_record_value(record, "x2", 0.0) or 0.0),
        float(_record_value(record, "y2", 0.0) or 0.0),
    )


def _parse_bbox_string(value: str) -> List[float]:
    text = value.strip().strip("[]()")
    if not text:
        return []
    return [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(float(ax1), float(bx1))
    iy1 = max(float(ay1), float(by1))
    ix2 = min(float(ax2), float(bx2))
    iy2 = min(float(ay2), float(by2))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, float(ax2) - float(ax1)) * max(0.0, float(ay2) - float(ay1))
    area_b = max(0.0, float(bx2) - float(bx1)) * max(0.0, float(by2) - float(by1))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0

