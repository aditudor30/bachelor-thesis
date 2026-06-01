"""Ultralytics YOLO inference wrapper for SmartSpaces videos."""

from pathlib import Path
from typing import List, Optional, Union

import cv2

from deep_oc_sort_3d.detection2d.yolo_label_utils import xyxy_to_xywh
from deep_oc_sort_3d.detection2d.yolo_types import Detection2D


def load_yolo_model(model_path: Union[str, Path]):
    """Load an Ultralytics YOLO model or raise a clear dependency error."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics is not installed. Install it with: pip install ultralytics")
    return YOLO(str(model_path))


def run_yolo_on_video(
    model,
    video_path: Path,
    scene_id: int,
    scene_name: str,
    split: str,
    camera_id: str,
    conf_threshold: float = 0.25,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    imgsz: int = 1280,
    device: Optional[str] = None,
) -> List[Detection2D]:
    """Run YOLO frame-by-frame on one video and return common detections."""
    detections = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError("Could not open video: %s" % video_path)

    frame_id = 0
    stride = max(int(frame_stride), 1)
    try:
        while True:
            if max_frames is not None and frame_id >= int(max_frames):
                break
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            if frame_id % stride != 0:
                frame_id += 1
                continue

            predict_kwargs = {
                "conf": float(conf_threshold),
                "imgsz": int(imgsz),
                "verbose": False,
            }
            if device is not None:
                predict_kwargs["device"] = str(device)
            results = model(frame_bgr, **predict_kwargs)
            if results:
                detections.extend(
                    _detections_from_result(
                        result=results[0],
                        scene_id=scene_id,
                        scene_name=scene_name,
                        split=split,
                        camera_id=camera_id,
                        frame_id=frame_id,
                    )
                )
            frame_id += 1
    finally:
        capture.release()
    return detections


def _detections_from_result(
    result,
    scene_id: int,
    scene_name: str,
    split: str,
    camera_id: str,
    frame_id: int,
) -> List[Detection2D]:
    detections = []
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    for index in range(xyxy.shape[0]):
        class_id = int(cls[index])
        class_name = str(names.get(class_id, class_id))
        bbox_xyxy = (
            float(xyxy[index][0]),
            float(xyxy[index][1]),
            float(xyxy[index][2]),
            float(xyxy[index][3]),
        )
        detections.append(
            Detection2D(
                scene_id=scene_id,
                scene_name=scene_name,
                split=split,
                camera_id=camera_id,
                frame_id=frame_id,
                class_id=class_id,
                class_name=class_name,
                confidence=float(conf[index]),
                bbox_xyxy=bbox_xyxy,
                bbox_xywh=xyxy_to_xywh(bbox_xyxy),
                source="yolo",
            )
        )
    return detections
