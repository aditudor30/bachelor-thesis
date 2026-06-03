import numpy as np

from deep_oc_sort_3d.reid.crop_extraction import (
    clamp_bbox_xyxy,
    crop_image_by_bbox,
    expand_bbox,
    sample_frame_records_for_track,
)


def test_bbox_clamp_expand_crop_and_sampling():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    bbox = (-5.0, 2.0, 12.0, 15.0)
    assert clamp_bbox_xyxy(bbox, 20, 10) == (0, 2, 12, 10)
    expanded = expand_bbox((2.0, 2.0, 8.0, 8.0), 0.5, 20, 10)
    assert expanded == (0, 0, 11, 10)
    crop = crop_image_by_bbox(image, (1.0, 1.0, 5.0, 5.0))
    assert crop is not None
    assert crop.shape == (4, 4, 3)

    records = [{"frame_id": index, "confidence": 1.0 - index * 0.1} for index in range(10)]
    sampled = sample_frame_records_for_track(records, max_crops=3, strategy="uniform")
    assert len(sampled) == 3
    assert sampled[0]["frame_id"] == 0

