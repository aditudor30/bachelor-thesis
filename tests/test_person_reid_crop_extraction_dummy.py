import numpy as np

from deep_oc_sort_3d.person_reid.crop_extraction import crop_image_xyxy, expand_and_clip_bbox


def test_person_reid_crop_extraction_clips_bbox():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    bbox = (-5.0, 2.0, 12.0, 12.0)
    clipped = expand_and_clip_bbox(bbox, 20, 10, 0.0)
    crop = crop_image_xyxy(image, bbox)
    assert clipped == (0, 2, 12, 10)
    assert crop is not None
    assert crop.shape == (8, 12, 3)

