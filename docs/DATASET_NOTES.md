# Dataset Notes

## Dataset Structure

```text
MTMC_Tracking_2026/
  train/
    Warehouse_000/ ... Warehouse_019/
      videos/
      depth_maps/
      ground_truth.json
      calibration.json
      map.png
  val/
    Warehouse_020/ ... Warehouse_022/
      videos/
      depth_maps/
      ground_truth.json
      calibration.json
      map.png
  test/
    Warehouse_023/ ... Warehouse_025/
      videos/
      calibration.json
      map.png
```

## Split Differences

Train and val contain:

- `videos/`
- `depth_maps/`
- `ground_truth.json`
- `calibration.json`
- `map.png`

Test contains:

- `videos/`
- `calibration.json`
- `map.png`

Test does not contain depth maps or ground truth. This is normal and must not be treated as an error.

## Depth Notes

- Depth maps are stored as H5 files per camera.
- File names are camera-based, for example `Camera_0000.h5`.
- H5 files are large and must be read lazily, frame by frame.
- Depth appears to be in millimeters.
- Convert depth to meters with `/1000.0`.
- Default depth sampling method is `center_median`.
- `bottom_center` remains useful as a diagnostic alternative.

## Frame Indexing

- Ground-truth frame keys start at `"0"`.
- OpenCV frame indexing is 0-based.
- `frame_idx=0` corresponds to `ground_truth["0"]`.

## Ground Truth Keys

The real ground-truth JSON uses keys with spaces:

- `object type`
- `object id`
- `3d location`
- `3d bounding box scale`
- `3d bounding box rotation`
- `2d bounding box visible`

Parsers should also tolerate underscore-style variants where possible.

## Classes

| Class ID | Class Name |
| --- | --- |
| 0 | Person |
| 1 | Forklift |
| 2 | PalletTruck |
| 3 | Transporter |
| 4 | FourierGR1T2 |
| 5 | AgilityDigit |
| 6 | NovaCarter |

## Validation Notes

- Train has all seven classes.
- Official val does not contain all classes.
- Official val is missing FourierGR1T2 and AgilityDigit.
- Transporter and NovaCarter are very rare in official val.
- Internal holdout is needed for diagnostic evaluation of robotic classes.

