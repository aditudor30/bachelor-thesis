"""Visualize global tracks as coordinate-space BEV trajectories."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from deep_oc_sort_3d.visualization3d.bev_visualization import plot_bev_tracks
from deep_oc_sort_3d.visualization3d.visualization_io import (
    filter_records_by_class,
    filter_records_by_global_track_id,
    load_global_frame_records_csv,
)


def main() -> None:
    args = parse_args()
    if args.generic_csv is not None:
        records = load_global_frame_records_csv(args.generic_csv)
    elif args.track1 is not None:
        records = load_track1_rows(args.track1)
    else:
        raise ValueError("Provide --generic-csv or --track1")
    if args.class_name is not None:
        records = filter_records_by_class(records, args.class_name)
    if args.global_track_id is not None:
        records = filter_records_by_global_track_id(records, args.global_track_id)
    map_image = load_map_image(args.map_path)
    summary = plot_bev_tracks(records, args.output, map_image=map_image, max_tracks=args.max_tracks, class_name=None)
    print("records: %d" % len(records))
    print("tracks_plotted: %d" % int(summary.get("tracks_plotted", 0)))
    print("output: %s" % args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generic-csv", type=Path, default=None)
    parser.add_argument("--track1", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map", dest="map_path", type=Path, default=None)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--global-track-id", type=int, default=None)
    parser.add_argument("--max-tracks", type=int, default=100)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def load_track1_rows(path: Path) -> List[Dict[str, Any]]:
    """Load confirmed Track 1 text rows into generic-like dictionaries."""
    rows = []
    columns = [
        "scene_id",
        "class_id",
        "global_track_id",
        "frame_id",
        "center_x",
        "center_y",
        "center_z",
        "width_3d",
        "length_3d",
        "height_3d",
        "yaw",
    ]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != len(columns):
                continue
            row = dict(zip(columns, parts))
            row["scene_name"] = "Warehouse_%03d" % int(float(row["scene_id"]))
            row["class_name"] = ""
            rows.append(row)
    return rows


def load_map_image(path: Any):
    """Load an optional map image for BEV diagnostics."""
    if path is None:
        return None
    if not Path(path).exists():
        return None
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


if __name__ == "__main__":
    main()
