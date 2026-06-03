"""Paper/demo panel builders for MVP visualization outputs."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def build_paper_panel_from_images(
    image_paths: List[Path],
    output_path: Union[str, Path],
    titles: Optional[List[str]] = None,
    caption_labels: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Build a clean image grid for paper/demo figures."""
    if not image_paths:
        raise ValueError("image_paths must contain at least one image")
    images = [np.asarray(Image.open(path).convert("RGB")) for path in image_paths]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = len(images)
    cols = min(3, count)
    rows = int(np.ceil(float(count) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for index, axis in enumerate(axes_arr):
        axis.axis("off")
        if index >= count:
            continue
        axis.imshow(images[index])
        title = ""
        if caption_labels is not None and index < len(caption_labels):
            title += "%s. " % caption_labels[index]
        if titles is not None and index < len(titles):
            title += str(titles[index])
        if title:
            axis.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(str(output), dpi=180)
    plt.close(fig)
    return {"images": count, "rows": rows, "cols": cols}


def build_mvp_demo_panel(
    tracking_image: Path,
    cuboid_image: Path,
    bev_image: Path,
    output_path: Union[str, Path],
) -> Dict[str, int]:
    """Build a 3-panel MVP result figure."""
    return build_paper_panel_from_images(
        [tracking_image, cuboid_image, bev_image],
        output_path,
        titles=[
            "2D tracking and global IDs",
            "3D cuboid diagnostic",
            "Coordinate-space BEV trajectories",
        ],
        caption_labels=["A", "B", "C"],
    )

