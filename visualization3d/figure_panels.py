"""Small figure panel helpers for demo images."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_frame_grid_panel(
    image_paths: List[Path],
    output_path: Union[str, Path],
    titles: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Create a grid panel from saved image paths."""
    frames = []
    for path in image_paths:
        frames.append(np.asarray(Image.open(path).convert("RGB")))
    return make_tracking_demo_panel(frames, output_path, titles=titles)


def make_tracking_demo_panel(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    titles: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Save a simple grid panel from RGB frames."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = len(frames)
    if count == 0:
        raise ValueError("At least one frame is required")
    cols = int(np.ceil(np.sqrt(float(count))))
    rows = int(np.ceil(float(count) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for index, axis in enumerate(axes_arr):
        axis.axis("off")
        if index >= count:
            continue
        axis.imshow(frames[index])
        if titles is not None and index < len(titles):
            axis.set_title(str(titles[index]))
    fig.tight_layout()
    fig.savefig(str(output), dpi=150)
    plt.close(fig)
    return {"images": count, "rows": rows, "cols": cols}
