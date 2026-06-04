"""Load and query final class-wise 3D priors."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from deep_oc_sort_3d.audit3d.audit3d_io import read_json_if_exists
from deep_oc_sort_3d.pseudo3d.pseudo3d_types import Pseudo3DPriors


class Pseudo3DPriorTable:
    """Lookup table for class-wise pseudo-3D dimension priors."""

    def __init__(self, priors_by_class_id: Dict[int, Pseudo3DPriors]) -> None:
        self.priors_by_class_id = priors_by_class_id

    def get(self, class_id: int) -> Optional[Pseudo3DPriors]:
        """Return prior for class_id when available."""
        return self.priors_by_class_id.get(int(class_id))

    def require(self, class_id: int) -> Pseudo3DPriors:
        """Return prior or raise KeyError."""
        prior = self.get(class_id)
        if prior is None:
            raise KeyError("Missing pseudo-3D prior for class_id=%s" % class_id)
        return prior


def load_pseudo3d_priors(path: Union[str, Path]) -> Pseudo3DPriorTable:
    """Load final class priors from Step 15B JSON."""
    data = read_json_if_exists(path)
    classes = data.get("classes", [])
    priors = {}
    for item in classes:
        if not isinstance(item, dict):
            continue
        prior = prior_from_dict(item)
        if prior is not None:
            priors[int(prior.class_id)] = prior
    return Pseudo3DPriorTable(priors)


def prior_from_dict(item: Dict[str, Any]) -> Optional[Pseudo3DPriors]:
    """Create Pseudo3DPriors from a final prior dictionary."""
    try:
        class_id = int(item["class_id"])
        class_name = str(item.get("class_name", ""))
        width = float(item["robust_width"])
        length = float(item["robust_length"])
        height = float(item["robust_height"])
    except (KeyError, TypeError, ValueError):
        return None
    return Pseudo3DPriors(
        class_id=class_id,
        class_name=class_name,
        width=width,
        length=length,
        height=height,
        confidence_level=str(item.get("confidence_level", "unknown")),
        source=str(item.get("selected_prior_source", "step15b_final_priors")),
    )

