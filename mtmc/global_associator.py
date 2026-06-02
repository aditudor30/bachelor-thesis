"""High-level global MTMC associator."""

from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_association_cost import merge_global_association_config
from deep_oc_sort_3d.mtmc.global_association_graph import (
    build_candidate_pairs,
    build_global_tracks_from_edges,
    compute_edges_for_pairs,
)
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge, GlobalTrack


class GlobalMTMCAssociator:
    """Prototype global MTMC association without ReID."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = merge_global_association_config(config)

    def associate(
        self,
        candidates: List[MTMCTrackletCandidate],
        show_progress: bool = True,
    ) -> Tuple[List[GlobalTrack], List[GlobalAssociationEdge], Dict[str, int]]:
        """Associate clean MTMC candidates into global tracks."""
        valid_candidates = [candidate for candidate in candidates if candidate.is_candidate]
        pairs = build_candidate_pairs(valid_candidates, self.config, show_progress=show_progress)
        edges = compute_edges_for_pairs(valid_candidates, pairs, self.config, show_progress=show_progress)
        global_tracks, candidate_id_to_global_track_id = build_global_tracks_from_edges(
            valid_candidates,
            edges,
            self.config,
            show_progress=show_progress,
        )
        return global_tracks, edges, candidate_id_to_global_track_id
