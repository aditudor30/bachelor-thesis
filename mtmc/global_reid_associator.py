"""Global MTMC associator with optional ReID appearance cost."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_association_cost import merge_global_association_config
from deep_oc_sort_3d.mtmc.global_association_graph import (
    build_candidate_pairs,
    build_global_tracks_from_edges,
    compute_edges_for_pairs,
)
from deep_oc_sort_3d.mtmc.global_reid_cost import attach_reid_cost_to_edge
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge, GlobalTrack
from deep_oc_sort_3d.mtmc.reid_embedding_lookup import ReIDEmbeddingLookup
from deep_oc_sort_3d.mtmc.transition_cost import transition_pair_to_global_edge
from deep_oc_sort_3d.mtmc.transition_diagnostics import build_transition_candidate_pairs


def default_reid_association_config() -> Dict[str, Any]:
    """Return default ReID association options."""
    return {
        "use_reid": True,
        "reid_root": "output/reid_embeddings/yolo11m_medium_conf001_colorhist",
        "embedding_backend": "color_histogram",
        "appearance_weight": 0.10,
        "geometry_only_fallback": True,
        "require_embeddings_for_edge": False,
        "prefer_candidate_embeddings": True,
        "normalize_embeddings": True,
    }


def split_global_reid_config(config: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a mixed config into global-MTMC and ReID sections."""
    config = {} if config is None else dict(config)
    if "global_mtmc" in config or "reid" in config:
        global_section = config.get("global_mtmc", {})
        reid_section = config.get("reid", {})
    else:
        global_section = config
        reid_section = {}
    if not isinstance(global_section, dict):
        global_section = {}
    if not isinstance(reid_section, dict):
        reid_section = {}
    global_config = merge_global_association_config(global_section)
    reid_config = default_reid_association_config()
    reid_config.update(reid_section)
    return global_config, reid_config


class GlobalMTMCReIDAssociator:
    """Global association variant that augments geometry edges with ReID cost."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        reid_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.global_config, self.reid_config = split_global_reid_config(config)
        if reid_root is not None:
            self.reid_config["reid_root"] = str(reid_root)
        self.embedding_lookup = self._build_lookup()

    def associate(
        self,
        candidates: List[MTMCTrackletCandidate],
        show_progress: bool = True,
    ) -> Tuple[List[GlobalTrack], List[GlobalAssociationEdge], Dict[str, int]]:
        """Associate candidates into global tracks with optional appearance cost."""
        valid_candidates = [candidate for candidate in candidates if candidate.is_candidate]
        self._preload_embeddings(valid_candidates, show_progress)
        overlap_edges = self._build_overlap_edges(valid_candidates, show_progress)
        transition_edges = self._build_transition_edges(valid_candidates, show_progress)
        all_edges = overlap_edges + transition_edges
        candidate_by_id = {candidate.candidate_id: candidate for candidate in valid_candidates}
        reid_edges = []
        for edge in _progress_iter(all_edges, show_progress, "ReID edge costs", "edge"):
            candidate_a = candidate_by_id.get(edge.candidate_id_a)
            candidate_b = candidate_by_id.get(edge.candidate_id_b)
            if candidate_a is None or candidate_b is None:
                edge.used_reid = False
                edge.reid_missing_reason = "candidate_missing"
                edge.geometry_cost = float(edge.cost)
                edge.total_cost = float(edge.cost)
                reid_edges.append(edge)
                continue
            reid_edges.append(
                attach_reid_cost_to_edge(edge, candidate_a, candidate_b, self.embedding_lookup, self.reid_config)
            )
        global_tracks, candidate_id_to_global_track_id = build_global_tracks_from_edges(
            valid_candidates,
            reid_edges,
            self.global_config,
            show_progress=show_progress,
        )
        _mark_reid_notes(global_tracks, self.reid_config)
        return global_tracks, reid_edges, candidate_id_to_global_track_id

    def _build_lookup(self) -> Optional[ReIDEmbeddingLookup]:
        if not bool(self.reid_config.get("use_reid", True)):
            return None
        return ReIDEmbeddingLookup(
            self.reid_config.get("reid_root", ""),
            prefer_candidate_embeddings=bool(self.reid_config.get("prefer_candidate_embeddings", True)),
            normalize=bool(self.reid_config.get("normalize_embeddings", True)),
        )

    def _preload_embeddings(self, candidates: List[MTMCTrackletCandidate], show_progress: bool) -> None:
        if self.embedding_lookup is None:
            return
        scene_keys = sorted(set([(candidate.subset, candidate.scene_name) for candidate in candidates]))
        for subset, scene_name in _progress_iter(scene_keys, show_progress, "ReID embedding scenes", "scene"):
            self.embedding_lookup.load_for_subset_scene(str(subset), str(scene_name))

    def _build_overlap_edges(
        self,
        candidates: List[MTMCTrackletCandidate],
        show_progress: bool,
    ) -> List[GlobalAssociationEdge]:
        if not bool(self.global_config.get("enable_overlap_association", True)):
            return []
        overlap_config = dict(self.global_config)
        overlap_config["enable_transition_association"] = False
        pairs = build_candidate_pairs(candidates, overlap_config, show_progress=show_progress)
        return compute_edges_for_pairs(candidates, pairs, overlap_config, show_progress=show_progress)

    def _build_transition_edges(
        self,
        candidates: List[MTMCTrackletCandidate],
        show_progress: bool,
    ) -> List[GlobalAssociationEdge]:
        if not bool(self.global_config.get("enable_transition_association", False)):
            return []
        transition_pairs = build_transition_candidate_pairs(candidates, self.global_config, show_progress=show_progress)
        edges = []
        for pair in _progress_iter(transition_pairs, show_progress, "transition edges", "pair"):
            edge = transition_pair_to_global_edge(
                pair,
                float(pair.transition_cost) if pair.transition_cost is not None else 1e9,
                pair.accepted_by_threshold,
                pair.reject_reason,
            )
            edges.append(edge)
        return edges


def _mark_reid_notes(global_tracks: List[GlobalTrack], reid_config: Dict[str, Any]) -> None:
    label = "reid_disabled" if not bool(reid_config.get("use_reid", True)) else "geometry_plus_reid"
    weight = float(reid_config.get("appearance_weight", 0.0))
    for track in global_tracks:
        track.notes = "%s_w%.3f" % (label, weight)


def _progress_iter(values: Iterable[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: Iterable[Any], desc: str) -> Iterable[Any]:
    total = len(values) if hasattr(values, "__len__") else None
    for index, value in enumerate(values):
        if total is None:
            print("%s: item %d" % (desc, index + 1))
        elif index == 0 or (index + 1) % 1000 == 0 or index + 1 == total:
            print("%s: item %d/%d" % (desc, index + 1, total))
        yield value
