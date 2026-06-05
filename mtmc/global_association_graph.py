"""Graph utilities for global MTMC association."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.mtmc.global_association_cost import (
    apply_global_association_per_class_overrides,
    compute_global_association_cost,
    merge_global_association_config,
    temporal_gap,
    temporal_overlap,
)
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge, GlobalTrack


class UnionFind:
    """Small union-find implementation for association components."""

    def __init__(self, size: int) -> None:
        self.parent = [index for index in range(int(size))]
        self.rank = [0 for _index in range(int(size))]

    def find(self, index: int) -> int:
        """Return component root."""
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, left: int, right: int) -> bool:
        """Merge two components if different."""
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1
        return True


def build_candidate_pairs(
    candidates: List[MTMCTrackletCandidate],
    config: Dict[str, Any],
    show_progress: bool = True,
) -> List[Tuple[int, int]]:
    """Build blocked candidate pairs without all-pairs across the full dataset."""
    cfg = merge_global_association_config(config)
    groups = _group_candidates(candidates, cfg)
    pairs = []
    group_items = sorted(groups.items(), key=lambda item: item[0])
    for key, indices in _progress_iter(group_items, show_progress, "global pair groups", "group"):
        _unused_key = key
        group_indices = _limit_group(indices, cfg)
        group_pairs = _pairs_for_group(candidates, group_indices, cfg)
        pairs.extend(group_pairs)
    return pairs


def compute_edges_for_pairs(
    candidates: List[MTMCTrackletCandidate],
    pairs: List[Tuple[int, int]],
    config: Dict[str, Any],
    show_progress: bool = True,
) -> List[GlobalAssociationEdge]:
    """Compute association edges for candidate pairs."""
    cfg = merge_global_association_config(config)
    edges = []
    iterator = _progress_iter(pairs, show_progress, "global edge costs", "pair")
    for index_a, index_b in iterator:
        edge = compute_global_association_cost(candidates[index_a], candidates[index_b], cfg)
        edges.append(edge)
    return edges


def build_global_tracks_from_edges(
    candidates: List[MTMCTrackletCandidate],
    edges: List[GlobalAssociationEdge],
    config: Dict[str, Any],
    show_progress: bool = True,
) -> Tuple[List[GlobalTrack], Dict[str, int]]:
    """Build global tracks from accepted edges using constrained union-find."""
    cfg = merge_global_association_config(config)
    index_by_id = {candidate.candidate_id: index for index, candidate in enumerate(candidates)}
    union_find = UnionFind(len(candidates))
    component_members = {index: [index] for index in range(len(candidates))}
    accepted_edges = [edge for edge in edges if edge.accepted]
    accepted_edges = sorted(accepted_edges, key=lambda item: float(item.cost))

    for edge in _progress_iter(accepted_edges, show_progress, "global union edges", "edge"):
        if edge.candidate_id_a not in index_by_id or edge.candidate_id_b not in index_by_id:
            continue
        index_a = index_by_id[edge.candidate_id_a]
        index_b = index_by_id[edge.candidate_id_b]
        root_a = union_find.find(index_a)
        root_b = union_find.find(index_b)
        if root_a == root_b:
            continue
        members_a = component_members[root_a]
        members_b = component_members[root_b]
        if not _can_merge_components(candidates, members_a, members_b, cfg):
            continue
        merged = union_find.union(root_a, root_b)
        if merged:
            new_root = union_find.find(root_a)
            old_root = root_b if new_root == root_a else root_a
            merged_members = component_members.pop(root_a, members_a) + component_members.pop(root_b, members_b)
            component_members[new_root] = sorted(set(merged_members))
            if old_root in component_members:
                component_members.pop(old_root)

    groups = _components_from_union_find(union_find, len(candidates))
    tracks = []
    candidate_id_to_global_track_id = {}
    global_track_id = 0
    for members in _progress_iter(groups, show_progress, "global track components", "component"):
        if len(members) < int(cfg["min_candidates_per_global_track"]):
            continue
        if len(members) == 1 and not bool(cfg["include_singletons"]):
            continue
        track = _build_global_track(global_track_id, [candidates[index] for index in members])
        tracks.append(track)
        for candidate_id in track.candidate_ids:
            candidate_id_to_global_track_id[candidate_id] = int(global_track_id)
        global_track_id += 1
    return tracks, candidate_id_to_global_track_id


def _group_candidates(
    candidates: List[MTMCTrackletCandidate],
    cfg: Dict[str, Any],
) -> Dict[Tuple[str, str, int], List[int]]:
    groups = {}
    for index, candidate in enumerate(candidates):
        if bool(cfg["class_must_match"]):
            key = (str(candidate.subset), str(candidate.scene_name), int(candidate.class_id))
        else:
            key = (str(candidate.subset), str(candidate.scene_name), -1)
        groups.setdefault(key, []).append(index)
    return groups


def _limit_group(indices: List[int], cfg: Dict[str, Any]) -> List[int]:
    max_candidates = cfg.get("max_candidates_per_group")
    if max_candidates is None:
        return indices
    return indices[: int(max_candidates)]


def _pairs_for_group(
    candidates: List[MTMCTrackletCandidate],
    indices: List[int],
    cfg: Dict[str, Any],
) -> List[Tuple[int, int]]:
    sorted_indices = sorted(indices, key=lambda index: (candidates[index].start_frame, candidates[index].end_frame))
    pairs = []
    transition_enabled = bool(cfg["enable_transition_association"])
    allow_same_camera = bool(cfg["allow_same_camera_links"])
    for outer_pos, index_a in enumerate(sorted_indices):
        candidate_a = candidates[index_a]
        max_gap_a = _candidate_max_temporal_gap(candidate_a, cfg)
        for index_b in sorted_indices[outer_pos + 1 :]:
            candidate_b = candidates[index_b]
            max_gap = max(max_gap_a, _candidate_max_temporal_gap(candidate_b, cfg))
            latest_start = int(candidate_a.end_frame) + (max_gap if transition_enabled else 0)
            if int(candidate_b.start_frame) > latest_start:
                break
            if not allow_same_camera and candidate_a.camera_id == candidate_b.camera_id:
                continue
            if temporal_overlap(candidate_a, candidate_b) > 0:
                pairs.append((index_a, index_b))
                continue
            if transition_enabled and temporal_gap(candidate_a, candidate_b) <= max_gap:
                pairs.append((index_a, index_b))
    return pairs


def _candidate_max_temporal_gap(candidate: MTMCTrackletCandidate, cfg: Dict[str, Any]) -> int:
    effective = apply_global_association_per_class_overrides(cfg, candidate.class_id, candidate.class_name)
    return int(effective["max_temporal_gap"])


def _can_merge_components(
    candidates: List[MTMCTrackletCandidate],
    members_a: List[int],
    members_b: List[int],
    cfg: Dict[str, Any],
) -> bool:
    merged = [candidates[index] for index in members_a + members_b]
    scenes = set([candidate.scene_name for candidate in merged])
    if len(scenes) > 1:
        return False
    if bool(cfg["class_must_match"]):
        class_ids = set([int(candidate.class_id) for candidate in merged])
        if len(class_ids) > 1:
            return False
    if not bool(cfg["allow_same_camera_links"]):
        camera_ids = [candidate.camera_id for candidate in merged]
        if len(camera_ids) != len(set(camera_ids)):
            return False
    return True


def _components_from_union_find(union_find: UnionFind, size: int) -> List[List[int]]:
    groups = {}
    for index in range(size):
        root = union_find.find(index)
        groups.setdefault(root, []).append(index)
    return sorted(groups.values(), key=lambda members: (min(members), len(members)))


def _build_global_track(global_track_id: int, candidates: List[MTMCTrackletCandidate]) -> GlobalTrack:
    ordered = sorted(candidates, key=lambda item: (item.start_frame, item.camera_id, item.local_track_id))
    trajectories = []
    centers = []
    gt_id_counts = {}
    for candidate in ordered:
        for item in candidate.trajectory_3d_sampled:
            if len(item) < 4:
                continue
            trajectories.append((int(item[0]), float(item[1]), float(item[2]), float(item[3])))
            centers.append([float(item[1]), float(item[2]), float(item[3])])
        if candidate.center_3d_mean is not None:
            centers.append([float(value) for value in np.asarray(candidate.center_3d_mean, dtype=float).reshape(-1)[:3]])
        if candidate.majority_gt_object_id is not None:
            key = str(int(candidate.majority_gt_object_id))
            gt_id_counts[key] = gt_id_counts.get(key, 0) + 1

    trajectories = sorted(trajectories, key=lambda item: (item[0], item[1], item[2], item[3]))
    center_3d_mean = None
    if centers:
        center_3d_mean = np.mean(np.asarray(centers, dtype=float), axis=0)
    majority_gt_object_id = None
    gt_purity = None
    if gt_id_counts:
        majority_key = max(gt_id_counts.keys(), key=lambda key: gt_id_counts[key])
        majority_gt_object_id = int(majority_key)
        gt_purity = float(gt_id_counts[majority_key]) / float(sum(gt_id_counts.values()))
    start_frame = min([candidate.start_frame for candidate in ordered])
    end_frame = max([candidate.end_frame for candidate in ordered])
    mean_confidence = float(np.mean(np.asarray([candidate.mean_confidence for candidate in ordered], dtype=float)))
    max_confidence = float(max([candidate.max_confidence for candidate in ordered]))
    camera_ids = sorted(set([candidate.camera_id for candidate in ordered]))
    return GlobalTrack(
        global_track_id=int(global_track_id),
        scene_name=str(ordered[0].scene_name),
        subset=str(ordered[0].subset),
        split=str(ordered[0].split),
        class_id=int(ordered[0].class_id),
        class_name=str(ordered[0].class_name),
        candidate_ids=[candidate.candidate_id for candidate in ordered],
        camera_ids=camera_ids,
        local_track_ids=[int(candidate.local_track_id) for candidate in ordered],
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        duration=int(end_frame - start_frame + 1),
        num_candidates=len(ordered),
        num_cameras=len(camera_ids),
        mean_confidence=mean_confidence,
        max_confidence=max_confidence,
        trajectory_3d_sampled=trajectories,
        center_3d_mean=center_3d_mean,
        majority_gt_object_id=majority_gt_object_id,
        gt_purity=gt_purity,
        num_gt_ids=len(gt_id_counts),
        gt_id_counts=gt_id_counts,
        notes="no_reid_geometry_only",
    )


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
