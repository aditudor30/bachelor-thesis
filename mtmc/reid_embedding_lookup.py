"""Lookup utilities for MTMC candidate ReID embeddings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from deep_oc_sort_3d.mtmc.candidate_types import MTMCTrackletCandidate
from deep_oc_sort_3d.reid.reid_io import read_reid_embeddings_jsonl, read_reid_embeddings_npy
from deep_oc_sort_3d.reid.reid_types import normalize_embedding_l2


@dataclass
class CandidateEmbeddingLookupResult:
    """Result of a candidate embedding lookup."""

    candidate_id: str
    found: bool
    embedding: Optional[np.ndarray]
    embedding_dim: Optional[int]
    backend: Optional[str]
    source_path: Optional[str]
    missing_reason: str


class ReIDEmbeddingLookup:
    """Cached candidate/tracklet ReID embedding lookup."""

    def __init__(
        self,
        reid_root: Union[str, Path],
        prefer_candidate_embeddings: bool = True,
        normalize: bool = True,
    ) -> None:
        self.reid_root = Path(reid_root)
        self.prefer_candidate_embeddings = bool(prefer_candidate_embeddings)
        self.normalize = bool(normalize)
        self.loaded_scene_keys = set()
        self.by_candidate_id = {}
        self.by_full_key = {}
        self.by_track_key = {}
        self.load_stats = {"files_loaded": 0, "records_loaded": 0, "missing_files": 0}

    def load_for_subset_scene(self, subset: str, scene_name: str) -> None:
        """Load candidate and tracklet embeddings for one subset/scene."""
        scene_key = (str(subset), str(scene_name))
        if scene_key in self.loaded_scene_keys:
            return
        roots = []
        if self.prefer_candidate_embeddings:
            roots.append(self.reid_root / "candidate_embeddings" / subset / scene_name)
            roots.append(self.reid_root / "tracklet_embeddings" / subset / scene_name)
        else:
            roots.append(self.reid_root / "tracklet_embeddings" / subset / scene_name)
            roots.append(self.reid_root / "candidate_embeddings" / subset / scene_name)
        for root in roots:
            if not root.exists():
                self.load_stats["missing_files"] += 1
                continue
            for path in sorted(root.glob("*.jsonl")):
                self._load_jsonl(path)
            for npy_path, metadata_path in self._find_npy_metadata_pairs(root):
                self._load_npy(npy_path, metadata_path)
        self.loaded_scene_keys.add(scene_key)

    def get_embedding(self, candidate: MTMCTrackletCandidate) -> CandidateEmbeddingLookupResult:
        """Get an embedding for a candidate."""
        self.load_for_subset_scene(candidate.subset, candidate.scene_name)
        return self.get_embedding_by_key(
            candidate.subset,
            candidate.scene_name,
            candidate.camera_id,
            candidate.local_track_id,
            candidate.candidate_id,
            candidate.class_id,
        )

    def get_embedding_by_key(
        self,
        subset: str,
        scene_name: str,
        camera_id: str,
        local_track_id: int,
        candidate_id: Optional[str],
        class_id: int,
    ) -> CandidateEmbeddingLookupResult:
        """Lookup by candidate id, then full key, then track key."""
        lookup_candidate_id = "" if candidate_id is None else str(candidate_id)
        if lookup_candidate_id and lookup_candidate_id in self.by_candidate_id:
            return self._result(lookup_candidate_id, self.by_candidate_id[lookup_candidate_id], "")
        full_key = (str(subset), str(scene_name), str(camera_id), int(local_track_id), int(class_id))
        if full_key in self.by_full_key:
            return self._result(lookup_candidate_id, self.by_full_key[full_key], "")
        track_key = (str(subset), str(scene_name), str(camera_id), int(local_track_id))
        if track_key in self.by_track_key:
            return self._result(lookup_candidate_id, self.by_track_key[track_key], "")
        return CandidateEmbeddingLookupResult(
            candidate_id=lookup_candidate_id,
            found=False,
            embedding=None,
            embedding_dim=None,
            backend=None,
            source_path=None,
            missing_reason="embedding_not_found",
        )

    def summary(self) -> Dict[str, Any]:
        """Return lookup cache summary."""
        return {
            "reid_root": str(self.reid_root),
            "candidate_ids": len(self.by_candidate_id),
            "full_keys": len(self.by_full_key),
            "track_keys": len(self.by_track_key),
            "loaded_scenes": len(self.loaded_scene_keys),
            "load_stats": dict(self.load_stats),
        }

    def _load_jsonl(self, path: Path) -> None:
        records = read_reid_embeddings_jsonl(path)
        self.load_stats["files_loaded"] += 1
        self.load_stats["records_loaded"] += len(records)
        for record in records:
            self._index_record(record, str(path))

    def _load_npy(self, npy_path: Path, metadata_path: Path) -> None:
        records = read_reid_embeddings_npy(npy_path, metadata_path)
        self.load_stats["files_loaded"] += 1
        self.load_stats["records_loaded"] += len(records)
        for record in records:
            self._index_record(record, str(npy_path))

    def _index_record(self, record: Any, source_path: str) -> None:
        embedding = np.asarray(record.embedding, dtype=float).reshape(-1)
        if self.normalize:
            embedding = normalize_embedding_l2(embedding)
        payload = {
            "embedding": embedding,
            "embedding_dim": int(embedding.size),
            "backend": record.backend,
            "source_path": str(source_path),
        }
        if record.candidate_id not in (None, ""):
            self._set_if_absent(self.by_candidate_id, str(record.candidate_id), payload)
        if record.local_track_id is not None:
            full_key = (
                str(record.subset),
                str(record.scene_name),
                str(record.camera_id),
                int(record.local_track_id),
                int(record.class_id),
            )
            track_key = (
                str(record.subset),
                str(record.scene_name),
                str(record.camera_id),
                int(record.local_track_id),
            )
            self._set_if_absent(self.by_full_key, full_key, payload)
            self._set_if_absent(self.by_track_key, track_key, payload)

    def _find_npy_metadata_pairs(self, root: Path) -> Tuple[Tuple[Path, Path], ...]:
        pairs = []
        for npy_path in sorted(root.glob("*.npy")):
            candidates = [
                npy_path.with_suffix(".metadata.csv"),
                npy_path.with_name("%s.metadata.csv" % npy_path.stem),
            ]
            for metadata_path in candidates:
                if metadata_path.exists():
                    pairs.append((npy_path, metadata_path))
                    break
        return tuple(pairs)

    def _set_if_absent(self, mapping: Dict[Any, Any], key: Any, payload: Dict[str, Any]) -> None:
        if key not in mapping:
            mapping[key] = payload

    def _result(self, candidate_id: str, payload: Dict[str, Any], missing_reason: str) -> CandidateEmbeddingLookupResult:
        return CandidateEmbeddingLookupResult(
            candidate_id=str(candidate_id),
            found=True,
            embedding=payload.get("embedding"),
            embedding_dim=payload.get("embedding_dim"),
            backend=payload.get("backend"),
            source_path=payload.get("source_path"),
            missing_reason=missing_reason,
        )
