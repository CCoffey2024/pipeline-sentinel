from __future__ import annotations

from typing import Protocol

import numpy as np

from .embeddings import Embedder
from .types import FrameContext, RepresentationObservation, Track


class RepresentationAnalyzer(Protocol):
    """Produce descriptive track representations without assigning anomaly meaning."""

    name: str

    def reset(self) -> None: ...

    def provenance(self) -> dict[str, object]: ...

    def update(
        self,
        frame: FrameContext,
        tracks: list[Track],
    ) -> list[RepresentationObservation]: ...


class TrackCropRepresentationAnalyzer:
    """Periodically embed mature tracked-object crops with a bounded per-frame workload."""

    name = "track_crop_representation"

    def __init__(
        self,
        embedder: Embedder,
        *,
        labels: set[str] | None = None,
        min_track_hits: int = 3,
        sample_every_n_hits: int = 15,
        max_per_frame: int = 16,
        pad_px: int = 6,
        min_crop_size: int = 12,
    ) -> None:
        if min_track_hits < 1:
            raise ValueError("min_track_hits must be positive")
        if sample_every_n_hits < 1:
            raise ValueError("sample_every_n_hits must be positive")
        if max_per_frame < 1:
            raise ValueError("max_per_frame must be positive")
        if pad_px < 0:
            raise ValueError("pad_px cannot be negative")
        if min_crop_size < 1:
            raise ValueError("min_crop_size must be positive")
        self.embedder = embedder
        self.labels = set(labels) if labels is not None else None
        self.min_track_hits = int(min_track_hits)
        self.sample_every_n_hits = int(sample_every_n_hits)
        self.max_per_frame = int(max_per_frame)
        self.pad_px = int(pad_px)
        self.min_crop_size = int(min_crop_size)
        self._last_sampled_hits: dict[int, int] = {}
        self._previous_embeddings: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self._last_sampled_hits.clear()
        self._previous_embeddings.clear()

    def provenance(self) -> dict[str, object]:
        return {
            "embedder": self.embedder.name,
            "labels": sorted(self.labels) if self.labels is not None else None,
            "min_track_hits": self.min_track_hits,
            "sample_every_n_hits": self.sample_every_n_hits,
            "max_per_frame": self.max_per_frame,
            "pad_px": self.pad_px,
            "min_crop_size": self.min_crop_size,
            "embedding_normalization": "l2",
            "crop_resize": "aspect_preserving_letterbox",
        }

    def _eligible(self, track: Track) -> bool:
        if track.hits < self.min_track_hits:
            return False
        if self.labels is not None and track.label not in self.labels:
            return False
        if (track.hits - self.min_track_hits) % self.sample_every_n_hits != 0:
            return False
        return self._last_sampled_hits.get(track.track_id) != track.hits

    def _crop(self, frame: FrameContext, track: Track) -> np.ndarray | None:
        x1 = max(0, track.x1 - self.pad_px)
        y1 = max(0, track.y1 - self.pad_px)
        x2 = min(frame.width, track.x2 + self.pad_px)
        y2 = min(frame.height, track.y2 + self.pad_px)
        if x2 - x1 < self.min_crop_size or y2 - y1 < self.min_crop_size:
            return None
        crop = frame.image[y1:y2, x1:x2]
        return crop if crop.size else None

    def update(
        self,
        frame: FrameContext,
        tracks: list[Track],
    ) -> list[RepresentationObservation]:
        candidates = sorted(
            (track for track in tracks if self._eligible(track)),
            key=lambda track: (-track.confidence, track.track_id),
        )[: self.max_per_frame]
        selected: list[Track] = []
        crops: list[np.ndarray] = []
        for track in candidates:
            crop = self._crop(frame, track)
            if crop is None:
                continue
            selected.append(track)
            crops.append(crop)

        if not crops:
            return []

        embeddings = np.asarray(self.embedder.encode(crops), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(selected):
            raise RuntimeError("Embedder returned an invalid batch shape")
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
            raise RuntimeError("Embedder returned a non-finite or zero-length representation")
        embeddings = embeddings / norms[:, np.newaxis]

        observations: list[RepresentationObservation] = []
        for track, embedding in zip(selected, embeddings, strict=True):
            previous = self._previous_embeddings.get(track.track_id)
            similarity = None
            if previous is not None:
                similarity = float(np.clip(np.dot(previous, embedding), -1.0, 1.0))
            observations.append(
                RepresentationObservation(
                    frame_number=frame.frame_number,
                    timestamp_s=frame.timestamp_s,
                    track_id=track.track_id,
                    label=track.label,
                    embedding=embedding,
                    source=self.embedder.name,
                    track_hits=track.hits,
                    x1=track.x1,
                    y1=track.y1,
                    x2=track.x2,
                    y2=track.y2,
                    previous_cosine_similarity=similarity,
                )
            )
            self._last_sampled_hits[track.track_id] = track.hits
            self._previous_embeddings[track.track_id] = embedding.copy()
        return observations
