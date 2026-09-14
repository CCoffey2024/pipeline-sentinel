from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .embeddings import Embedder
from .types import AnomalyObservation, FrameContext, Track


def cosine_distance_to_centroid(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Compute cosine distance from each embedding to one centroid vector."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    center = np.asarray(centroid, dtype=np.float32).reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2-D matrix")
    if center.shape[1] != matrix.shape[1]:
        raise ValueError("embedding dimension does not match reference centroid")

    row_norm = np.linalg.norm(matrix, axis=1)
    center_norm = float(np.linalg.norm(center))
    if center_norm == 0.0 or np.any(row_norm == 0.0):
        raise ValueError("cosine distance is undefined for zero-length embeddings")
    similarity = (matrix @ center.ravel()) / (row_norm * center_norm)
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1.0 - similarity


@dataclass(frozen=True, slots=True)
class AnomalyReference:
    """Persisted normal-activity centroid and threshold for one embedder."""

    embedder_name: str
    centroid: np.ndarray = field(repr=False)
    threshold: float
    sample_count: int
    quantile: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        centroid = np.asarray(self.centroid, dtype=np.float32)
        if centroid.ndim != 1 or centroid.size == 0:
            raise ValueError("centroid must be a non-empty 1-D vector")
        if not self.embedder_name:
            raise ValueError("embedder_name cannot be empty")
        if self.threshold < 0:
            raise ValueError("threshold cannot be negative")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be between 0 and 1")
        object.__setattr__(self, "centroid", centroid)

    @property
    def embedding_dim(self) -> int:
        return int(self.centroid.size)

    def save(self, path: Path) -> Path:
        target = Path(path).expanduser().resolve()
        if target.suffix.lower() != ".npz":
            target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1,
            "embedder_name": self.embedder_name,
            "threshold": float(self.threshold),
            "sample_count": int(self.sample_count),
            "quantile": float(self.quantile),
            "embedding_dim": self.embedding_dim,
            "metadata": self.metadata,
        }
        np.savez_compressed(
            target,
            centroid=self.centroid,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return target

    @classmethod
    def load(cls, path: Path) -> AnomalyReference:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as data:
            if "centroid" not in data or "metadata_json" not in data:
                raise ValueError(f"Invalid anomaly reference artifact: {source}")
            centroid = np.asarray(data["centroid"], dtype=np.float32)
            metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported anomaly reference format version")
        reference = cls(
            embedder_name=str(metadata["embedder_name"]),
            centroid=centroid,
            threshold=float(metadata["threshold"]),
            sample_count=int(metadata["sample_count"]),
            quantile=float(metadata["quantile"]),
            metadata=dict(metadata.get("metadata") or {}),
        )
        if int(metadata.get("embedding_dim", reference.embedding_dim)) != reference.embedding_dim:
            raise ValueError("Anomaly reference embedding dimension metadata is inconsistent")
        return reference


def fit_normal_reference(
    embeddings: np.ndarray,
    *,
    embedder_name: str,
    quantile: float = 0.95,
    metadata: dict[str, Any] | None = None,
) -> AnomalyReference:
    """Fit the Notebook-06 normal-centroid anomaly model."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("at least two normal embeddings are required")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    centroid = matrix.mean(axis=0)
    scores = cosine_distance_to_centroid(matrix, centroid)
    threshold = float(np.quantile(scores, quantile))
    return AnomalyReference(
        embedder_name=embedder_name,
        centroid=centroid,
        threshold=threshold,
        sample_count=matrix.shape[0],
        quantile=float(quantile),
        metadata=dict(metadata or {}),
    )


class AnomalyAnalyzer(Protocol):
    """Score runtime track observations against a learned normal reference."""

    name: str

    def reset(self) -> None: ...

    def update(
        self,
        frame: FrameContext,
        tracks: list[Track],
    ) -> list[AnomalyObservation]: ...


class TrackCropAnomalyAnalyzer:
    """Embed tracked object crops and score them against a normal centroid."""

    name = "track_crop_normal_centroid"

    def __init__(
        self,
        embedder: Embedder,
        reference: AnomalyReference,
        *,
        labels: set[str] | None = None,
        min_track_hits: int = 3,
        pad_px: int = 6,
        min_crop_size: int = 12,
    ) -> None:
        if min_track_hits < 1:
            raise ValueError("min_track_hits must be positive")
        if pad_px < 0:
            raise ValueError("pad_px cannot be negative")
        if min_crop_size < 1:
            raise ValueError("min_crop_size must be positive")
        if reference.embedder_name != embedder.name:
            raise ValueError(
                f"Reference embedder '{reference.embedder_name}' does not match '{embedder.name}'"
            )
        self.embedder = embedder
        self.reference = reference
        self.labels = set(labels) if labels is not None else None
        self.min_track_hits = int(min_track_hits)
        self.pad_px = int(pad_px)
        self.min_crop_size = int(min_crop_size)

    def reset(self) -> None:
        return None

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
    ) -> list[AnomalyObservation]:
        selected: list[Track] = []
        crops: list[np.ndarray] = []
        for track in tracks:
            if track.hits < self.min_track_hits:
                continue
            if self.labels is not None and track.label not in self.labels:
                continue
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
        if embeddings.shape[1] != self.reference.embedding_dim:
            raise RuntimeError(
                "Embedder output dimension does not match the anomaly reference artifact"
            )
        scores = cosine_distance_to_centroid(embeddings, self.reference.centroid)

        observations: list[AnomalyObservation] = []
        for track, score in zip(selected, scores, strict=True):
            value = float(score)
            observations.append(
                AnomalyObservation(
                    frame_number=frame.frame_number,
                    timestamp_s=frame.timestamp_s,
                    track_id=track.track_id,
                    label=track.label,
                    score=value,
                    threshold=float(self.reference.threshold),
                    is_anomaly=value > self.reference.threshold,
                    source=self.name,
                    metadata={
                        "embedder": self.embedder.name,
                        "reference_samples": self.reference.sample_count,
                        "reference_quantile": self.reference.quantile,
                        "track_hits": track.hits,
                    },
                )
            )
        return observations


def encode_in_batches(
    embedder: Embedder,
    images: Sequence[np.ndarray],
    *,
    batch_size: int = 16,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunks: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        chunk = np.asarray(embedder.encode(images[start : start + batch_size]), dtype=np.float32)
        if chunk.ndim != 2:
            raise RuntimeError("Embedder returned an invalid batch shape")
        chunks.append(chunk)
    if not chunks:
        raise ValueError("no images were provided")
    return np.vstack(chunks)
