from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Modality = Literal["EO", "IR", "OTHER"]


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Canonical persisted record for one extracted video frame.

    Downstream components can use this contract for provenance and manifests without knowing how
    the frame was acquired.
    """

    frame_id: str
    frame_number: int
    timestamp_s: float
    image_path: Path
    width: int
    height: int
    sensor_id: str
    modality: Modality
    source_path: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        data["source_path"] = str(self.source_path)
        return data


@dataclass(frozen=True, slots=True)
class FrameContext:
    """In-memory frame passed between runtime components.

    ``FrameRecord`` is the serializable ETL/manifest contract. ``FrameContext`` is the runtime
    contract that carries the actual pixels required by detectors and other online components.
    """

    frame_number: int
    timestamp_s: float
    image: np.ndarray = field(repr=False)
    source_path: Path | None = None
    sensor_id: str = "VIDEO_01"
    modality: Modality = "EO"

    def __post_init__(self) -> None:
        if self.frame_number < 0:
            raise ValueError("frame_number cannot be negative")
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s cannot be negative")
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            raise ValueError("image must be a non-empty numpy array")
        if self.image.ndim not in {2, 3}:
            raise ValueError("image must have shape HxW or HxWxC")

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(frozen=True, slots=True)
class Detection:
    """Portable detector output used by the rest of Pipeline Sentinel."""

    frame_number: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    source: str
    object_id: int | None = None
    scenario_role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
