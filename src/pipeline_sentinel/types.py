from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Modality = Literal["EO", "IR", "OTHER"]


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Canonical record for one extracted video frame.

    Downstream components consume this contract instead of knowing how the frame was acquired.
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
