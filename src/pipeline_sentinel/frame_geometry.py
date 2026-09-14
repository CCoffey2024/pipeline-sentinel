from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .types import FrameContext


@dataclass(slots=True)
class FrameGeometryNormalizer:
    """Normalize a frame stream to one aspect-preserving coordinate system.

    Encoded video normally has stable dimensions, but folders and browser-selected still images may
    mix resolutions or orientations. Detection, tracking, annotation, and video encoding all need a
    consistent coordinate system, so the first frame establishes a canonical canvas and later
    mismatched frames are scaled and letterboxed onto it.
    """

    width: int
    height: int
    fill_value: int = 114
    total_frames: int = 0
    normalized_frames: int = 0
    source_dimensions: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("canonical frame dimensions must be positive")
        if not 0 <= self.fill_value <= 255:
            raise ValueError("fill_value must be between 0 and 255")

    @classmethod
    def from_frame(cls, frame: FrameContext) -> FrameGeometryNormalizer:
        return cls(width=frame.width, height=frame.height)

    def normalize(self, frame: FrameContext) -> FrameContext:
        """Return ``frame`` unchanged or with pixels letterboxed to the canonical canvas."""

        self.total_frames += 1
        self.source_dimensions[f"{frame.width}x{frame.height}"] += 1
        if frame.width == self.width and frame.height == self.height:
            return frame

        scale = min(self.width / frame.width, self.height / frame.height)
        resized_width = min(self.width, max(1, round(frame.width * scale)))
        resized_height = min(self.height, max(1, round(frame.height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            frame.image,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        if frame.image.ndim == 3 and resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        canvas_shape = (self.height, self.width, *frame.image.shape[2:])
        canvas = np.full(canvas_shape, self.fill_value, dtype=frame.image.dtype)
        x_offset = (self.width - resized_width) // 2
        y_offset = (self.height - resized_height) // 2
        canvas[
            y_offset : y_offset + resized_height,
            x_offset : x_offset + resized_width,
            ...,
        ] = resized

        self.normalized_frames += 1
        return replace(frame, image=canvas)

    def provenance(self) -> dict[str, object]:
        """Return JSON-safe geometry provenance for the run manifest."""

        return {
            "policy": "letterbox_to_first_frame",
            "canonical_width": self.width,
            "canonical_height": self.height,
            "fill_value": self.fill_value,
            "total_frames": self.total_frames,
            "normalized_frames": self.normalized_frames,
            "source_dimensions": dict(sorted(self.source_dimensions.items())),
        }
