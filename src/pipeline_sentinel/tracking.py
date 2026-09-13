from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import Detection, FrameContext, Track


class Tracker(Protocol):
    """Framework-neutral contract for associating detections across frames."""

    name: str

    def update(self, frame: FrameContext, detections: list[Detection]) -> list[Track]: ...


def box_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    """Return intersection-over-union for two XYXY boxes."""

    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    left_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
    right_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


@dataclass(slots=True)
class _TrackState:
    track_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    source: str
    hits: int
    first_frame_number: int
    first_timestamp_s: float
    last_frame_number: int
    last_timestamp_s: float
    missed_updates: int
    scenario_role: str | None
    metadata: dict[str, object]


class IoUTracker:
    """Small deterministic tracker used as Pipeline Sentinel's baseline tracker.

    Matching is greedy, one-to-one, class-aware IoU association. The component intentionally owns
    no detector- or framework-specific objects, which lets a future ByteTrack/DeepSORT adapter
    replace it without changing downstream event code.
    """

    name = "iou"

    def __init__(self, *, iou_threshold: float = 0.30, max_missed_updates: int = 2) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_missed_updates < 0:
            raise ValueError("max_missed_updates cannot be negative")
        self.iou_threshold = float(iou_threshold)
        self.max_missed_updates = int(max_missed_updates)
        self._next_track_id = 1
        self._active: dict[int, _TrackState] = {}

    def update(self, frame: FrameContext, detections: list[Detection]) -> list[Track]:
        candidates: list[tuple[float, int, int]] = []
        for track_id, state in self._active.items():
            for detection_index, detection in enumerate(detections):
                if detection.label != state.label:
                    continue
                overlap = box_iou(state.bbox, detection.xyxy)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, track_id, detection_index))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, int] = {}
        for _overlap, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = track_id

        for track_id, state in list(self._active.items()):
            if track_id not in matched_tracks:
                state.missed_updates += 1
                if state.missed_updates > self.max_missed_updates:
                    del self._active[track_id]

        outputs: list[Track] = []
        for detection_index, detection in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                state = _TrackState(
                    track_id=track_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=detection.xyxy,
                    source=detection.source,
                    hits=1,
                    first_frame_number=frame.frame_number,
                    first_timestamp_s=frame.timestamp_s,
                    last_frame_number=frame.frame_number,
                    last_timestamp_s=frame.timestamp_s,
                    missed_updates=0,
                    scenario_role=detection.scenario_role,
                    metadata=dict(detection.metadata),
                )
                self._active[track_id] = state
            else:
                state = self._active[track_id]
                state.confidence = detection.confidence
                state.bbox = detection.xyxy
                state.source = detection.source
                state.hits += 1
                state.last_frame_number = frame.frame_number
                state.last_timestamp_s = frame.timestamp_s
                state.missed_updates = 0
                state.scenario_role = detection.scenario_role
                state.metadata = dict(detection.metadata)

            metadata = dict(state.metadata)
            if detection.object_id is not None:
                metadata["detection_object_id"] = int(detection.object_id)
            outputs.append(
                Track(
                    track_id=state.track_id,
                    frame_number=frame.frame_number,
                    timestamp_s=frame.timestamp_s,
                    label=state.label,
                    confidence=state.confidence,
                    x1=state.bbox[0],
                    y1=state.bbox[1],
                    x2=state.bbox[2],
                    y2=state.bbox[3],
                    source=self.name,
                    hits=state.hits,
                    first_frame_number=state.first_frame_number,
                    first_timestamp_s=state.first_timestamp_s,
                    scenario_role=state.scenario_role,
                    metadata=metadata,
                )
            )

        outputs.sort(key=lambda track: track.track_id)
        return outputs
