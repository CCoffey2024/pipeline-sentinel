from __future__ import annotations

import math
from typing import Protocol

from .types import Alert, Event, FrameContext, Severity, Track


class EventDetector(Protocol):
    """Convert track history into explicit temporal/semantic events."""

    name: str

    def reset(self) -> None: ...

    def update(self, frame: FrameContext, tracks: list[Track]) -> list[Event]: ...


class AlertPolicy(Protocol):
    """Decide whether an event should become a human-facing alert."""

    name: str

    def evaluate(self, event: Event) -> Alert | None: ...


class ScenarioRoleEventDetector:
    """Deterministic semantic-event adapter for the synthetic/reference backend.

    ``scenario_role`` exists only on known reference annotations. This adapter keeps that test-only
    semantic signal out of the tracker and turns it into a real ``Event`` exactly once per
    track/role. Learned detectors do not receive scenario roles and therefore cannot manufacture
    these events.
    """

    name = "scenario_role"

    def __init__(self, *, normal_roles: set[str] | None = None) -> None:
        self.normal_roles = set(normal_roles or {"normal_maintenance"})
        self._emitted: set[tuple[int, str]] = set()

    def reset(self) -> None:
        self._emitted.clear()

    def update(self, frame: FrameContext, tracks: list[Track]) -> list[Event]:
        events: list[Event] = []
        for track in tracks:
            role = track.scenario_role
            if not role:
                continue
            key = (track.track_id, role)
            if key in self._emitted:
                continue
            self._emitted.add(key)
            severity: Severity = "info" if role in self.normal_roles else "warning"
            events.append(
                Event(
                    event_id=f"scenario:{track.track_id}:{role}",
                    frame_number=frame.frame_number,
                    timestamp_s=frame.timestamp_s,
                    event_type=role,
                    severity=severity,
                    source=self.name,
                    message=f"Track {track.track_id} classified as scenario role '{role}'.",
                    track_id=track.track_id,
                    label=track.label,
                    confidence=track.confidence,
                    metadata={"hits": track.hits},
                )
            )
        return events


class DwellEventDetector:
    """Baseline track-history rule for persistent, spatially constrained objects.

    This is intentionally a small deterministic rule rather than a claim of mission-grade
    loitering analytics. It provides a real learned-detector path from tracks to events while the
    more sophisticated zone and behavior components are developed behind the same contract.
    """

    name = "dwell"

    def __init__(
        self,
        *,
        min_hits: int = 30,
        max_displacement_px: float = 40.0,
        labels: set[str] | None = None,
    ) -> None:
        if min_hits < 2:
            raise ValueError("min_hits must be at least 2")
        if max_displacement_px < 0:
            raise ValueError("max_displacement_px cannot be negative")
        self.min_hits = int(min_hits)
        self.max_displacement_px = float(max_displacement_px)
        self.labels = set(labels) if labels is not None else None
        self._origins: dict[int, tuple[float, float]] = {}
        self._max_displacement: dict[int, float] = {}
        self._emitted: set[int] = set()

    def reset(self) -> None:
        self._origins.clear()
        self._max_displacement.clear()
        self._emitted.clear()

    @staticmethod
    def _centroid(track: Track) -> tuple[float, float]:
        return ((track.x1 + track.x2) / 2.0, (track.y1 + track.y2) / 2.0)

    def update(self, frame: FrameContext, tracks: list[Track]) -> list[Event]:
        events: list[Event] = []
        for track in tracks:
            if self.labels is not None and track.label not in self.labels:
                continue
            centroid = self._centroid(track)
            origin = self._origins.setdefault(track.track_id, centroid)
            displacement = math.hypot(centroid[0] - origin[0], centroid[1] - origin[1])
            maximum = max(self._max_displacement.get(track.track_id, 0.0), displacement)
            self._max_displacement[track.track_id] = maximum

            if track.track_id in self._emitted:
                continue
            if track.hits < self.min_hits or maximum > self.max_displacement_px:
                continue

            self._emitted.add(track.track_id)
            events.append(
                Event(
                    event_id=f"dwell:{track.track_id}",
                    frame_number=frame.frame_number,
                    timestamp_s=frame.timestamp_s,
                    event_type="dwell",
                    severity="warning",
                    source=self.name,
                    message=(
                        f"Track {track.track_id} persisted for {track.hits} observations within "
                        f"{self.max_displacement_px:.1f}px of its origin."
                    ),
                    track_id=track.track_id,
                    label=track.label,
                    confidence=track.confidence,
                    metadata={
                        "hits": track.hits,
                        "duration_s": track.duration_s,
                        "max_displacement_px": maximum,
                        "configured_max_displacement_px": self.max_displacement_px,
                    },
                )
            )
        return events


class SeverityAlertPolicy:
    """Promote events at or above a configured severity threshold."""

    name = "severity"
    _RANK = {"info": 0, "warning": 1, "critical": 2}

    def __init__(self, *, minimum_severity: Severity = "warning") -> None:
        self.minimum_severity = minimum_severity

    def evaluate(self, event: Event) -> Alert | None:
        if self._RANK[event.severity] < self._RANK[self.minimum_severity]:
            return None
        return Alert(
            alert_id=f"alert:{event.event_id}",
            event_id=event.event_id,
            frame_number=event.frame_number,
            timestamp_s=event.timestamp_s,
            alert_type=event.event_type,
            severity=event.severity,
            source=self.name,
            message=event.message,
            track_id=event.track_id,
            label=event.label,
            confidence=event.confidence,
            metadata={"event_source": event.source},
        )
