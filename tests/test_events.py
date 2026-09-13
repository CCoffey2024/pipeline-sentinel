import numpy as np

from pipeline_sentinel.events import (
    DwellEventDetector,
    ScenarioRoleEventDetector,
    SeverityAlertPolicy,
)
from pipeline_sentinel.types import FrameContext, Track


def _frame(number: int) -> FrameContext:
    return FrameContext(
        frame_number=number,
        timestamp_s=number / 10.0,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )


def _track(
    frame_number: int,
    *,
    track_id: int = 1,
    hits: int = 1,
    x1: int = 10,
    role: str | None = None,
) -> Track:
    return Track(
        track_id=track_id,
        frame_number=frame_number,
        timestamp_s=frame_number / 10.0,
        label="person",
        confidence=0.9,
        x1=x1,
        y1=10,
        x2=x1 + 10,
        y2=30,
        source="iou",
        hits=hits,
        first_frame_number=0,
        first_timestamp_s=0.0,
        scenario_role=role,
    )


def test_scenario_role_event_is_emitted_once_and_normal_is_not_alerted() -> None:
    detector = ScenarioRoleEventDetector()
    policy = SeverityAlertPolicy(minimum_severity="warning")

    normal_events = detector.update(_frame(0), [_track(0, role="normal_maintenance")])
    repeated = detector.update(_frame(1), [_track(1, hits=2, role="normal_maintenance")])

    assert len(normal_events) == 1
    assert normal_events[0].severity == "info"
    assert repeated == []
    assert policy.evaluate(normal_events[0]) is None


def test_warning_event_is_promoted_to_alert() -> None:
    detector = ScenarioRoleEventDetector()
    policy = SeverityAlertPolicy(minimum_severity="warning")

    events = detector.update(_frame(0), [_track(0, role="dismount_loiter")])
    alert = policy.evaluate(events[0])

    assert events[0].event_type == "dismount_loiter"
    assert events[0].severity == "warning"
    assert alert is not None
    assert alert.alert_type == "dismount_loiter"
    assert alert.event_id == events[0].event_id


def test_dwell_event_requires_persistence_and_low_displacement() -> None:
    detector = DwellEventDetector(min_hits=3, max_displacement_px=5.0, labels={"person"})

    assert detector.update(_frame(0), [_track(0, hits=1, x1=10)]) == []
    assert detector.update(_frame(1), [_track(1, hits=2, x1=11)]) == []
    events = detector.update(_frame(2), [_track(2, hits=3, x1=12)])
    repeated = detector.update(_frame(3), [_track(3, hits=4, x1=12)])

    assert len(events) == 1
    assert events[0].event_type == "dwell"
    assert events[0].track_id == 1
    assert repeated == []


def test_dwell_event_rejects_track_that_moves_too_far() -> None:
    detector = DwellEventDetector(min_hits=3, max_displacement_px=5.0)

    detector.update(_frame(0), [_track(0, hits=1, x1=10)])
    detector.update(_frame(1), [_track(1, hits=2, x1=20)])
    events = detector.update(_frame(2), [_track(2, hits=3, x1=30)])

    assert events == []
