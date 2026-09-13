import numpy as np

from pipeline_sentinel.tracking import IoUTracker, box_iou
from pipeline_sentinel.types import Detection, FrameContext


def _frame(number: int) -> FrameContext:
    return FrameContext(
        frame_number=number,
        timestamp_s=number / 10.0,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )


def _detection(frame_number: int, *, label: str = "person", x1: int = 10) -> Detection:
    return Detection(
        frame_number=frame_number,
        label=label,
        confidence=0.9,
        x1=x1,
        y1=10,
        x2=x1 + 20,
        y2=30,
        source="fixture",
    )


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == 1 / 3


def test_iou_tracker_preserves_id_across_small_motion() -> None:
    tracker = IoUTracker(iou_threshold=0.30)

    first = tracker.update(_frame(0), [_detection(0, x1=10)])
    second = tracker.update(_frame(1), [_detection(1, x1=12)])

    assert first[0].track_id == 1
    assert second[0].track_id == 1
    assert second[0].hits == 2
    assert second[0].first_frame_number == 0


def test_iou_tracker_is_class_aware() -> None:
    tracker = IoUTracker(iou_threshold=0.30)

    person = tracker.update(_frame(0), [_detection(0, label="person")])
    car = tracker.update(_frame(1), [_detection(1, label="car")])

    assert person[0].track_id == 1
    assert car[0].track_id == 2


def test_iou_tracker_expires_after_missed_updates() -> None:
    tracker = IoUTracker(iou_threshold=0.30, max_missed_updates=1)

    assert tracker.update(_frame(0), [_detection(0)])[0].track_id == 1
    assert tracker.update(_frame(1), []) == []
    assert tracker.update(_frame(2), []) == []
    assert tracker.update(_frame(3), [_detection(3)])[0].track_id == 2


def test_iou_tracker_reset_restarts_run_state() -> None:
    tracker = IoUTracker()
    assert tracker.update(_frame(0), [_detection(0)])[0].track_id == 1

    tracker.reset()

    reset_track = tracker.update(_frame(10), [_detection(10)])
    assert reset_track[0].track_id == 1
    assert reset_track[0].hits == 1
