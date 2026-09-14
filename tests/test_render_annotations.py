from __future__ import annotations

import numpy as np

from pipeline_sentinel.pipeline import _class_color, _draw_track_annotation
from pipeline_sentinel.types import Track


def _track(label: str = "car") -> Track:
    return Track(
        track_id=7,
        frame_number=0,
        timestamp_s=0.0,
        label=label,
        confidence=0.91,
        x1=20,
        y1=30,
        x2=100,
        y2=80,
        source="test",
        hits=1,
        first_frame_number=0,
        first_timestamp_s=0.0,
    )


def test_class_colors_are_stable_and_non_white() -> None:
    car = _class_color("car")
    assert car == _class_color("CAR")
    assert car == _class_color(" car ")
    assert car != _class_color("person")
    assert car not in {(255, 255, 255), (180, 180, 180)}


def test_track_annotation_uses_class_color() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    track = _track("car")
    color = _class_color(track.label)

    _draw_track_annotation(image, track, is_alert=False, is_anomaly=False)

    assert tuple(int(value) for value in image[30, 20]) == color
    assert np.count_nonzero(image) > 0


def test_alert_and_anomaly_annotations_remain_colored() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    track = _track("person")
    color = _class_color(track.label)

    _draw_track_annotation(image, track, is_alert=True, is_anomaly=True)

    assert tuple(int(value) for value in image[30, 20]) == color
