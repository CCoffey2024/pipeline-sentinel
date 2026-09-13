from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pipeline_sentinel.types import FrameContext
from pipeline_sentinel.yolo import YoloDetector


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.array(
            [
                [10.2, 20.4, 80.8, 90.1],
                [95.0, 95.0, 110.0, 120.0],
            ],
            dtype=np.float32,
        )
        self.conf = np.array([0.91, 0.80], dtype=np.float32)
        self.cls = np.array([2, 0], dtype=np.float32)


class FakeResult:
    def __init__(self) -> None:
        self.boxes = FakeBoxes()
        self.names = {0: "person", 2: "car"}


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[FakeResult]:
        self.calls.append(kwargs)
        return [FakeResult()]


def _frame() -> FrameContext:
    return FrameContext(
        frame_number=12,
        timestamp_s=1.2,
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        sensor_id="EO_CAM_01",
        modality="EO",
    )


def test_yolo_adapter_normalizes_framework_output_without_ultralytics() -> None:
    model = FakeModel()
    detector = YoloDetector(
        model_name="fake.pt",
        confidence=0.30,
        iou=0.60,
        imgsz=512,
        device="cpu",
        class_ids=[0, 2],
        model=model,
    )

    detections = detector.detect(_frame())

    assert len(detections) == 2
    first, second = detections
    assert first.frame_number == 12
    assert first.label == "car"
    assert first.xyxy == (10, 20, 81, 90)
    assert first.source == "yolo"
    assert first.scenario_role is None
    assert first.metadata == {"class_id": 2, "model": "fake.pt"}
    assert first.confidence == pytest.approx(0.91)

    # Adapter owns framework cleanup: boxes are clipped to the Pipeline Sentinel frame boundary.
    assert second.label == "person"
    assert second.xyxy == (95, 95, 99, 99)

    call = model.calls[0]
    assert call["source"].shape == (100, 100, 3)
    assert call["conf"] == 0.30
    assert call["iou"] == 0.60
    assert call["imgsz"] == 512
    assert call["device"] == "cpu"
    assert call["classes"] == [0, 2]
    assert call["verbose"] is False


def test_yolo_adapter_validates_configuration() -> None:
    model = FakeModel()

    with pytest.raises(ValueError, match="confidence"):
        YoloDetector(confidence=1.5, model=model)
    with pytest.raises(ValueError, match="iou"):
        YoloDetector(iou=-0.1, model=model)
    with pytest.raises(ValueError, match="imgsz"):
        YoloDetector(imgsz=0, model=model)
    with pytest.raises(ValueError, match="class_ids"):
        YoloDetector(class_ids=[-1], model=model)
