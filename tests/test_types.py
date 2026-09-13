from pathlib import Path

import numpy as np
import pytest

from pipeline_sentinel.types import Detection, FrameContext, FrameRecord


def test_core_contracts_round_trip() -> None:
    frame = FrameRecord(
        frame_id="demo_000001",
        frame_number=1,
        timestamp_s=0.1,
        image_path=Path("frame.jpg"),
        width=640,
        height=360,
        sensor_id="EO_CAM_01",
        modality="EO",
        source_path=Path("demo.mp4"),
    )
    context = FrameContext(
        frame_number=1,
        timestamp_s=0.1,
        image=np.zeros((360, 640, 3), dtype=np.uint8),
        source_path=Path("demo.mp4"),
        sensor_id="EO_CAM_01",
        modality="EO",
    )
    detection = Detection(
        frame_number=1,
        label="vehicle",
        confidence=0.9,
        x1=10,
        y1=20,
        x2=40,
        y2=60,
        source="test",
    )

    assert frame.to_dict()["frame_id"] == "demo_000001"
    assert context.width == 640
    assert context.height == 360
    assert detection.xyxy == (10, 20, 40, 60)


def test_frame_context_rejects_invalid_images() -> None:
    with pytest.raises(ValueError, match="non-empty numpy array"):
        FrameContext(frame_number=0, timestamp_s=0.0, image=np.array([]))
