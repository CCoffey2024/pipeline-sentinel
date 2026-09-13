from pathlib import Path

from pipeline_sentinel.types import Detection, FrameRecord


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
    assert detection.xyxy == (10, 20, 40, 60)
