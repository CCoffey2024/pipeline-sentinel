import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_sentinel.demo import run_demo
from pipeline_sentinel.pipeline import DETECTION_COLUMNS, PipelineSentinel
from pipeline_sentinel.synthetic import generate_demo_video
from pipeline_sentinel.types import Detection, FrameContext


class ObservationOnlyDetector:
    """Learned-detector stand-in: emits observations but no mission event semantics."""

    name = "observation_only"

    def detect(self, frame: FrameContext) -> list[Detection]:
        return [
            Detection(
                frame_number=frame.frame_number,
                label="car",
                confidence=0.8,
                x1=5,
                y1=5,
                x2=25,
                y2=25,
                source=self.name,
            )
        ]


def test_end_to_end_reference_demo(tmp_path: Path) -> None:
    artifacts = run_demo(tmp_path / "demo", frame_count=120, size=(320, 180))

    assert artifacts.annotated_video.exists() and artifacts.annotated_video.stat().st_size > 0
    assert artifacts.detections_csv.exists()
    assert artifacts.alerts_csv.exists()
    assert artifacts.run_manifest.exists()

    detections = pd.read_csv(artifacts.detections_csv)
    alerts = pd.read_csv(artifacts.alerts_csv)
    assert not detections.empty
    assert not alerts.empty
    assert "normal_maintenance" not in set(alerts["scenario_role"])
    assert "intrusion_vehicle" in set(alerts["scenario_role"])

    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))
    assert manifest["detector_backend"] == "ground_truth"
    assert manifest["frames_processed"] == 120
    assert manifest["detections_emitted"] == len(detections)
    assert manifest["alerts_emitted"] == len(alerts)


def test_detection_is_not_automatically_an_alert(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    gt = tmp_path / "unused_gt.csv"
    generate_demo_video(video, gt, frame_count=3, size=(64, 64))

    artifacts = PipelineSentinel(ObservationOnlyDetector()).run_video(video, tmp_path / "run")
    detections = pd.read_csv(artifacts.detections_csv)
    alerts = pd.read_csv(artifacts.alerts_csv)
    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))

    assert len(detections) == 3
    assert alerts.empty
    assert list(alerts.columns) == DETECTION_COLUMNS
    assert manifest["detector_backend"] == "observation_only"
    assert manifest["frames_processed"] == 3
    assert manifest["detections_emitted"] == 3
    assert manifest["alerts_emitted"] == 0


def test_pipeline_accepts_image_sequence_frames(tmp_path: Path) -> None:
    frames = [
        FrameContext(
            frame_number=10 + index,
            timestamp_s=index / 5.0,
            image=np.zeros((48, 64, 3), dtype=np.uint8),
            source_path=Path(f"frame_{index}.jpg"),
            sensor_id="VISDRONE:test",
            modality="EO",
        )
        for index in range(3)
    ]

    artifacts = PipelineSentinel(ObservationOnlyDetector()).run_frames(
        frames,
        tmp_path / "sequence",
        render_fps=5.0,
        source_name="fixture-sequence",
        run_metadata={"dataset": "fixture"},
    )

    detections = pd.read_csv(artifacts.detections_csv)
    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))

    assert len(detections) == 3
    assert detections["frame_number"].tolist() == [10, 11, 12]
    assert manifest["input_source"] == "fixture-sequence"
    assert manifest["first_frame_number"] == 10
    assert manifest["last_frame_number"] == 12
    assert manifest["metadata"]["dataset"] == "fixture"
