import json
from pathlib import Path

import pandas as pd

from pipeline_sentinel.demo import run_demo
from pipeline_sentinel.pipeline import PipelineSentinel
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
    assert artifacts.alerts_csv.exists()
    assert artifacts.run_manifest.exists()

    alerts = pd.read_csv(artifacts.alerts_csv)
    assert not alerts.empty
    assert "normal_maintenance" not in set(alerts["scenario_role"])
    assert "intrusion_vehicle" in set(alerts["scenario_role"])

    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))
    assert manifest["detector_backend"] == "ground_truth"
    assert manifest["frames_processed"] == 120
    assert manifest["detections_emitted"] > 0
    assert manifest["alerts_emitted"] == len(alerts)


def test_detection_is_not_automatically_an_alert(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    gt = tmp_path / "unused_gt.csv"
    generate_demo_video(video, gt, frame_count=3, size=(64, 64))

    artifacts = PipelineSentinel(ObservationOnlyDetector()).run_video(video, tmp_path / "run")
    alerts = pd.read_csv(artifacts.alerts_csv)
    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))

    assert alerts.empty
    assert list(alerts.columns) == [
        "frame_number",
        "timestamp_s",
        "label",
        "object_id",
        "scenario_role",
        "confidence",
        "detector",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    assert manifest["detector_backend"] == "observation_only"
    assert manifest["frames_processed"] == 3
    assert manifest["detections_emitted"] == 3
    assert manifest["alerts_emitted"] == 0
