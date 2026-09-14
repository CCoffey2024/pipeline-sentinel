import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_sentinel.demo import run_demo
from pipeline_sentinel.pipeline import ALERT_COLUMNS, PipelineSentinel
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


class RecordingDetector(ObservationOnlyDetector):
    def __init__(self) -> None:
        self.frame_shapes: list[tuple[int, ...]] = []

    def detect(self, frame: FrameContext) -> list[Detection]:
        self.frame_shapes.append(frame.image.shape)
        return super().detect(frame)


def test_end_to_end_reference_demo(tmp_path: Path) -> None:
    artifacts = run_demo(tmp_path / "demo", frame_count=120, size=(320, 180))

    assert artifacts.annotated_video.exists() and artifacts.annotated_video.stat().st_size > 0
    assert artifacts.detections_csv.exists()
    assert artifacts.tracks_csv.exists()
    assert artifacts.events_csv.exists()
    assert artifacts.alerts_csv.exists()
    assert artifacts.run_manifest.exists()

    detections = pd.read_csv(artifacts.detections_csv)
    tracks = pd.read_csv(artifacts.tracks_csv)
    events = pd.read_csv(artifacts.events_csv)
    alerts = pd.read_csv(artifacts.alerts_csv)
    assert not detections.empty
    assert not tracks.empty
    assert not events.empty
    assert not alerts.empty
    assert "normal_maintenance" in set(events["event_type"])
    assert "normal_maintenance" not in set(alerts["alert_type"])
    assert "intrusion_vehicle" in set(alerts["alert_type"])

    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))
    assert manifest["detector_backend"] == "ground_truth"
    assert manifest["tracker_backend"] == "iou"
    assert manifest["event_detector"] == "scenario_role"
    assert manifest["alert_policy"] == "severity"
    assert manifest["frames_processed"] == 120
    assert manifest["detections_emitted"] == len(detections)
    assert manifest["track_observations_emitted"] == len(tracks)
    assert manifest["events_emitted"] == len(events)
    assert manifest["alerts_emitted"] == len(alerts)


def test_detection_is_not_automatically_an_event_or_alert(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    gt = tmp_path / "unused_gt.csv"
    generate_demo_video(video, gt, frame_count=3, size=(64, 64))

    artifacts = PipelineSentinel(ObservationOnlyDetector()).run_video(video, tmp_path / "run")
    detections = pd.read_csv(artifacts.detections_csv)
    tracks = pd.read_csv(artifacts.tracks_csv)
    events = pd.read_csv(artifacts.events_csv)
    alerts = pd.read_csv(artifacts.alerts_csv)
    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))

    assert len(detections) == 3
    assert len(tracks) == 3
    assert tracks["track_id"].nunique() == 1
    assert events.empty
    assert alerts.empty
    assert list(alerts.columns) == ALERT_COLUMNS
    assert manifest["detector_backend"] == "observation_only"
    assert manifest["tracker_backend"] == "iou"
    assert manifest["event_detector"] is None
    assert manifest["frames_processed"] == 3
    assert manifest["detections_emitted"] == 3
    assert manifest["track_observations_emitted"] == 3
    assert manifest["unique_tracks"] == 1
    assert manifest["events_emitted"] == 0
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
    tracks = pd.read_csv(artifacts.tracks_csv)
    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))

    assert len(detections) == 3
    assert detections["frame_number"].tolist() == [10, 11, 12]
    assert tracks["track_id"].tolist() == [1, 1, 1]
    assert manifest["input_source"] == "fixture-sequence"
    assert manifest["first_frame_number"] == 10
    assert manifest["last_frame_number"] == 12
    assert manifest["metadata"]["dataset"] == "fixture"


def test_pipeline_normalizes_mixed_frame_sizes_before_detection(tmp_path: Path) -> None:
    frames = [
        FrameContext(
            frame_number=0,
            timestamp_s=0.0,
            image=np.zeros((48, 64, 3), dtype=np.uint8),
            source_path=Path("portraitish.jpg"),
        ),
        FrameContext(
            frame_number=1,
            timestamp_s=1.0,
            image=np.zeros((32, 96, 3), dtype=np.uint8),
            source_path=Path("landscape.jpg"),
        ),
    ]
    detector = RecordingDetector()

    artifacts = PipelineSentinel(detector).run_frames(
        frames,
        tmp_path / "mixed-size-sequence",
        render_fps=1.0,
    )

    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))
    normalization = manifest["metadata"]["frame_normalization"]
    assert detector.frame_shapes == [(48, 64, 3), (48, 64, 3)]
    assert manifest["frames_processed"] == 2
    assert normalization["policy"] == "letterbox_to_first_frame"
    assert normalization["canonical_width"] == 64
    assert normalization["canonical_height"] == 48
    assert normalization["normalized_frames"] == 1
    assert normalization["source_dimensions"] == {"64x48": 1, "96x32": 1}
