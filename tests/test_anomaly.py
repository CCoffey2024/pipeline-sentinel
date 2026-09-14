from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline_sentinel.anomaly import (
    AnomalyReference,
    TrackCropAnomalyAnalyzer,
    cosine_distance_to_centroid,
    fit_normal_reference,
)
from pipeline_sentinel.events import ConsecutiveAnomalyEventDetector
from pipeline_sentinel.pipeline import PipelineSentinel
from pipeline_sentinel.reference import build_reference_from_directory
from pipeline_sentinel.types import Detection, FrameContext, Track


class MeanBucketEmbedder:
    name = "fake:mean-bucket"

    def encode(self, images):
        rows = []
        for image in images:
            if float(np.mean(image)) < 100.0:
                rows.append([1.0, 0.0])
            else:
                rows.append([0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


class FixedDetector:
    name = "fixed"

    def detect(self, frame: FrameContext) -> list[Detection]:
        return [
            Detection(
                frame_number=frame.frame_number,
                label="person",
                confidence=0.9,
                x1=5,
                y1=5,
                x2=30,
                y2=30,
                source=self.name,
            )
        ]


def _track(track_id: int, x1: int, x2: int, *, hits: int = 3) -> Track:
    return Track(
        track_id=track_id,
        frame_number=3,
        timestamp_s=0.3,
        label="person",
        confidence=0.9,
        x1=x1,
        y1=5,
        x2=x2,
        y2=30,
        source="iou",
        hits=hits,
        first_frame_number=1,
        first_timestamp_s=0.1,
    )


def test_cosine_distance_to_centroid() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = cosine_distance_to_centroid(embeddings, np.asarray([1.0, 0.0]))

    assert np.allclose(scores, [0.0, 1.0])


def test_reference_round_trip(tmp_path: Path) -> None:
    reference = fit_normal_reference(
        np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]], dtype=np.float32),
        embedder_name="fake",
        quantile=0.95,
        metadata={"fixture": True},
    )
    path = reference.save(tmp_path / "normal.npz")
    loaded = AnomalyReference.load(path)

    assert loaded.embedder_name == "fake"
    assert loaded.sample_count == 3
    assert loaded.quantile == 0.95
    assert loaded.metadata == {"fixture": True}
    assert np.allclose(loaded.centroid, reference.centroid)
    assert loaded.threshold == reference.threshold


def test_track_crop_analyzer_scores_normal_and_unusual_crops() -> None:
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:, 40:] = 255
    frame = FrameContext(frame_number=3, timestamp_s=0.3, image=image)
    reference = AnomalyReference(
        embedder_name=MeanBucketEmbedder.name,
        centroid=np.asarray([1.0, 0.0], dtype=np.float32),
        threshold=0.2,
        sample_count=10,
        quantile=0.95,
    )
    analyzer = TrackCropAnomalyAnalyzer(
        MeanBucketEmbedder(),
        reference,
        min_track_hits=1,
        pad_px=0,
    )

    observations = analyzer.update(frame, [_track(1, 5, 30), _track(2, 45, 70)])

    assert len(observations) == 2
    assert observations[0].is_anomaly is False
    assert observations[0].score == 0.0
    assert observations[1].is_anomaly is True
    assert observations[1].score == 1.0


def test_reference_builder_records_source_provenance(tmp_path: Path) -> None:
    crops = tmp_path / "normal"
    crops.mkdir()
    for index, value in enumerate((10, 20, 30)):
        image = np.full((24, 24, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(crops / f"crop-{index}.jpg"), image)

    output = tmp_path / "normal-reference.npz"
    reference = build_reference_from_directory(
        crops,
        output,
        embedder=MeanBucketEmbedder(),
        quantile=0.95,
        batch_size=2,
    )

    assert output.exists()
    assert reference.sample_count == 3
    assert reference.metadata["source_image_count"] == 3
    assert len(reference.metadata["source_sha256"]) == 64


def test_anomaly_evidence_becomes_event_only_after_persistence(tmp_path: Path) -> None:
    reference = AnomalyReference(
        embedder_name=MeanBucketEmbedder.name,
        centroid=np.asarray([1.0, 0.0], dtype=np.float32),
        threshold=0.2,
        sample_count=10,
        quantile=0.95,
    )
    analyzer = TrackCropAnomalyAnalyzer(
        MeanBucketEmbedder(),
        reference,
        min_track_hits=1,
        pad_px=0,
    )
    pipeline = PipelineSentinel(
        FixedDetector(),
        anomaly_analyzer=analyzer,
        anomaly_event_detector=ConsecutiveAnomalyEventDetector(min_consecutive=2),
    )
    frames = [
        FrameContext(
            frame_number=index,
            timestamp_s=index / 10.0,
            image=np.full((48, 64, 3), 255, dtype=np.uint8),
        )
        for index in range(3)
    ]

    artifacts = pipeline.run_frames(frames, tmp_path / "run", render_fps=10.0)
    anomalies = pd.read_csv(artifacts.anomalies_csv)
    events = pd.read_csv(artifacts.events_csv)
    alerts = pd.read_csv(artifacts.alerts_csv)

    assert len(anomalies) == 3
    assert anomalies["is_anomaly"].all()
    assert len(events) == 1
    assert events.iloc[0]["event_type"] == "visual_anomaly"
    assert events.iloc[0]["frame_number"] == 1
    assert len(alerts) == 1
    assert alerts.iloc[0]["alert_type"] == "visual_anomaly"
