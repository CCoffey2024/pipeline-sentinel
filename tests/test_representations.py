from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_sentinel.pipeline import PipelineSentinel
from pipeline_sentinel.representations import TrackCropRepresentationAnalyzer
from pipeline_sentinel.types import Detection, FrameContext, Track


class MeanColorEmbedder:
    name = "fake:mean-color"

    def encode(self, images):
        return np.asarray(
            [image.mean(axis=(0, 1)) + np.asarray([1.0, 0.0, 0.0]) for image in images],
            dtype=np.float32,
        )


class FixedDetector:
    name = "fixed"

    def detect(self, frame: FrameContext) -> list[Detection]:
        return [
            Detection(
                frame_number=frame.frame_number,
                label="person",
                confidence=0.9,
                x1=4,
                y1=4,
                x2=28,
                y2=28,
                source=self.name,
            )
        ]


def _track(*, track_id: int = 1, hits: int = 3, label: str = "person") -> Track:
    return Track(
        track_id=track_id,
        frame_number=hits,
        timestamp_s=hits / 10.0,
        label=label,
        confidence=0.9,
        x1=4,
        y1=4,
        x2=28,
        y2=28,
        source="iou",
        hits=hits,
        first_frame_number=0,
        first_timestamp_s=0.0,
    )


def test_representation_analyzer_samples_mature_tracks_periodically() -> None:
    analyzer = TrackCropRepresentationAnalyzer(
        MeanColorEmbedder(),
        labels={"person"},
        min_track_hits=3,
        sample_every_n_hits=2,
        pad_px=0,
    )
    frame = FrameContext(
        frame_number=3,
        timestamp_s=0.3,
        image=np.full((32, 32, 3), (10, 20, 30), dtype=np.uint8),
    )

    assert analyzer.update(frame, [_track(hits=2)]) == []
    first = analyzer.update(frame, [_track(hits=3)])
    assert len(first) == 1
    assert np.isclose(first[0].embedding_norm, 1.0)
    assert first[0].previous_cosine_similarity is None
    assert analyzer.update(frame, [_track(hits=3)]) == []
    assert analyzer.update(frame, [_track(hits=4)]) == []
    second = analyzer.update(frame, [_track(hits=5)])
    assert len(second) == 1
    assert np.isclose(second[0].previous_cosine_similarity, 1.0)
    assert np.isclose(second[0].representation_change, 0.0, atol=1e-6)
    assert analyzer.provenance() == {
        "embedder": "fake:mean-color",
        "labels": ["person"],
        "min_track_hits": 3,
        "sample_every_n_hits": 2,
        "max_per_frame": 16,
        "pad_px": 0,
        "min_crop_size": 12,
        "embedding_normalization": "l2",
        "crop_resize": "aspect_preserving_letterbox",
    }


def test_representation_analyzer_bounds_each_frame_by_confidence() -> None:
    analyzer = TrackCropRepresentationAnalyzer(
        MeanColorEmbedder(),
        min_track_hits=1,
        sample_every_n_hits=1,
        max_per_frame=2,
        pad_px=0,
    )
    frame = FrameContext(
        frame_number=1,
        timestamp_s=0.1,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
    )
    tracks = [_track(track_id=index, hits=1) for index in range(1, 5)]

    observations = analyzer.update(frame, tracks)

    assert [observation.track_id for observation in observations] == [1, 2]


def test_pipeline_streams_representation_index_and_vectors(tmp_path: Path) -> None:
    analyzer = TrackCropRepresentationAnalyzer(
        MeanColorEmbedder(),
        min_track_hits=1,
        sample_every_n_hits=1,
        pad_px=0,
    )
    pipeline = PipelineSentinel(
        FixedDetector(),
        representation_analyzer=analyzer,
    )
    frames = [
        FrameContext(
            frame_number=index,
            timestamp_s=index / 10.0,
            image=np.full((32, 32, 3), index * 10, dtype=np.uint8),
            source_path=Path(f"frame-{index}.jpg"),
        )
        for index in range(3)
    ]

    artifacts = pipeline.run_frames(frames, tmp_path / "run", render_fps=10.0)

    index = pd.read_csv(artifacts.representations_csv)
    vectors = np.fromfile(artifacts.representation_embeddings_f32, dtype="<f4").reshape(3, 3)
    assert index["embedding_row"].tolist() == [0, 1, 2]
    assert index["track_id"].tolist() == [1, 1, 1]
    assert index["embedding_dim"].tolist() == [3, 3, 3]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert artifacts.representation_manifest_json.is_file()
