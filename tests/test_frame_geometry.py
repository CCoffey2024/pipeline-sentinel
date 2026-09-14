from pathlib import Path

import numpy as np

from pipeline_sentinel.frame_geometry import FrameGeometryNormalizer
from pipeline_sentinel.types import FrameContext


def _frame(image: np.ndarray, number: int = 0) -> FrameContext:
    return FrameContext(
        frame_number=number,
        timestamp_s=float(number),
        image=image,
        source_path=Path(f"frame_{number:04d}.jpg"),
    )


def test_matching_frame_is_not_copied() -> None:
    frame = _frame(np.zeros((48, 64, 3), dtype=np.uint8))
    normalizer = FrameGeometryNormalizer.from_frame(frame)

    result = normalizer.normalize(frame)

    assert result is frame
    assert normalizer.provenance()["normalized_frames"] == 0


def test_landscape_frame_is_letterboxed_without_distortion() -> None:
    normalizer = FrameGeometryNormalizer(width=60, height=100, fill_value=114)
    source = np.full((30, 60, 3), 255, dtype=np.uint8)

    result = normalizer.normalize(_frame(source))

    assert result.image.shape == (100, 60, 3)
    assert np.all(result.image[:35] == 114)
    assert np.all(result.image[35:65] == 255)
    assert np.all(result.image[65:] == 114)
    assert normalizer.provenance() == {
        "policy": "letterbox_to_first_frame",
        "canonical_width": 60,
        "canonical_height": 100,
        "fill_value": 114,
        "total_frames": 1,
        "normalized_frames": 1,
        "source_dimensions": {"60x30": 1},
    }


def test_portrait_frame_is_pillarboxed_without_distortion() -> None:
    normalizer = FrameGeometryNormalizer(width=100, height=60, fill_value=0)
    source = np.full((60, 30, 3), 255, dtype=np.uint8)

    result = normalizer.normalize(_frame(source))

    assert result.image.shape == (60, 100, 3)
    assert np.all(result.image[:, :35] == 0)
    assert np.all(result.image[:, 35:65] == 255)
    assert np.all(result.image[:, 65:] == 0)
