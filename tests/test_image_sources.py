from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pipeline_sentinel.image_sources import (
    ImageFolderSequence,
    UavdtDataset,
    inspect_local_source,
    open_local_sequence,
)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_generic_image_folder_streams_without_copying(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    _write_image(frames / "frame1.jpg", 10)
    _write_image(frames / "frame2.jpg", 20)
    _write_image(frames / "frame10.jpg", 30)

    sequence = ImageFolderSequence(frames, "frames", frames)
    emitted = list(
        sequence.iter_frames(
            frame_step=2,
            render_fps=10.0,
            sensor_id="EO_LOCAL_01",
            modality="EO",
        )
    )

    assert sequence.frame_count == 3
    assert [frame.source_path.name for frame in emitted] == ["frame1.jpg", "frame10.jpg"]
    assert [frame.frame_number for frame in emitted] == [0, 2]
    assert [frame.timestamp_s for frame in emitted] == [0.0, 0.2]
    assert all(frame.sensor_id == "EO_LOCAL_01" for frame in emitted)
    assert all(frame.source_path.parent == frames.resolve() for frame in emitted)


def test_uavdt_discovers_original_benchmark_layout(tmp_path: Path) -> None:
    root = tmp_path / "UAVDT"
    sequence_dir = root / "UAV-benchmark-M" / "M0203"
    _write_image(sequence_dir / "img000001.jpg", 10)
    _write_image(sequence_dir / "img000002.jpg", 20)
    gt_dir = root / "UAV-benchmark-MOTD_v1.0" / "GT"
    gt_dir.mkdir(parents=True)
    (gt_dir / "M0203_gt.txt").write_text("", encoding="utf-8")

    dataset = UavdtDataset(root)
    assert dataset.sequence_ids() == ["M0203"]
    sequence = dataset.sequence("M0203")
    assert sequence.frame_count == 2
    assert sequence.annotation_candidates == [gt_dir.resolve() / "M0203_gt.txt"]

    emitted = list(
        sequence.iter_frames(
            max_frames=1,
            render_fps=30.0,
            sensor_id="UAVDT_01",
            modality="EO",
        )
    )
    assert emitted[0].frame_number == 1
    assert emitted[0].source_path.name == "img000001.jpg"


def test_inspection_reads_metadata_in_place(tmp_path: Path) -> None:
    frames = tmp_path / "sequence"
    _write_image(frames / "0001.jpg", 10)
    _write_image(frames / "0002.jpg", 20)

    info = inspect_local_source("image_folder", frames)
    assert info["imagery_access"] == "read_in_place"
    assert info["imagery_copied"] is False
    assert info["sequence_count"] == 1
    assert info["sequences"][0]["frame_count"] == 2

    opened = open_local_sequence("image_folder", frames)
    assert opened.source_root == frames.resolve()
