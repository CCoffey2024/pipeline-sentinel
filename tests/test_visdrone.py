from pathlib import Path

import cv2
import numpy as np

from pipeline_sentinel.visdrone import VisDroneDataset


def _build_visdrone_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "VisDrone2019-VID-val"
    frames_dir = root / "sequences" / "uav_test_sequence"
    annotations_dir = root / "annotations"
    frames_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    for frame_index in range(1, 5):
        image = np.full((48, 64, 3), frame_index * 10, dtype=np.uint8)
        assert cv2.imwrite(str(frames_dir / f"{frame_index:07d}.jpg"), image)

    (annotations_dir / "uav_test_sequence.txt").write_text(
        "\n".join(
            [
                "1,101,10,12,20,15,1,4,0,0",
                "1,102,1,1,8,8,0,0,0,0",
                "2,103,20,5,10,20,1,1,0,1",
                "4,104,30,10,12,10,1,6,1,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_visdrone_dataset_discovers_sequences_and_split(tmp_path: Path) -> None:
    root = _build_visdrone_fixture(tmp_path)
    dataset = VisDroneDataset(root)

    assert dataset.split_name == "val"
    assert dataset.sequence_ids() == ["uav_test_sequence"]
    assert dataset.summary()["sequence_count"] == 1
    assert dataset.summary()["annotation_file_count"] == 1


def test_visdrone_dataset_resolves_nested_archive_root(tmp_path: Path) -> None:
    outer = tmp_path / "downloaded-val"
    nested = _build_visdrone_fixture(outer)

    dataset = VisDroneDataset(outer)

    assert dataset.requested_root == outer.resolve()
    assert dataset.root == nested.resolve()
    assert dataset.split_name == "val"
    assert dataset.sequence_ids() == ["uav_test_sequence"]


def test_visdrone_dataset_accepts_sequences_directory(tmp_path: Path) -> None:
    root = _build_visdrone_fixture(tmp_path)
    dataset = VisDroneDataset(root / "sequences")

    assert dataset.root == root.resolve()
    assert dataset.sequence_ids() == ["uav_test_sequence"]


def test_visdrone_annotations_are_normalized_to_xyxy(tmp_path: Path) -> None:
    sequence = VisDroneDataset(_build_visdrone_fixture(tmp_path)).first_sequence()
    ground_truth = sequence.normalized_ground_truth()

    car = ground_truth.loc[ground_truth["target_id"] == 101].iloc[0]
    ignored = ground_truth.loc[ground_truth["target_id"] == 102].iloc[0]

    assert car["label"] == "car"
    assert (int(car["x1"]), int(car["y1"]), int(car["x2"]), int(car["y2"])) == (
        10,
        12,
        30,
        27,
    )
    assert bool(car["ignored"]) is False
    assert bool(ignored["ignored"]) is True

    detections = sequence.detections_for_frame(1)
    assert len(detections) == 1
    assert detections[0].label == "car"
    assert detections[0].xyxy == (10, 12, 30, 27)
    assert detections[0].object_id == 101


def test_visdrone_sequence_yields_runtime_frames_with_sampling(tmp_path: Path) -> None:
    sequence = VisDroneDataset(_build_visdrone_fixture(tmp_path)).first_sequence()
    frames = list(sequence.iter_frames(frame_step=2, max_frames=2, render_fps=10.0))

    assert [frame.frame_number for frame in frames] == [1, 3]
    assert [frame.timestamp_s for frame in frames] == [0.0, 0.2]
    assert all(frame.sensor_id == "VISDRONE:uav_test_sequence" for frame in frames)
    assert all(frame.modality == "EO" for frame in frames)
    assert all(frame.image.shape == (48, 64, 3) for frame in frames)


def test_visdrone_ground_truth_matches_sampled_runtime_frames(tmp_path: Path) -> None:
    sequence = VisDroneDataset(_build_visdrone_fixture(tmp_path)).first_sequence()
    selected = sequence.selected_frame_numbers(frame_step=2, max_frames=2)
    ground_truth = sequence.normalized_ground_truth(frame_numbers=selected)

    assert selected == [1, 3]
    assert set(ground_truth["frame_index"].astype(int)) == {1}
    assert set(ground_truth["target_id"].astype(int)) == {101, 102}
