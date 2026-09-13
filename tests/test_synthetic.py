from pathlib import Path

import cv2

from pipeline_sentinel.synthetic import generate_demo_video


def test_synthetic_generator_is_reproducible_and_labeled(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    ground_truth = tmp_path / "demo_gt.csv"
    annotations = generate_demo_video(video, ground_truth, frame_count=120, size=(320, 180))

    assert video.exists() and video.stat().st_size > 0
    assert ground_truth.exists() and ground_truth.stat().st_size > 0
    assert {"vehicle", "person", "aerial_object", "smoke"}.issubset(
        set(annotations["label"])
    )
    assert {
        "normal_maintenance",
        "intrusion_vehicle",
        "dismount_loiter",
        "small_uas_like",
    }.issubset(set(annotations["scenario_role"]))

    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 120
    cap.release()


def test_synthetic_generator_same_seed_same_annotations(tmp_path: Path) -> None:
    first = generate_demo_video(
        tmp_path / "first.mp4",
        tmp_path / "first.csv",
        frame_count=40,
        size=(160, 96),
        seed=11,
    )
    second = generate_demo_video(
        tmp_path / "second.mp4",
        tmp_path / "second.csv",
        frame_count=40,
        size=(160, 96),
        seed=11,
    )

    assert first.to_dict("records") == second.to_dict("records")
