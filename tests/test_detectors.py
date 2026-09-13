from pathlib import Path

import pandas as pd
import pytest

from pipeline_sentinel.detectors import GroundTruthDetector


def test_ground_truth_detector_emits_normalized_detections(tmp_path: Path) -> None:
    annotations = pd.DataFrame(
        [
            {
                "frame_number": 4,
                "label": "vehicle",
                "object_id": 7,
                "x1": 10,
                "y1": 20,
                "x2": 30,
                "y2": 40,
                "scenario_role": "intrusion_vehicle",
            },
            {
                "frame_number": 5,
                "label": "person",
                "object_id": 8,
                "x1": 50,
                "y1": 60,
                "x2": 70,
                "y2": 90,
                "scenario_role": "dismount_loiter",
            },
        ]
    )
    path = tmp_path / "gt.csv"
    annotations.to_csv(path, index=False)

    detector = GroundTruthDetector(path)
    detections = detector.detect(4)

    assert detector.name == "ground_truth"
    assert len(detections) == 1
    detection = detections[0]
    assert detection.frame_number == 4
    assert detection.label == "vehicle"
    assert detection.xyxy == (10, 20, 30, 40)
    assert detection.confidence == 1.0
    assert detection.source == "ground_truth"
    assert detection.object_id == 7
    assert detection.scenario_role == "intrusion_vehicle"
    assert detector.detect(99) == []


def test_ground_truth_detector_rejects_bad_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"frame_number": 0, "label": "vehicle"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        GroundTruthDetector(path)
