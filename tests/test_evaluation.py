import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline_sentinel.evaluation import (
    SHARED_ONTOLOGY_NAME,
    box_iou,
    evaluate_visdrone_run,
    evaluate_visdrone_tables,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame_number": 1,
                "label": "person",
                "confidence": 0.90,
                "x1": 0,
                "y1": 0,
                "x2": 10,
                "y2": 10,
            },
            {
                "frame_number": 1,
                "label": "person",
                "confidence": 0.70,
                "x1": 0,
                "y1": 0,
                "x2": 10,
                "y2": 10,
            },
            {
                "frame_number": 1,
                "label": "motorcycle",
                "confidence": 0.80,
                "x1": 20,
                "y1": 20,
                "x2": 30,
                "y2": 30,
            },
            {
                "frame_number": 2,
                "label": "car",
                "confidence": 0.60,
                "x1": 40,
                "y1": 40,
                "x2": 50,
                "y2": 50,
            },
            {
                "frame_number": 1,
                "label": "sports ball",
                "confidence": 0.50,
                "x1": 60,
                "y1": 60,
                "x2": 70,
                "y2": 70,
            },
        ]
    )


def _ground_truth() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame_index": 1,
                "target_id": 101,
                "label": "pedestrian",
                "ignored": False,
                "x1": 0,
                "y1": 0,
                "x2": 10,
                "y2": 10,
            },
            {
                "frame_index": 1,
                "target_id": 102,
                "label": "motor",
                "ignored": False,
                "x1": 20,
                "y1": 20,
                "x2": 30,
                "y2": 30,
            },
            {
                "frame_index": 2,
                "target_id": 103,
                "label": "bicycle",
                "ignored": False,
                "x1": 5,
                "y1": 5,
                "x2": 15,
                "y2": 15,
            },
            {
                "frame_index": 2,
                "target_id": 104,
                "label": "van",
                "ignored": False,
                "x1": 70,
                "y1": 70,
                "x2": 80,
                "y2": 80,
            },
            {
                "frame_index": 2,
                "target_id": 105,
                "label": "pedestrian",
                "ignored": True,
                "x1": 80,
                "y1": 80,
                "x2": 90,
                "y2": 90,
            },
        ]
    )


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (10, 10, 20, 20)) == pytest.approx(0.0)
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_shared_ontology_iou_evaluation() -> None:
    result = evaluate_visdrone_tables(_predictions(), _ground_truth(), iou_threshold=0.50)

    person = result.class_metrics.set_index("shared_class").loc["person"]
    motor = result.class_metrics.set_index("shared_class").loc["motor"]
    bicycle = result.class_metrics.set_index("shared_class").loc["bicycle"]
    car = result.class_metrics.set_index("shared_class").loc["car"]

    assert person["true_positive"] == 1
    assert person["false_positive"] == 1
    assert person["false_negative"] == 0
    assert person["precision"] == pytest.approx(0.5)
    assert person["recall"] == pytest.approx(1.0)

    assert motor["true_positive"] == 1
    assert motor["false_positive"] == 0
    assert motor["false_negative"] == 0

    assert bicycle["predictions"] == 0
    assert bicycle["ground_truth"] == 1
    assert bicycle["false_negative"] == 1
    assert bicycle["recall"] == pytest.approx(0.0)

    assert car["predictions"] == 1
    assert car["ground_truth"] == 0
    assert car["false_positive"] == 1
    assert car["precision"] == pytest.approx(0.0)

    assert result.summary["ontology"]["name"] == SHARED_ONTOLOGY_NAME
    assert result.summary["overall"]["true_positive"] == 2
    assert result.summary["overall"]["false_positive"] == 2
    assert result.summary["overall"]["false_negative"] == 1
    assert result.summary["overall"]["precision"] == pytest.approx(0.5)
    assert result.summary["overall"]["recall"] == pytest.approx(2 / 3)
    assert result.summary["excluded_prediction_labels"] == {"sports ball": 1}
    assert result.summary["excluded_ground_truth_labels"] == {"van": 1}
    assert result.summary["counts"]["ground_truth_ignored"] == 1

    statuses = result.matches["status"].value_counts().to_dict()
    assert statuses == {"TP": 2, "FP": 2, "FN": 1}


def test_evaluate_visdrone_run_writes_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _predictions().to_csv(run_dir / "detections.csv", index=False)
    _ground_truth().to_csv(run_dir / "ground_truth.csv", index=False)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "pipeline_sentinel_version": "0.4.0",
                "input_source": "VisDrone2019-VID:val/fixture",
                "detector_backend": "yolo",
                "frames_processed": 2,
                "first_frame_number": 1,
                "last_frame_number": 2,
                "metadata": {
                    "split": "val",
                    "sequence_id": "fixture",
                    "ground_truth_scope": "processed_frames",
                    "detector_model": "fake-yolo.pt",
                    "detector_confidence": 0.25,
                    "detector_nms_iou": 0.70,
                    "detector_imgsz": 640,
                    "detector_device": "cpu",
                    "detector_class_ids": None,
                },
            }
        ),
        encoding="utf-8",
    )

    result, artifacts = evaluate_visdrone_run(run_dir, iou_threshold=0.50)

    assert artifacts.summary_json.exists()
    assert artifacts.class_metrics_csv.exists()
    assert artifacts.matches_csv.exists()
    assert artifacts.excluded_predictions_csv.exists()
    assert artifacts.excluded_ground_truth_csv.exists()

    summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    assert summary["run"]["sequence_id"] == "fixture"
    assert summary["run"]["model"] == "fake-yolo.pt"
    assert summary["overall"]["true_positive"] == 2
    assert result.summary["iou_threshold"] == pytest.approx(0.50)


def test_evaluate_visdrone_run_rejects_unscoped_ground_truth(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _predictions().to_csv(run_dir / "detections.csv", index=False)
    _ground_truth().to_csv(run_dir / "ground_truth.csv", index=False)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"metadata": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ground_truth_scope='processed_frames'"):
        evaluate_visdrone_run(run_dir)
