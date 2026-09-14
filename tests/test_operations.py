import json
from pathlib import Path

import numpy as np
import pytest

from pipeline_sentinel.operations import run_configured_video
from pipeline_sentinel.synthetic import generate_demo_video
from pipeline_sentinel.types import Detection, FrameContext


class FakeYoloDetector:
    name = "fake_yolo"

    def __init__(self, **_kwargs: object) -> None:
        pass

    def detect(self, frame: FrameContext) -> list[Detection]:
        return [
            Detection(
                frame_number=frame.frame_number,
                label="person",
                confidence=0.9,
                x1=5,
                y1=5,
                x2=20,
                y2=30,
                source=self.name,
            )
        ]


class FailingYoloDetector(FakeYoloDetector):
    name = "failing_yolo"

    def detect(self, frame: FrameContext) -> list[Detection]:
        raise RuntimeError(f"synthetic inference failure at frame {frame.frame_number}")


class FakeDinoV2Embedder:
    name = "fake_dinov2"

    def __init__(self, **_kwargs: object) -> None:
        pass

    def encode(self, images) -> np.ndarray:
        return np.ones((len(images), 4), dtype=np.float32)


def _config(path: Path, *, representations_enabled: bool = False) -> Path:
    path.write_text(
        """
detector:
  backend: yolo
  model: fake.pt
  confidence: 0.25
  nms_iou: 0.70
  imgsz: 960
tracker:
  backend: iou
  iou_threshold: 0.30
  max_missed_updates: 2
representation:
  enabled: REPRESENTATIONS_ENABLED
  min_track_hits: 1
  sample_every_n_hits: 1
  max_per_frame: 4
  pad_px: 0
  min_crop_size: 4
events:
  dwell:
    enabled: false
alerts:
  minimum_severity: warning
runtime:
  sensor_id: EO_TEST
  modality: EO
""".replace("REPRESENTATIONS_ENABLED", str(representations_enabled).lower()).strip(),
        encoding="utf-8",
    )
    return path


def test_configured_run_writes_operational_artifacts(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "input.mp4"
    generate_demo_video(video, tmp_path / "gt.csv", frame_count=3, size=(64, 64))
    config_path = _config(tmp_path / "production.yaml")
    output = tmp_path / "run"

    monkeypatch.setattr("pipeline_sentinel.operations.YoloDetector", FakeYoloDetector)
    artifacts = run_configured_video(video, config_path, output_dir=output)

    assert artifacts.output_dir == output.resolve()
    assert artifacts.pipeline.detections_csv.exists()
    assert artifacts.pipeline.tracks_csv.exists()
    assert artifacts.effective_config_json.exists()
    assert artifacts.run_log_jsonl.exists()
    assert artifacts.run_status_json.exists()

    status = json.loads(artifacts.run_status_json.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["run_id"] == artifacts.run_id
    assert status["error"] is None

    manifest = json.loads(artifacts.pipeline.run_manifest.read_text(encoding="utf-8"))
    assert manifest["metadata"]["run_id"] == artifacts.run_id
    assert len(manifest["metadata"]["config_sha256"]) == 64

    log_events = [
        json.loads(line)["event"]
        for line in artifacts.run_log_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert log_events == ["run_started", "run_completed"]


def test_configured_run_executes_dinov2_representation_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "input.mp4"
    generate_demo_video(video, tmp_path / "gt.csv", frame_count=3, size=(64, 64))
    config_path = _config(tmp_path / "production.yaml", representations_enabled=True)

    monkeypatch.setattr("pipeline_sentinel.operations.YoloDetector", FakeYoloDetector)
    monkeypatch.setattr("pipeline_sentinel.operations.DinoV2Embedder", FakeDinoV2Embedder)
    artifacts = run_configured_video(video, config_path, output_dir=tmp_path / "run")

    vectors = np.fromfile(artifacts.pipeline.representation_embeddings_f32, dtype="<f4")
    manifest = json.loads(
        artifacts.pipeline.representation_manifest_json.read_text(encoding="utf-8")
    )
    assert vectors.shape == (12,)
    assert manifest["enabled"] is True
    assert manifest["observations"] == 3
    assert manifest["embedding_dimension"] == 4
    assert manifest["configuration"]["embedder"] == "fake_dinov2"


def test_failed_run_preserves_failure_status(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "input.mp4"
    generate_demo_video(video, tmp_path / "gt.csv", frame_count=2, size=(64, 64))
    config_path = _config(tmp_path / "production.yaml")
    output = tmp_path / "failed-run"

    monkeypatch.setattr("pipeline_sentinel.operations.YoloDetector", FailingYoloDetector)
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        run_configured_video(video, config_path, output_dir=output)

    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error"]["type"] == "RuntimeError"
    assert "synthetic inference failure" in status["error"]["message"]

    log_events = [
        json.loads(line)["event"]
        for line in (output / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert log_events == ["run_started", "run_failed"]
