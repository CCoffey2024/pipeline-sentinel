from pathlib import Path

import pytest

from pipeline_sentinel.config import ConfigError, load_production_config


def test_bundled_production_config_is_valid() -> None:
    config, provenance = load_production_config()

    assert config.detector.backend == "yolo"
    assert config.detector.imgsz == 960
    assert config.tracker.backend == "iou"
    assert config.representation.enabled is True
    assert config.representation.backend == "dinov2"
    assert config.representation.sample_every_n_hits == 15
    assert config.anomaly.enabled is False
    assert config.anomaly.backend == "dinov2"
    assert config.events.dwell.enabled is False
    assert provenance.path.name == "production.yaml"
    assert len(provenance.sha256) == 64


def test_load_production_config_and_hash(tmp_path: Path) -> None:
    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        """
detector:
  backend: yolo
  model: yolo26n.pt
  confidence: 0.30
  nms_iou: 0.65
  imgsz: 960
tracker:
  backend: iou
  iou_threshold: 0.25
  max_missed_updates: 3
representation:
  enabled: true
  backend: dinov2
  model: dinov2_vits14
  device: cpu
  labels: [person, car]
  min_track_hits: 4
  sample_every_n_hits: 10
  max_per_frame: 8
  pad_px: 8
  min_crop_size: 16
anomaly:
  enabled: true
  backend: dinov2
  reference_path: refs/normal.npz
  model: dinov2_vits14
  device: cpu
  labels: [person]
  min_track_hits: 4
  pad_px: 8
  min_crop_size: 16
  event_min_consecutive: 2
  event_severity: warning
events:
  dwell:
    enabled: true
    min_hits: 20
    max_displacement_px: 30
    labels: [person]
alerts:
  minimum_severity: warning
runtime:
  sensor_id: EO_TEST
  modality: EO
""".strip(),
        encoding="utf-8",
    )

    config, provenance = load_production_config(config_path)

    assert config.detector.model == "yolo26n.pt"
    assert config.detector.imgsz == 960
    assert config.tracker.max_missed_updates == 3
    assert config.representation.enabled is True
    assert config.representation.labels == ("person", "car")
    assert config.representation.sample_every_n_hits == 10
    assert config.representation.max_per_frame == 8
    assert config.anomaly.enabled is True
    assert config.anomaly.reference_path == "refs/normal.npz"
    assert config.anomaly.labels == ("person",)
    assert config.anomaly.event_min_consecutive == 2
    assert config.events.dwell.enabled is True
    assert config.events.dwell.labels == ("person",)
    assert config.runtime.sensor_id == "EO_TEST"
    assert provenance.path == config_path.resolve()
    assert len(provenance.sha256) == 64


def test_config_rejects_unknown_option(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("detector:\n  typo_confidence: 0.25\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Unknown detector option"):
        load_production_config(config_path)


def test_config_rejects_invalid_probability(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("detector:\n  confidence: 1.5\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="between 0 and 1"):
        load_production_config(config_path)


def test_enabled_anomaly_requires_reference_path(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("anomaly:\n  enabled: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="reference_path is required"):
        load_production_config(config_path)
