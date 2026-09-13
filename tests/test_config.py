from pathlib import Path

import pytest

from pipeline_sentinel.config import ConfigError, load_production_config


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
