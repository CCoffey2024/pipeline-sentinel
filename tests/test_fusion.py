import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline_sentinel.fusion import (
    FusionConfig,
    SensorEventEvidence,
    TemporalConsensusFuser,
    box_iou,
    fuse_run_directories,
    load_fusion_config,
)


def evidence(
    sensor_id: str,
    modality: str,
    *,
    event_id: str,
    timestamp_s: float,
    label: str | None = "person",
    bbox: tuple[float, float, float, float] | None = (10, 10, 30, 40),
    severity: str = "warning",
) -> SensorEventEvidence:
    return SensorEventEvidence(
        run_dir=Path(f"/{sensor_id}"),
        run_id=f"run-{sensor_id}",
        sensor_id=sensor_id,
        modality=modality,
        event_id=event_id,
        frame_number=10,
        timestamp_s=timestamp_s,
        event_type="visual_anomaly",
        severity=severity,
        source="fixture",
        track_id=1,
        label=label,
        confidence=0.8,
        bbox=bbox,
    )


def test_bundled_fusion_config_is_valid() -> None:
    config, provenance = load_fusion_config()

    assert config.strategy == "temporal_consensus"
    assert config.min_sensors == 2
    assert config.require_matching_label is True
    assert config.spatial_iou_threshold is None
    assert provenance.path.name == "fusion.yaml"
    assert len(provenance.sha256) == 64


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_temporal_consensus_fuses_distinct_sensors_once() -> None:
    config = FusionConfig(max_time_delta_s=0.5, min_sensors=2)
    fused, contributors = TemporalConsensusFuser(config).fuse(
        [
            evidence("EO_CAM", "EO", event_id="eo-1", timestamp_s=10.0),
            evidence("IR_CAM", "IR", event_id="ir-1", timestamp_s=10.2),
        ]
    )

    assert len(fused) == 1
    assert fused[0].event_type == "visual_anomaly"
    assert fused[0].source == "temporal_consensus"
    assert fused[0].track_id is None
    assert fused[0].metadata["support_count"] == 2
    assert set(fused[0].metadata["sensor_ids"]) == {"EO_CAM", "IR_CAM"}
    assert len(contributors) == 2
    assert {row["event_id"] for row in contributors} == {"eo-1", "ir-1"}


def test_temporal_consensus_requires_time_and_label_compatibility() -> None:
    fuser = TemporalConsensusFuser(FusionConfig(max_time_delta_s=0.25))

    late, _ = fuser.fuse(
        [
            evidence("EO_CAM", "EO", event_id="eo", timestamp_s=1.0),
            evidence("IR_CAM", "IR", event_id="ir", timestamp_s=1.5),
        ]
    )
    mismatch, _ = fuser.fuse(
        [
            evidence("EO_CAM", "EO", event_id="eo", timestamp_s=1.0, label="person"),
            evidence("IR_CAM", "IR", event_id="ir", timestamp_s=1.1, label="vehicle"),
        ]
    )

    assert late == []
    assert mismatch == []


def test_spatial_gate_is_explicit() -> None:
    far_box = (100, 100, 120, 130)
    without_gate, _ = TemporalConsensusFuser(
        FusionConfig(max_time_delta_s=0.5, spatial_iou_threshold=None)
    ).fuse(
        [
            evidence("EO_CAM", "EO", event_id="eo", timestamp_s=2.0),
            evidence("IR_CAM", "IR", event_id="ir", timestamp_s=2.1, bbox=far_box),
        ]
    )
    with_gate, _ = TemporalConsensusFuser(
        FusionConfig(max_time_delta_s=0.5, spatial_iou_threshold=0.3)
    ).fuse(
        [
            evidence("EO_CAM", "EO", event_id="eo", timestamp_s=2.0),
            evidence("IR_CAM", "IR", event_id="ir", timestamp_s=2.1, bbox=far_box),
        ]
    )

    assert len(without_gate) == 1
    assert with_gate == []


def _write_run(
    root: Path,
    *,
    sensor_id: str,
    modality: str,
    event_id: str,
    timestamp_s: float,
    severity: str = "warning",
) -> Path:
    root.mkdir(parents=True)
    manifest = {
        "pipeline_sentinel_version": "0.9.0",
        "sensor_id": sensor_id,
        "modality": modality,
        "metadata": {"run_id": f"run-{sensor_id}"},
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "event_id": event_id,
                "frame_number": 10,
                "timestamp_s": timestamp_s,
                "event_type": "visual_anomaly",
                "severity": severity,
                "track_id": 1,
                "label": "person",
                "confidence": 0.85,
                "source": "fixture",
                "message": "fixture",
                "metadata": "{}",
            }
        ]
    ).to_csv(root / "events.csv", index=False)
    pd.DataFrame(
        [
            {
                "frame_number": 10,
                "timestamp_s": timestamp_s,
                "track_id": 1,
                "label": "person",
                "confidence": 0.85,
                "hits": 5,
                "duration_s": 1.0,
                "first_frame_number": 5,
                "first_timestamp_s": 0.5,
                "scenario_role": None,
                "tracker": "fixture",
                "sensor_id": sensor_id,
                "modality": modality,
                "source_path": "fixture.mp4",
                "x1": 10,
                "y1": 10,
                "x2": 30,
                "y2": 40,
            }
        ]
    ).to_csv(root / "tracks.csv", index=False)
    return root


def test_fuse_run_directories_writes_auditable_artifacts(tmp_path: Path) -> None:
    eo = _write_run(
        tmp_path / "eo",
        sensor_id="EO_CAM",
        modality="EO",
        event_id="eo-event",
        timestamp_s=5.0,
    )
    ir = _write_run(
        tmp_path / "ir",
        sensor_id="IR_CAM",
        modality="IR",
        event_id="ir-event",
        timestamp_s=5.2,
        severity="critical",
    )
    config_path = tmp_path / "fusion.yaml"
    config_path.write_text(
        """
strategy: temporal_consensus
max_time_delta_s: 0.5
min_sensors: 2
require_matching_label: true
spatial_iou_threshold: 0.3
alerts:
  minimum_severity: warning
""".strip(),
        encoding="utf-8",
    )

    artifacts = fuse_run_directories(
        [eo, ir],
        config_path=config_path,
        output_dir=tmp_path / "fused",
    )

    assert artifacts.events_csv.exists()
    assert artifacts.alerts_csv.exists()
    assert artifacts.contributors_csv.exists()
    assert artifacts.manifest_json.exists()
    assert artifacts.effective_config_json.exists()

    events = pd.read_csv(artifacts.events_csv)
    alerts = pd.read_csv(artifacts.alerts_csv)
    contributors = pd.read_csv(artifacts.contributors_csv)
    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))

    assert len(events) == 1
    assert events.iloc[0]["severity"] == "critical"
    assert len(alerts) == 1
    assert len(contributors) == 2
    assert set(contributors["sensor_id"]) == {"EO_CAM", "IR_CAM"}
    assert manifest["distinct_sensors"] == 2
    assert manifest["fused_events"] == 1
    assert manifest["fused_alerts"] == 1
    assert manifest["assumptions"]["fusion_level"] == "semantic_event_late_fusion"
    assert manifest["assumptions"]["raw_pixels_fused"] is False
    assert manifest["assumptions"]["pixel_registration_required"] is True


def test_duplicate_sensor_id_is_rejected(tmp_path: Path) -> None:
    first = _write_run(
        tmp_path / "first",
        sensor_id="EO_CAM",
        modality="EO",
        event_id="one",
        timestamp_s=1.0,
    )
    second = _write_run(
        tmp_path / "second",
        sensor_id="EO_CAM",
        modality="EO",
        event_id="two",
        timestamp_s=1.1,
    )

    with pytest.raises(ValueError, match="duplicate sensor_id"):
        fuse_run_directories([first, second], output_dir=tmp_path / "out")
