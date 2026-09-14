from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from pipeline_sentinel.fusion import FusionArtifacts
from pipeline_sentinel.operations import ProductionRunArtifacts
from pipeline_sentinel.operator_jobs import OperatorJobManager, safe_upload_name
from pipeline_sentinel.pipeline import RunArtifacts


def _wait(manager: OperatorJobManager, job_id: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get_job(job_id)
        if job.status in {"completed", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job did not finish: {job_id}")


def _fake_run(video_path: Path, config_path: Path, *, output_dir: Path | None = None):
    assert output_dir is not None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    paths = {
        "annotated_video": output / "annotated_video.mp4",
        "detections_csv": output / "detections.csv",
        "tracks_csv": output / "tracks.csv",
        "anomalies_csv": output / "anomalies.csv",
        "events_csv": output / "events.csv",
        "alerts_csv": output / "alerts.csv",
        "run_manifest": output / "run_manifest.json",
        "effective_config_json": output / "effective_config.json",
        "run_log_jsonl": output / "run_log.jsonl",
        "run_status_json": output / "run_status.json",
    }
    paths["annotated_video"].write_bytes(b"fake-video")
    for name in ("detections_csv", "tracks_csv", "anomalies_csv"):
        paths[name].write_text("frame_number\n", encoding="utf-8")
    paths["events_csv"].write_text(
        "event_id,frame_number,timestamp_s,event_type,severity,track_id,label,confidence,source,message,metadata\n",
        encoding="utf-8",
    )
    paths["alerts_csv"].write_text(
        "alert_id,event_id,frame_number,timestamp_s,alert_type,severity,track_id,label,confidence,source,message,metadata\n",
        encoding="utf-8",
    )
    paths["run_manifest"].write_text(
        json.dumps(
            {
                "sensor_id": config["runtime"]["sensor_id"],
                "modality": config["runtime"]["modality"],
                "frames_processed": 12,
                "detections_emitted": 30,
                "unique_tracks": 4,
                "anomalies_flagged": 1,
                "events_emitted": 2,
                "alerts_emitted": 1,
            }
        ),
        encoding="utf-8",
    )
    paths["effective_config_json"].write_text("{}", encoding="utf-8")
    paths["run_log_jsonl"].write_text("", encoding="utf-8")
    paths["run_status_json"].write_text("{}", encoding="utf-8")

    pipeline = RunArtifacts(
        annotated_video=paths["annotated_video"],
        detections_csv=paths["detections_csv"],
        tracks_csv=paths["tracks_csv"],
        anomalies_csv=paths["anomalies_csv"],
        events_csv=paths["events_csv"],
        alerts_csv=paths["alerts_csv"],
        run_manifest=paths["run_manifest"],
    )
    return ProductionRunArtifacts(
        run_id="pipeline-run-test",
        output_dir=output,
        effective_config_json=paths["effective_config_json"],
        run_log_jsonl=paths["run_log_jsonl"],
        run_status_json=paths["run_status_json"],
        pipeline=pipeline,
    )


def _fake_fusion(run_dirs, *, config_path=None, output_dir=None):
    assert len(run_dirs) == 2
    assert output_dir is not None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = output / "fusion_events.csv"
    alerts = output / "fusion_alerts.csv"
    contributors = output / "fusion_contributors.csv"
    manifest = output / "fusion_manifest.json"
    config = output / "effective_fusion_config.json"
    events.write_text(
        "event_id,frame_number,timestamp_s,event_type,severity,track_id,label,confidence,source,message,metadata\n",
        encoding="utf-8",
    )
    alerts.write_text(
        "alert_id,event_id,frame_number,timestamp_s,alert_type,severity,track_id,label,confidence,source,message,metadata\n",
        encoding="utf-8",
    )
    contributors.write_text("fusion_event_id,sensor_id\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "fusion_id": "fusion-test",
                "distinct_sensors": 2,
                "input_events": 4,
                "fused_events": 2,
                "fused_alerts": 1,
                "config_source": "test-fusion.yaml",
            }
        ),
        encoding="utf-8",
    )
    config.write_text("{}", encoding="utf-8")
    return FusionArtifacts(
        fusion_id="fusion-test",
        output_dir=output,
        events_csv=events,
        alerts_csv=alerts,
        contributors_csv=contributors,
        manifest_json=manifest,
        effective_config_json=config,
    )


def test_safe_upload_name_strips_browser_paths():
    assert safe_upload_name(r"C:\fake\path\eo clip.mp4") == "C_fake_path_eo_clip.mp4"
    assert safe_upload_name("../../ir.mov") == "ir.mov"


def test_operator_run_persists_runtime_identity(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline_sentinel.operator_jobs.run_configured_video", _fake_run)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"not-a-real-video")
    manager = OperatorJobManager(tmp_path / "workspace")
    try:
        submitted = manager.submit_run(video, sensor_id="IR_CAM_07", modality="IR")
        job = _wait(manager, submitted.job_id)
        assert job.status == "completed"
        assert job.pipeline_run_id == "pipeline-run-test"
        assert job.sensor_id == "IR_CAM_07"
        assert job.modality == "IR"
        assert job.summary == {
            "frames": 12,
            "detections": 30,
            "tracks": 4,
            "anomalies": 1,
            "events": 2,
            "alerts": 1,
        }
        assert manager.artifact_path(job.job_id, "run_manifest").is_file()
    finally:
        manager.close()

    reopened = OperatorJobManager(tmp_path / "workspace")
    try:
        restored = reopened.get_job(submitted.job_id)
        assert restored.status == "completed"
        assert restored.pipeline_run_id == "pipeline-run-test"
    finally:
        reopened.close()


def test_operator_fusion_uses_completed_distinct_sensor_runs(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline_sentinel.operator_jobs.run_configured_video", _fake_run)
    monkeypatch.setattr("pipeline_sentinel.operator_jobs.fuse_run_directories", _fake_fusion)
    eo = tmp_path / "eo.mp4"
    ir = tmp_path / "ir.mp4"
    eo.write_bytes(b"eo")
    ir.write_bytes(b"ir")
    manager = OperatorJobManager(tmp_path / "workspace")
    try:
        eo_job = _wait(manager, manager.submit_run(eo, sensor_id="EO_1", modality="EO").job_id)
        ir_job = _wait(manager, manager.submit_run(ir, sensor_id="IR_1", modality="IR").job_id)
        fused = _wait(manager, manager.submit_fusion([eo_job.job_id, ir_job.job_id]).job_id)
        assert fused.status == "completed"
        assert fused.fusion_id == "fusion-test"
        assert fused.summary == {"input_events": 4, "events": 2, "alerts": 1, "sensors": 2}
        assert manager.artifact_path(fused.job_id, "fusion_manifest_json").is_file()
    finally:
        manager.close()
