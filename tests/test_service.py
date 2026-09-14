from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from pipeline_sentinel.operations import ProductionRunArtifacts
from pipeline_sentinel.pipeline import RunArtifacts
from pipeline_sentinel.service import create_app


def _fake_run(video_path: Path, config_path: Path, *, output_dir: Path | None = None):
    assert output_dir is not None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    annotated = output / "annotated_video.mp4"
    detections = output / "detections.csv"
    tracks = output / "tracks.csv"
    anomalies = output / "anomalies.csv"
    events = output / "events.csv"
    alerts = output / "alerts.csv"
    manifest = output / "run_manifest.json"
    effective = output / "effective_config.json"
    log = output / "run_log.jsonl"
    status = output / "run_status.json"

    annotated.write_bytes(b"video")
    detections.write_text("frame_number\n", encoding="utf-8")
    tracks.write_text("frame_number\n", encoding="utf-8")
    anomalies.write_text("frame_number\n", encoding="utf-8")
    events.write_text(
        "event_id,frame_number,timestamp_s,event_type,severity,track_id,label,confidence,source,message,metadata\n",
        encoding="utf-8",
    )
    alerts.write_text(
        "alert_id,event_id,frame_number,timestamp_s,alert_type,severity,track_id,label,confidence,source,message,metadata\n"
        "a1,e1,4,0.4,visual_anomaly,warning,7,person,0.8,test,Review this track,{}\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "sensor_id": config["runtime"]["sensor_id"],
                "modality": config["runtime"]["modality"],
                "frames_processed": 5,
                "detections_emitted": 8,
                "unique_tracks": 2,
                "anomalies_flagged": 1,
                "events_emitted": 1,
                "alerts_emitted": 1,
            }
        ),
        encoding="utf-8",
    )
    effective.write_text("{}", encoding="utf-8")
    log.write_text("", encoding="utf-8")
    status.write_text("{}", encoding="utf-8")
    return ProductionRunArtifacts(
        run_id="api-test-run",
        output_dir=output,
        effective_config_json=effective,
        run_log_jsonl=log,
        run_status_json=status,
        pipeline=RunArtifacts(
            annotated_video=annotated,
            detections_csv=detections,
            tracks_csv=tracks,
            anomalies_csv=anomalies,
            events_csv=events,
            alerts_csv=alerts,
            run_manifest=manifest,
        ),
    )


def test_operator_console_health_and_uploaded_run(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline_sentinel.operator_jobs.run_configured_video", _fake_run)
    app = create_app(workspace=tmp_path / "operator", max_upload_bytes=1024 * 1024)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Pipeline Sentinel" in page.text
        assert "Start Analysis Run" in page.text

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post(
            "/api/jobs/run",
            data={"sensor_id": "IR_CAM_TEST", "modality": "IR"},
            files={"file": ("thermal.mp4", b"fake-input", "video/mp4")},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        deadline = time.monotonic() + 3.0
        payload = None
        while time.monotonic() < deadline:
            payload = client.get(f"/api/jobs/{job_id}").json()
            if payload["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert payload is not None
        assert payload["status"] == "completed"
        assert payload["sensor_id"] == "IR_CAM_TEST"
        assert payload["modality"] == "IR"
        assert payload["summary"]["alerts"] == 1

        alerts = client.get(f"/api/jobs/{job_id}/alerts")
        assert alerts.status_code == 200
        assert alerts.json()[0]["alert_type"] == "visual_anomaly"

        artifacts = client.get(f"/api/jobs/{job_id}/artifacts").json()["artifacts"]
        assert "annotated_video" in artifacts
        video = client.get(artifacts["annotated_video"]["url"])
        assert video.status_code == 200
        assert video.content == b"video"


def test_operator_rejects_unsupported_upload_extension(tmp_path):
    app = create_app(workspace=tmp_path / "operator")
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs/run",
            data={"sensor_id": "EO_1", "modality": "EO"},
            files={"file": ("notes.txt", b"nope", "text/plain")},
        )
        assert response.status_code == 400
