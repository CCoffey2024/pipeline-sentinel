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
    representations = output / "representations.csv"
    representation_embeddings = output / "representation_embeddings.f32"
    representation_manifest = output / "representation_manifest.json"
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
    representations.write_text("frame_number\n", encoding="utf-8")
    representation_embeddings.write_bytes(b"")
    representation_manifest.write_text(
        '{"enabled": true, "observations": 2, "embedding_dimension": 384}',
        encoding="utf-8",
    )
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
                "representations_emitted": 2,
                "represented_tracks": 2,
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
            representations_csv=representations,
            representation_embeddings_f32=representation_embeddings,
            representation_manifest_json=representation_manifest,
            anomalies_csv=anomalies,
            events_csv=events,
            alerts_csv=alerts,
            run_manifest=manifest,
        ),
    )


def _fake_frames(
    frames,
    config_path: Path,
    *,
    source_name: str,
    render_fps: float,
    output_dir: Path | None = None,
    run_metadata=None,
):
    del frames, source_name, render_fps, run_metadata
    return _fake_run(Path("uploaded-frames"), config_path, output_dir=output_dir)


def _wait_for_job(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 3.0
    payload = None
    while time.monotonic() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)
    assert payload is not None
    return payload


def test_operator_console_health_and_uploaded_run(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline_sentinel.operator_jobs.run_configured_video", _fake_run)
    app = create_app(workspace=tmp_path / "operator", max_upload_bytes=1024 * 1024)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Pipeline Sentinel" in page.text
        assert "Start Analysis Run" in page.text

        script = client.get("/local-sources.js")
        assert script.status_code == 200
        assert "Media files" in script.text
        assert "Local folder / dataset" in script.text
        assert "DINOv2 track representations" in script.text
        assert "source-representations" in script.text
        assert "/api/jobs/media" in script.text

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

        payload = _wait_for_job(client, job_id)
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


def test_unified_media_endpoint_accepts_video(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline_sentinel.operator_jobs.run_configured_video", _fake_run)
    app = create_app(workspace=tmp_path / "operator")
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs/media",
            data={
                "sensor_id": "EO_MEDIA",
                "modality": "EO",
                "fps": "30",
                "representations_enabled": "false",
            },
            files=[("files", ("clip.mp4", b"video-bytes", "video/mp4"))],
        )
        assert response.status_code == 202
        payload = _wait_for_job(client, response.json()["job_id"])
        assert payload["status"] == "completed"
        assert payload["sensor_id"] == "EO_MEDIA"
        runtime_config = yaml.safe_load(
            (tmp_path / "operator" / "runtime-configs" / f"{payload['job_id']}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert runtime_config["representation"]["enabled"] is False


def test_unified_media_endpoint_accepts_image_sequence(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "pipeline_sentinel.operator_local_sources.run_configured_frames", _fake_frames
    )
    app = create_app(workspace=tmp_path / "operator")
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs/media",
            data={
                "sensor_id": "IR_FRAMES",
                "modality": "IR",
                "fps": "12.5",
                "representations_enabled": "false",
            },
            files=[
                ("files", ("frame0002.jpg", b"two", "image/jpeg")),
                ("files", ("frame0001.jpg", b"one", "image/jpeg")),
                ("files", ("frame0003.png", b"three", "image/png")),
            ],
        )
        assert response.status_code == 202
        payload = _wait_for_job(client, response.json()["job_id"])
        assert payload["status"] == "completed"
        assert payload["sensor_id"] == "IR_FRAMES"
        assert payload["modality"] == "IR"
        assert payload["summary"]["source_type"] == "image_folder"
        assert payload["summary"]["frames"] == 5
        runtime_config = yaml.safe_load(
            (tmp_path / "operator" / "runtime-configs" / f"{payload['job_id']}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert runtime_config["representation"]["enabled"] is False


def test_unified_media_endpoint_rejects_mixed_video_and_images(tmp_path):
    app = create_app(workspace=tmp_path / "operator")
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs/media",
            data={"sensor_id": "EO_1", "modality": "EO", "fps": "30"},
            files=[
                ("files", ("clip.mp4", b"video", "video/mp4")),
                ("files", ("frame0001.jpg", b"image", "image/jpeg")),
            ],
        )
        assert response.status_code == 400
        assert "mixed media runs" in response.json()["detail"]


def test_operator_rejects_unsupported_upload_extension(tmp_path):
    app = create_app(workspace=tmp_path / "operator")
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs/run",
            data={"sensor_id": "EO_1", "modality": "EO"},
            files={"file": ("notes.txt", b"nope", "text/plain")},
        )
        assert response.status_code == 400

        media_response = client.post(
            "/api/jobs/media",
            data={"sensor_id": "EO_1", "modality": "EO", "fps": "30"},
            files=[("files", ("notes.txt", b"nope", "text/plain"))],
        )
        assert media_response.status_code == 400
