from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import pipeline_sentinel.service as service_module
from pipeline_sentinel.operator_jobs import OperatorJob
from pipeline_sentinel.service import create_app


def _completed_job(app, tmp_path: Path, *, managed_input: bool = False) -> tuple[OperatorJob, Path]:
    manager = app.state.job_manager
    job_id = "run-results-test"
    output = manager.runs_dir / job_id
    output.mkdir(parents=True, exist_ok=True)

    (output / "detections.csv").write_text(
        "frame_number,timestamp_s,label,confidence,detector,x1,y1,x2,y2\n"
        "0,0.0,car,0.90,yolo,1,2,10,12\n"
        "0,0.0,person,0.80,yolo,20,3,28,18\n"
        "1,0.033,car,0.70,yolo,2,2,11,12\n",
        encoding="utf-8",
    )
    (output / "tracks.csv").write_text(
        "frame_number,timestamp_s,track_id,label,confidence,hits,duration_s,x1,y1,x2,y2\n"
        "0,0.0,1,car,0.90,1,0.0,1,2,10,12\n"
        "1,0.033,1,car,0.70,2,0.033,2,2,11,12\n"
        "0,0.0,2,person,0.80,1,0.0,20,3,28,18\n",
        encoding="utf-8",
    )
    (output / "anomalies.csv").write_text(
        "frame_number,timestamp_s,track_id,label,score,threshold,is_anomaly\n",
        encoding="utf-8",
    )
    (output / "events.csv").write_text(
        "event_id,frame_number,timestamp_s,event_type,severity,label,message\n",
        encoding="utf-8",
    )
    (output / "alerts.csv").write_text(
        "alert_id,event_id,frame_number,timestamp_s,alert_type,severity,label,message\n",
        encoding="utf-8",
    )
    (output / "annotated_video.mp4").write_bytes(b"video-evidence")

    if managed_input:
        source = manager.uploads_dir / "upload-test" / "source.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"uploaded-source")
    else:
        source = tmp_path / "external-imagery" / "frame001.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"original-source")

    job = OperatorJob(
        job_id=job_id,
        kind="run",
        status="completed",
        created_at_utc="2026-09-14T12:00:00Z",
        started_at_utc="2026-09-14T12:00:01Z",
        completed_at_utc="2026-09-14T12:00:02Z",
        display_name="test-source",
        output_dir=str(output),
        input_path=str(source),
        sensor_id="EO_TEST",
        modality="EO",
        artifacts={
            "annotated_video": str(output / "annotated_video.mp4"),
            "detections_csv": str(output / "detections.csv"),
            "tracks_csv": str(output / "tracks.csv"),
            "anomalies_csv": str(output / "anomalies.csv"),
            "events_csv": str(output / "events.csv"),
            "alerts_csv": str(output / "alerts.csv"),
        },
        summary={
            "frames": 2,
            "detections": 3,
            "tracks": 2,
            "anomalies": 0,
            "events": 0,
            "alerts": 0,
        },
    )
    manager._register(job)
    return job, source


def test_operator_results_summary_and_tables(tmp_path: Path) -> None:
    app = create_app(workspace=tmp_path / "operator")
    job, _ = _completed_job(app, tmp_path)

    with TestClient(app) as client:
        response = client.get(f"/api/jobs/{job.job_id}/results")
        assert response.status_code == 200
        payload = response.json()
        assert payload["kind"] == "run"
        assert payload["classes"][0]["label"] == "car"
        assert payload["classes"][0]["detections"] == 2
        assert payload["classes"][0]["tracks"] == 1
        assert payload["storage"]["total_bytes"] > 0
        assert payload["storage"]["input_is_managed"] is False

        detections = client.get(f"/api/jobs/{job.job_id}/detections?limit=2")
        assert detections.status_code == 200
        assert len(detections.json()) == 2

        tracks = client.get(f"/api/jobs/{job.job_id}/tracks?limit=2")
        assert tracks.status_code == 200
        assert len(tracks.json()) == 2

        script = client.get("/results-console.js")
        assert script.status_code == 200
        assert "Delete run & files" in script.text
        assert "Detections by class" in script.text
        assert "Browse annotated frames (codec-safe)" in script.text
        assert "state.resultTabs" in script.text

        page = client.get("/")
        assert page.status_code == 200
        assert "detailRevision" in page.text
        assert "if (!previousSelected || active || changed) await renderDetail" in page.text


def test_annotated_frame_preview_is_codec_independent(tmp_path: Path, monkeypatch) -> None:
    app = create_app(workspace=tmp_path / "operator")
    job, _ = _completed_job(app, tmp_path)

    class FakeCapture:
        def __init__(self) -> None:
            self.frame = 0

        def isOpened(self) -> bool:  # noqa: N802 - mirror OpenCV API
            return True

        def get(self, prop: int) -> float:
            if prop == service_module.cv2.CAP_PROP_FRAME_COUNT:
                return 3.0
            return 0.0

        def set(self, prop: int, value: float) -> bool:
            if prop == service_module.cv2.CAP_PROP_POS_FRAMES:
                self.frame = int(value)
            return True

        def read(self):
            image = np.full((8, 8, 3), self.frame * 20, dtype=np.uint8)
            return True, image

        def release(self) -> None:
            return None

    monkeypatch.setattr(service_module.cv2, "VideoCapture", lambda _: FakeCapture())

    with TestClient(app) as client:
        response = client.get(f"/api/jobs/{job.job_id}/preview-frame?frame=2")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["x-frame-index"] == "2"
        assert response.headers["x-frame-count"] == "3"
        assert response.content.startswith(b"\xff\xd8")

        outside = client.get(f"/api/jobs/{job.job_id}/preview-frame?frame=3")
        assert outside.status_code == 400


def test_delete_job_preserves_read_in_place_source(tmp_path: Path) -> None:
    app = create_app(workspace=tmp_path / "operator")
    job, source = _completed_job(app, tmp_path)
    output = Path(job.output_dir)

    with TestClient(app) as client:
        response = client.delete(f"/api/jobs/{job.job_id}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["deleted"] is True
        assert payload["external_source_preserved"] is True
        assert payload["freed_bytes"] > 0
        assert source.is_file()
        assert not output.exists()
        assert client.get(f"/api/jobs/{job.job_id}").status_code == 404


def test_delete_job_removes_managed_upload(tmp_path: Path) -> None:
    app = create_app(workspace=tmp_path / "operator")
    job, source = _completed_job(app, tmp_path, managed_input=True)
    upload_root = source.parent

    with TestClient(app) as client:
        response = client.delete(f"/api/jobs/{job.job_id}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["external_source_preserved"] is False
        assert not upload_root.exists()


def test_active_job_cannot_be_deleted(tmp_path: Path) -> None:
    app = create_app(workspace=tmp_path / "operator")
    manager = app.state.job_manager
    job = OperatorJob(
        job_id="run-active-test",
        kind="run",
        status="running",
        created_at_utc="2026-09-14T12:00:00Z",
        started_at_utc="2026-09-14T12:00:01Z",
        completed_at_utc=None,
        display_name="active-source",
        output_dir=str(manager.runs_dir / "run-active-test"),
        input_path=None,
        sensor_id="EO_TEST",
        modality="EO",
        artifacts={},
        summary={},
    )
    manager._register(job)

    with TestClient(app) as client:
        response = client.delete(f"/api/jobs/{job.job_id}")
        assert response.status_code == 409
        assert manager.get_job(job.job_id).status == "running"
