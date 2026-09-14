from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from pipeline_sentinel.service import create_app


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((20, 24, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_local_source_inspection_is_read_only(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    _write_image(frames / "000001.jpg")
    _write_image(frames / "000002.jpg")

    app = create_app(workspace=tmp_path / "operator", allow_local_sources=True)
    with TestClient(app) as client:
        response = client.post(
            "/api/local-sources/inspect",
            json={"source_type": "image_folder", "path": str(frames)},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["imagery_access"] == "read_in_place"
        assert payload["imagery_copied"] is False
        assert payload["sequences"][0]["frame_count"] == 2

        page = client.get("/")
        assert page.status_code == 200
        assert '/local-sources.js' in page.text
        script = client.get("/local-sources.js")
        assert script.status_code == 200
        assert "Local folder / dataset" in script.text
        assert "Image folder" in script.text


def test_remote_mode_disables_server_filesystem_source_access(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    _write_image(frames / "000001.jpg")

    app = create_app(workspace=tmp_path / "operator", allow_local_sources=False)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["local_sources_enabled"] is False
        response = client.post(
            "/api/local-sources/inspect",
            json={"source_type": "image_folder", "path": str(frames)},
        )
        assert response.status_code == 403
