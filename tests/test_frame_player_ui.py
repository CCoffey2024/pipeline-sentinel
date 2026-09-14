from __future__ import annotations

from fastapi.testclient import TestClient

from pipeline_sentinel.service import create_app


def test_results_console_includes_codec_safe_player_controls(tmp_path) -> None:
    app = create_app(workspace=tmp_path / "operator")

    with TestClient(app) as client:
        response = client.get("/results-console.js")

    assert response.status_code == 200
    script = response.text
    assert 'id="frame-play-button"' in script
    assert 'id="frame-prev-button"' in script
    assert 'id="frame-next-button"' in script
    assert 'id="frame-speed-select"' in script
    assert 'id="frame-loop-toggle"' in script
    assert 'id="frame-preview-time"' in script
    assert "manifest.render_fps" in script
    assert "formatTimestamp" in script
    assert "Pause" in script
