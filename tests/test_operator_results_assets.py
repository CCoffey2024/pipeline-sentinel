from fastapi.testclient import TestClient

from pipeline_sentinel.service import create_app


def test_operator_console_loads_results_console(tmp_path):
    app = create_app(workspace=tmp_path / "operator")
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert '<script src="/local-sources.js"></script>' in page.text
        assert '<script src="/results-console.js"></script>' in page.text

        script = client.get("/results-console.js")
        assert script.status_code == 200
        assert "Delete Run" in script.text
        assert "result-tabs" in script.text
