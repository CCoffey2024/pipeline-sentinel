import sys

from pipeline_sentinel.cli import main


def test_run_yolo_reports_missing_optional_dependency(monkeypatch, capsys) -> None:
    # Force the optional import path to fail even if a developer has installed the YOLO extra.
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pipeline-sentinel", "run-yolo", "missing.mp4"],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "uv sync --extra yolo --group dev" in captured.err
