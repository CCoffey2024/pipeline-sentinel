import json
from pathlib import Path

import pandas as pd

from pipeline_sentinel.demo import run_demo


def test_end_to_end_reference_demo(tmp_path: Path) -> None:
    artifacts = run_demo(tmp_path / "demo", frame_count=120, size=(320, 180))

    assert artifacts.annotated_video.exists() and artifacts.annotated_video.stat().st_size > 0
    assert artifacts.alerts_csv.exists()
    assert artifacts.run_manifest.exists()

    alerts = pd.read_csv(artifacts.alerts_csv)
    assert not alerts.empty
    assert "normal_maintenance" not in set(alerts["scenario_role"])
    assert "intrusion_vehicle" in set(alerts["scenario_role"])

    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))
    assert manifest["detector_backend"] == "ground_truth"
    assert manifest["frames_processed"] == 120
    assert manifest["alerts_emitted"] == len(alerts)
