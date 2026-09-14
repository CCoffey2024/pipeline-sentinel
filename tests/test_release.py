from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

from pipeline_sentinel import __version__

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_hatch_reads_version_from_package() -> None:
    data = _pyproject()

    assert "version" in data["project"]["dynamic"]
    assert "version" not in data["project"]
    assert data["tool"]["hatch"]["version"]["path"] == "src/pipeline_sentinel/__init__.py"


def test_default_config_version_matches_package() -> None:
    data = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))

    assert str(data["project"]["version"]) == __version__


def test_project_license_and_optional_yolo_boundary() -> None:
    data = _pyproject()
    project = data["project"]
    extras = project["optional-dependencies"]

    assert project["license"] == "Apache-2.0"
    assert "LICENSE" in project["license-files"]
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()
    assert not any("ultralytics" in dependency.lower() for dependency in extras["operator"])
    assert any("ultralytics" in dependency.lower() for dependency in extras["yolo"])


def test_python_module_entrypoint_reports_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pipeline_sentinel", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == f"pipeline-sentinel {__version__}"
