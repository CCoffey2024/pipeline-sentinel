from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

from pipeline_sentinel import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_hatch_reads_version_from_package() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "version" in data["project"]["dynamic"]
    assert "version" not in data["project"]
    assert data["tool"]["hatch"]["version"]["path"] == "src/pipeline_sentinel/__init__.py"


def test_default_config_version_matches_package() -> None:
    data = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))

    assert str(data["project"]["version"]) == __version__


def test_python_module_entrypoint_reports_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pipeline_sentinel", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == f"pipeline-sentinel {__version__}"
