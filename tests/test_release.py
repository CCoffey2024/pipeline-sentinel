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


def test_project_license_and_optional_model_boundaries() -> None:
    data = _pyproject()
    project = data["project"]
    extras = project["optional-dependencies"]

    assert project["license"] == "Apache-2.0"
    assert "LICENSE" in project["license-files"]
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()
    operator_dependencies = "\n".join(extras["operator"]).lower()
    assert "ultralytics" not in operator_dependencies
    assert "torch" not in operator_dependencies
    assert any("ultralytics" in dependency.lower() for dependency in extras["yolo"])
    dinov2_dependencies = "\n".join(extras["dinov2"]).lower()
    assert "torch" in dinov2_dependencies
    assert "torchvision" in dinov2_dependencies


def test_v013_shipment_documents_match_release_profile() -> None:
    quickstart = (ROOT / "TESTER-QUICKSTART.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "RELEASE-NOTES.md").read_text(encoding="utf-8")

    assert "0.13.0" in quickstart
    assert "[operator,yolo,dinov2]" in quickstart
    assert "0.13.0" in release_notes
    assert "[operator,yolo,dinov2]" in release_notes


def test_checksum_manifest_contains_only_release_artifacts(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    (tmp_path / "pipeline_sentinel-0.13.0-py3-none-any.whl").write_bytes(b"wheel")
    (tmp_path / "pipeline_sentinel-0.13.0.tar.gz").write_bytes(b"sdist")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_checksums.py"), str(tmp_path)],
        check=True,
    )

    manifest = (tmp_path / "SHA256SUMS.txt").read_text(encoding="utf-8")
    assert ".gitignore" not in manifest
    assert "pipeline_sentinel-0.13.0-py3-none-any.whl" in manifest
    assert "pipeline_sentinel-0.13.0.tar.gz" in manifest


def test_python_module_entrypoint_reports_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pipeline_sentinel", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == f"pipeline-sentinel {__version__}"
