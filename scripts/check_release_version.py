from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = ROOT / "src" / "pipeline_sentinel" / "__init__.py"
PYPROJECT_PATH = ROOT / "pyproject.toml"
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"
SEMVER_TAG = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")


def package_version() -> str:
    tree = ast.parse(INIT_PATH.read_text(encoding="utf-8"), filename=str(INIT_PATH))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    raise RuntimeError(f"Could not find string __version__ assignment in {INIT_PATH}")


def validate_pyproject() -> None:
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = data.get("project", {})
    dynamic = project.get("dynamic", [])
    if "version" not in dynamic:
        raise RuntimeError("pyproject.toml must declare version as dynamic")
    if "version" in project:
        raise RuntimeError("pyproject.toml must not duplicate a static project.version")
    version_path = data.get("tool", {}).get("hatch", {}).get("version", {}).get("path")
    expected = "src/pipeline_sentinel/__init__.py"
    if version_path != expected:
        raise RuntimeError(f"tool.hatch.version.path must be {expected!r}")


def validate_default_config(version: str) -> None:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    configured = data.get("project", {}).get("version")
    if str(configured) != version:
        raise RuntimeError(
            f"config/default.yaml project.version={configured!r} does not match package {version!r}"
        )


def validate_tag(tag: str, version: str) -> None:
    match = SEMVER_TAG.fullmatch(tag)
    if match is None:
        raise RuntimeError(f"Release tag must look like vMAJOR.MINOR.PATCH, got {tag!r}")
    if match.group("version") != version:
        raise RuntimeError(f"Release tag {tag!r} does not match package version v{version}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Pipeline Sentinel release version metadata")
    parser.add_argument("--tag", default=None, help="Optional Git tag to validate, e.g. v0.7.0")
    args = parser.parse_args()

    try:
        version = package_version()
        validate_pyproject()
        validate_default_config(version)
        if args.tag:
            validate_tag(args.tag, version)
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"release-version-check: ERROR: {exc}", file=sys.stderr)
        return 2

    message = f"release-version-check: package={version}"
    if args.tag:
        message += f" tag={args.tag}"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
