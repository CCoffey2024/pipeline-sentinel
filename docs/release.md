# Release engineering

Pipeline Sentinel releases are tag-driven GitHub releases built from the same commit that carries the
package version. The release workflow does not publish to PyPI; the MVP distribution is an auditable
GitHub Release containing a wheel, source archive, and SHA-256 checksum manifest.

## Version source of truth

`src/pipeline_sentinel/__init__.py` owns `__version__`. Hatch reads that value through
`tool.hatch.version`, so the wheel metadata and runtime `pipeline-sentinel --version` output come from
one source rather than duplicated version strings.

`config/default.yaml` contains a human-readable project version and is checked against the package
version by `scripts/check_release_version.py`.

Release tags must be exact semantic-version tags:

```text
vMAJOR.MINOR.PATCH
```

For example, package version `0.12.0` must be released from tag `v0.12.0`.

## Pull-request and CI gate

Normal CI performs a Linux release-candidate path plus a Windows MVP path.

Linux:

```text
install dev environment
    -> validate release metadata
    -> ruff
    -> pytest
    -> build wheel + sdist
    -> generate SHA256SUMS.txt
    -> clean-install core wheel
    -> smoke-test packaged CLIs/config/UI/source adapters
    -> retain candidate distributions as a short-lived artifact
```

Windows:

```text
install dev environment
    -> validate release metadata
    -> ruff
    -> pytest
    -> build wheel
    -> clean-install wheel with [operator]
    -> smoke-test installed operator CLI and API routes
```

This means the primary MVP workstation platform and the candidate distribution are exercised before
merge rather than discovering packaging problems only after a release tag exists.

## Creating a release

Do not tag a commit until CI on `main` is green and the version/changelog for that commit are final.
From an up-to-date `main` checkout:

```powershell
git switch main
git pull

uv sync --group dev
uv run python scripts/check_release_version.py
uv run ruff check .
uv run pytest
uv build

# For v0.12.0:
git tag -a v0.12.0 -m "Pipeline Sentinel v0.12.0"
git push origin v0.12.0
```

Pushing the tag starts `.github/workflows/release.yml`. Before creating the GitHub Release, the job:

1. validates that the Git tag exactly matches the package version;
2. runs lint and the full deterministic test suite;
3. builds the wheel and source distribution;
4. generates `SHA256SUMS.txt`;
5. installs the core wheel into a clean environment;
6. verifies package/CLI versions and bundled configuration/UI assets;
7. clean-installs the same wheel with `[operator]`;
8. verifies the installed operator command and primary operator API routes.

Only after those gates pass does the workflow create the GitHub Release and attach the distribution
artifacts. A failed gate leaves no new release.

The Windows clean-install gate runs during normal CI before the release commit is tagged.

## Release artifacts

A successful v0.12 release contains files similar to:

```text
pipeline_sentinel-0.12.0-py3-none-any.whl
pipeline_sentinel-0.12.0.tar.gz
SHA256SUMS.txt
```

Model weights, datasets, output runs, and notebook-generated data are deliberately not release
artifacts.

## Installing a released wheel

Download the wheel and `SHA256SUMS.txt` from the matching GitHub Release. On Windows, verify the
wheel hash before installation:

```powershell
Get-FileHash .\pipeline_sentinel-0.12.0-py3-none-any.whl -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
```

Create an isolated environment and install the operator MVP:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".\pipeline_sentinel-0.12.0-py3-none-any.whl[operator]"

pipeline-sentinel --version
pipeline-sentinel-operator --version
pipeline-sentinel-operator --open-browser
```

The core wheel intentionally does not install the optional learned detector runtime. A CLI-only
machine can install the core wheel or the `yolo` extra instead:

```powershell
python -m pip install .\pipeline_sentinel-0.12.0-py3-none-any.whl
python -m pip install ".\pipeline_sentinel-0.12.0-py3-none-any.whl[yolo]"
```

Model weights are external runtime assets and are not included in the wheel.

## MVP acceptance after installation

A green release workflow proves packaging and deterministic software behavior. Before calling a build
an accepted testing MVP on a real workstation, also complete `docs/mvp-acceptance.md` with actual
video/image inputs and record any source-codec or model-loading failures.

## Rollback

Releases are immutable historical artifacts. Do not replace the wheel attached to an existing tag.
If a defect is found, fix it on `main`, increment the patch version, and cut a new release tag. For
example, a defect in `v0.12.0` becomes `v0.12.1` rather than a silently replaced `v0.12.0` wheel.

## What this release process does not claim

A successful software release means the package was built, tested, versioned, and installed
reproducibly across the tested environments. It does not imply that a particular detector, tracking
threshold, event rule, anomaly reference, codec, or sensor is operationally validated for every
deployment domain. Model-quality and sensor-integration evidence remain separate from software-release
evidence.
