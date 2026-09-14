# Release engineering

Pipeline Sentinel releases are tag-driven GitHub releases built from the same commit that carries the
package version. The release workflow does not publish to PyPI; the first production distribution is
an auditable GitHub Release containing a wheel, source archive, and SHA-256 checksum manifest.

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

For example, package version `0.7.0` must be released from tag `v0.7.0`.

## Pull-request and CI gate

Every normal CI run now performs this sequence:

```text
install dev environment
    -> validate release metadata
    -> ruff
    -> pytest
    -> build wheel + sdist
    -> generate SHA256SUMS.txt
    -> install wheel in a clean virtual environment
    -> smoke-test both CLI entry points
    -> verify bundled production configuration
    -> retain distributions as a short-lived CI artifact
```

This means a candidate distribution is exercised before merge rather than discovering packaging
problems only after a release tag exists.

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

# For v0.7.0:
git tag -a v0.7.0 -m "Pipeline Sentinel v0.7.0"
git push origin v0.7.0
```

Pushing the tag starts `.github/workflows/release.yml`. Before creating the GitHub Release, the job:

1. validates that the Git tag exactly matches the package version;
2. runs lint and the full deterministic test suite;
3. builds the wheel and source distribution;
4. generates `SHA256SUMS.txt`;
5. installs the wheel into a clean environment;
6. verifies both `pipeline-sentinel --version` and `python -m pipeline_sentinel --version`;
7. verifies the bundled production configuration can be loaded.

Only after those gates pass does the workflow create the GitHub Release and attach the distribution
artifacts. A failed gate leaves no new release.

## Release artifacts

A successful release contains files similar to:

```text
pipeline_sentinel-0.7.0-py3-none-any.whl
pipeline_sentinel-0.7.0.tar.gz
SHA256SUMS.txt
```

Model weights, datasets, output runs, and notebook-generated data are deliberately not release
artifacts.

## Installing a released wheel

Download the wheel and `SHA256SUMS.txt` from the matching GitHub Release. On Windows, verify the
wheel hash before installation:

```powershell
Get-FileHash .\pipeline_sentinel-0.7.0-py3-none-any.whl -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
```

Create an isolated environment and install the wheel:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .\pipeline_sentinel-0.7.0-py3-none-any.whl

pipeline-sentinel --version
python -m pipeline_sentinel --version
```

The core wheel intentionally does not install the optional learned detector runtime. For a machine
that will execute YOLO inference, install the wheel with the `yolo` extra:

```powershell
python -m pip install ".\pipeline_sentinel-0.7.0-py3-none-any.whl[yolo]"
```

Then run the production entry point using the bundled production defaults or an explicit deployment
configuration:

```powershell
pipeline-sentinel run .\input.mp4

# or
pipeline-sentinel run .\input.mp4 --config .\production.yaml
```

## Rollback

Releases are immutable historical artifacts. Do not replace the wheel attached to an existing tag.
If a defect is found, fix it on `main`, increment the patch version, and cut a new release tag. For
example, a defect in `v0.7.0` becomes `v0.7.1` rather than a silently replaced `v0.7.0` wheel.

## What this release process does not claim

A successful software release means the package was built, tested, versioned, and installed
reproducibly. It does not imply that a particular detector, tracking threshold, event rule, or model
is operationally validated for every deployment domain. Model-quality evidence remains separate from
software-release evidence.
