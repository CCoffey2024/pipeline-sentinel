# Testing Pipeline Sentinel

Pipeline Sentinel separates **software correctness** from **model quality**. The tests in `tests/`
are intentionally small and deterministic; benchmark datasets and real workstation media answer a
different question.

## Fast local acceptance run

From PowerShell at the repository root:

```powershell
uv sync --group dev
uv run python scripts/check_release_version.py
uv run ruff check .
uv run pytest
uv build
uv run pipeline-sentinel demo --output outputs\acceptance
```

The deterministic demo exercises the package without downloading model weights. A successful run
produces the normal evidence chain, including detections, tracks, events, alerts, manifest, and
operational status/provenance artifacts.

The key semantic acceptance property is:

```text
Detection -> Track -> Event -> AlertPolicy -> Alert
```

Detections alone must not silently become alerts.

## Operator-service acceptance

The automated service tests verify:

- health and bundled console routes;
- backward-compatible encoded-video upload;
- unified `/api/jobs/media` video upload;
- uploaded still-image sequence submission;
- rejection of mixed video/image submissions;
- local image-folder inspection and read-in-place behavior;
- disabling local-filesystem source access for remote binds;
- persistent operator jobs, evidence artifacts, and fusion orchestration.

For real workstation media and a clean installed wheel, follow `docs/mvp-acceptance.md`.

## Optional YOLO workstation acceptance

The deterministic test environment does **not** run real learned-model inference. Validate that
separately on a machine where inference is intended to run:

```powershell
uv sync --extra yolo --group dev
```

An ordinary encoded video can be processed with:

```powershell
uv run pipeline-sentinel run-yolo input.mp4 `
  --output outputs\yolo `
  --model yolo26n.pt `
  --imgsz 960 `
  --device cpu
```

YOLO detections are tracked by default. `events.csv` and `alerts.csv` may legitimately contain only
headers when no event policy fires.

To explicitly exercise a learned-detector temporal rule:

```powershell
uv run pipeline-sentinel run-yolo input.mp4 `
  --output outputs\yolo-dwell `
  --model yolo26n.pt `
  --enable-dwell-events `
  --dwell-label person `
  --dwell-min-hits 30 `
  --dwell-max-displacement-px 40 `
  --device cpu
```

The dwell rule is a deterministic baseline for plumbing and controlled experiments, not a claim of
mission-grade behavior analytics.

## Current automated test layers

### Data contracts

`test_types.py` verifies persisted/runtime frame contracts and normalized detections. Track, event,
anomaly-observation, and alert contracts are exercised by their owning component and pipeline tests.

### Detector contract

`test_detectors.py` verifies the deterministic ground-truth backend emits Pipeline Sentinel
`Detection` objects and rejects malformed annotation schemas.

### YOLO adapter contract

`test_yolo.py` injects a fake model and proves that the optional provider runtime is normalized into
portable detections without requiring model weights, Torch, a GPU, or network access.

### Tracking

`test_tracking.py` verifies exact IoU math, stable IDs across small motion, class-aware association,
track expiration, and deterministic state reset.

### Anomaly, event, and alert contracts

The anomaly and event tests verify normal-reference scoring mechanics, persistence gating, event
creation, and severity-based alert promotion. A raw anomaly score remains evidence rather than an
alert.

### End-to-end orchestration

`test_pipeline.py` verifies the complete runtime artifact contract and separation of detections,
tracks, anomalies, events, and alerts.

### Source adapters and benchmarks

Image-folder, VisDrone, UAVDT, ingest, manifest, synthetic, COCO-preparation, and evaluator tests use
small local fixtures. No production dataset bytes are required by CI.

### Operator application

Operator-job and service tests verify queue state, persisted jobs, media uploads, local sources,
artifact access, and fusion without running heavyweight learned models.

## Workstation VisDrone acceptance

VisDrone remains useful as the current aerial benchmark source:

```powershell
uv run pipeline-sentinel run-visdrone `
  "D:\FMV\VisDrone\VisDrone2019-VID-val" `
  --output outputs\visdrone-acceptance `
  --max-frames 300 `
  --model yolo26n.pt `
  --imgsz 960 `
  --device cpu
```

The separate `evaluate-visdrone` command scores saved detections against ground truth; adding a
tracker does not silently alter detector-quality metrics.

## Why CI does not download models or benchmark data

CI should answer whether the application contracts and package are healthy, not whether a large
external model or dataset server is reachable. The split remains:

```text
framework-free deterministic tests
    -> repository/software correctness

clean wheel install + operator import/route smoke
    -> packaging correctness

actual runtime + weights + local media/benchmarks
    -> workstation integration and model quality
```

## GitHub Actions gates

The Linux CI job validates release metadata, lints, runs pytest, builds the wheel/source distribution,
generates checksums, clean-installs the core wheel, and smoke-tests packaged CLIs/config/UI/source
adapters.

The Windows MVP job independently runs lint/tests/build and then clean-installs the built wheel with
`[operator]`. It verifies the installed operator command and primary media/local-source routes. This
is the packaging gate most representative of the current workstation MVP target.

A failing CI run is a repository-level regression even if a notebook or developer checkout still
happens to execute.
