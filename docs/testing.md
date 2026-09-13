# Testing Pipeline Sentinel

Pipeline Sentinel separates **software correctness** from **model quality**. The tests in `tests/`
are intentionally small and deterministic; benchmark datasets answer a different question.

## Fast local acceptance run

From PowerShell at the repository root:

```powershell
uv sync --group dev
uv run ruff check .
uv run pytest
uv run pipeline-sentinel demo --output outputs\acceptance
```

The deterministic v0.5 demo should produce:

```text
outputs/acceptance/run/annotated_video.mp4
outputs/acceptance/run/detections.csv
outputs/acceptance/run/tracks.csv
outputs/acceptance/run/events.csv
outputs/acceptance/run/alerts.csv
outputs/acceptance/run/run_manifest.json
```

The key acceptance property is no longer merely that detections exist. The deterministic reference
path must exercise the full semantic chain:

```text
Detection -> Track -> Event -> AlertPolicy -> Alert
```

`normal_maintenance` should appear as an informational event but must not be promoted to an alert.
Warning scenario events should be promoted.

## Optional YOLO workstation acceptance

The default CI does **not** install the learned-model runtime. Validate that separately on a machine
where inference is intended to run:

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

YOLO detections are tracked by default, so even with no event detector enabled the run should produce
both `detections.csv` and `tracks.csv`. `events.csv` and `alerts.csv` may legitimately contain only
headers.

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

`test_types.py` verifies persisted/runtime frame contracts and normalized detections. New runtime
contracts (`Track`, `Event`, `Alert`) are exercised by their owning component tests and the full
pipeline tests.

### Detector contract

`test_detectors.py` verifies the deterministic ground-truth backend emits Pipeline Sentinel
`Detection` objects and rejects malformed annotation schemas.

### YOLO adapter contract

`test_yolo.py` injects a fake model and proves that the optional provider runtime is normalized into
portable detections without requiring model weights, Torch, a GPU, or network access.

### Baseline tracker

`test_tracking.py` verifies:

- exact IoU math;
- stable IDs across small motion;
- class-aware association;
- track expiration after missed updates;
- deterministic state reset between runs.

These tests prove tracker mechanics, not real-world MOT quality.

### Event and alert contracts

`test_events.py` verifies:

- reference scenario roles become events exactly once per track/role;
- informational events are not promoted by a warning-level policy;
- warning events are promoted to alerts;
- the baseline dwell detector requires both persistence and limited displacement;
- a moving track does not satisfy the dwell rule.

### End-to-end orchestration

`test_pipeline.py` verifies the complete runtime artifact contract. It checks that detections generate
track observations, that detections alone do not become events or alerts, and that the deterministic
reference demo produces separate `detections.csv`, `tracks.csv`, `events.csv`, and `alerts.csv`
evidence.

### Source and benchmark fixtures

The existing ingest, manifest, synthetic, COCO-preparation, VisDrone-adapter, and evaluator tests stay
framework-free and small. No production dataset bytes are required by CI.

## Workstation VisDrone acceptance

VisDrone remains useful as the current aerial acceptance/benchmark source, but dataset expansion is
not part of the v0.5 runtime milestone.

```powershell
uv run pipeline-sentinel run-visdrone `
  "D:\FMV\VisDrone\VisDrone2019-VID-val" `
  --output outputs\visdrone-acceptance `
  --max-frames 300 `
  --model yolo26n.pt `
  --imgsz 960 `
  --device cpu
```

A successful run now includes:

```text
annotated_video.mp4
detections.csv
tracks.csv
events.csv
alerts.csv
ground_truth.csv
run_manifest.json
```

The separate `evaluate-visdrone` command still scores `detections.csv` against ground truth; adding a
tracker does not silently alter detector-quality metrics.

## Why CI does not download models or benchmark data

CI should answer whether the application contracts are healthy, not whether a large external model
or dataset server is reachable. The split remains:

```text
framework-free deterministic tests
    -> repository/software correctness

actual runtime + weights + local benchmark data
    -> workstation integration and model quality
```

## GitHub Actions

The CI workflow remains deliberately small:

```text
checkout
-> install Python
-> install uv
-> uv sync --group dev
-> ruff check
-> pytest
```

A failing CI run is a repository-level regression even if one notebook still happens to execute on a
developer workstation.
