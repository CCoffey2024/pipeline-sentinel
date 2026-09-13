# Repository foundation validation — 2026-09-13

This note records the validation performed while building the repository-foundation branch.

## Local build-environment results

The package and new repository utilities were exercised against the available Python/OpenCV/pandas
environment.

Passed checks:

```text
python -m compileall -q src scripts tests
pytest
pipeline_sentinel.cli demo --frames 120 --width 320 --height 180
```

Automated tests: **12 passed**.

The deterministic demo processed **120 frames** and emitted **156 alert records**:

```text
intrusion_vehicle   50
dismount_loiter     46
small_uas_like      35
smoke_anomaly       25
```

Generated artifacts were non-empty:

```text
annotated_video.mp4   327,425 bytes
alerts.csv             10,475 bytes
run_manifest.json          487 bytes
```

The run manifest reported:

```text
pipeline_sentinel_version: 0.1.0
detector_backend: ground_truth
frames_processed: 120
alerts_emitted: 156
```

## What this validation does prove

It proves that the v0.1 application plumbing, deterministic source generation, ingest path, manifest
validation, reference detector, artifact generation, and the new COCO-subset preparation logic are
internally consistent in the validation environment.

## What this validation does not prove

It does not validate YOLO, MobileNet, DINOv2, GPU execution, COCO model quality, UAVDT model quality,
or Windows-specific dependency installation. Those are intentionally separate steps.

The build sandbox cannot be treated as a substitute for the normal `uv sync` acceptance run on the
development workstation.

## Workstation acceptance sequence

On a normal network-connected Windows development machine, run:

```powershell
uv sync --group dev
uv run ruff check .
uv run pytest
uv run pipeline-sentinel demo --output outputs\acceptance
```

That workstation run is the final acceptance check before the repository-foundation branch is
considered fully validated across the intended development environment.
