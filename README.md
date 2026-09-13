# Pipeline Sentinel

Pipeline Sentinel is a modular computer-vision prototype for turning video/sensor inputs into a
stable frame data contract, detections, anomaly/event signals, and human-facing alerts.

The project began as a sequence of learning notebooks. The application is now being extracted from
those notebooks so model runtimes can be swapped without rewriting the pipeline.

> **Scope:** defensive sensing, detection, tracking, anomaly scoring, and human-facing alerts for a
> fictional pipeline corridor. No automated engagement or weapons logic.

## v0.1 milestone

v0.1 establishes the software foundation before we move the learned models:

- installable `src/` package;
- canonical `FrameRecord` and `Detection` contracts;
- OpenCV video-ingest adapter;
- manifest validation;
- deterministic EO/IR synthetic data generator;
- framework-neutral `Detector` protocol;
- ground-truth reference detector for integration testing;
- thin `PipelineSentinel` orchestrator;
- command-line interface;
- repeatable tests and GitHub CI;
- preserved learning-notebook area;
- external benchmark staging and deterministic COCO subset tooling.

The ground-truth detector is deliberately a **test backend**, not a model. It lets us establish that
the software plumbing works before adding YOLO, MobileNet, DINOv2, tracking, and fusion.

## Quick start (Windows / PowerShell)

```powershell
git clone https://github.com/CCoffey2024/pipeline-sentinel.git
cd pipeline-sentinel

uv sync --group dev
uv run ruff check .
uv run pytest
uv run pipeline-sentinel demo --output outputs\demo
```

The demo produces an annotated video, alerts CSV, and run manifest under the selected output root.

Run the package against an arbitrary video for ETL only:

```powershell
uv run pipeline-sentinel extract input.mp4 `
  --frames-dir data\interim\frames\mission_001 `
  --manifest data\interim\mission_001_manifest.csv `
  --sensor-id EO_CAM_01 `
  --modality EO `
  --sample-every 2
```

## Repository map

```text
pipeline-sentinel/
├── .github/workflows/       CI
├── benchmarks/              external evaluation definitions/tooling
├── config/                  version-controlled defaults
├── data/                    local data staging; dataset bytes ignored
├── docs/                    architecture, migration, testing, validation notes
├── notebooks/learning/      preserved R&D / instructional snapshots
├── outputs/                 generated artifacts; ignored
├── scripts/                 developer and benchmark-preparation utilities
├── src/pipeline_sentinel/   production/runtime package
└── tests/                   deterministic automated tests
```

## Architecture

```text
Input source
    |
    v
Ingest adapter
    |
    v
FrameRecord contract
    |
    v
Detector adapter ----> Detection contract
    |                         |
    |                  future tracker
    |                  future embedder/anomaly
    |                  future EO/IR fusion
    v                         |
          Pipeline orchestration
                    |
                    v
          alerts / video / manifest
```

The important rule is that `pipeline.py` does **not** import YOLO, Torch, TensorRT, or another model
framework. A model runtime lives behind an adapter that returns the same `Detection` contract.

Read more:

- `docs/architecture.md` — component boundaries and design rules.
- `docs/migration-plan.md` — Notebook 01–09 extraction map.
- `docs/testing.md` — local acceptance and CI strategy.
- `docs/foundation-validation-2026-09-13.md` — current validation record.

## Notebooks

The notebooks remain the engineering/R&D record and the place for plots, experiments, model
comparisons, and explanations. They should increasingly import package code instead of defining
production logic in cells.

The repository currently preserves the Pipeline Sentinel notebook snapshots that were available in
the project archive: Notebook 01, the repaired Notebook 03 classifier/MobileNet path, and the fixed
Notebook 06 DINOv2 anomaly lesson. The remaining lessons are documented in
`notebooks/learning/README.md` and can be reconciled from the local learning workspace as we migrate
them.

## COCO and other benchmarks

COCO is an external benchmark, not a runtime dependency. Dataset bytes are excluded from Git.
`scripts/prepare_coco_subset.py` creates a deterministic smaller COCO-format subset from any standard
COCO export, including a Roboflow export.

Evaluation progression:

```text
synthetic integration test -> COCO generic benchmark -> UAVDT aerial/FMV benchmark
```

See `benchmarks/README.md` and `benchmarks/coco/README.md`.

## Current foundation validation

The repository-foundation work has been exercised in the available build environment with Python
bytecode compilation, **12 automated tests**, and the deterministic 120-frame end-to-end demo.
That validates the reference application plumbing; it does not claim validation of optional learned
backends or Windows dependency installation.

The intended workstation acceptance sequence remains:

```text
uv sync --group dev
-> ruff check
-> pytest
-> pipeline-sentinel demo
```

## Next extraction milestones

1. Notebook 04 -> real detector adapter (YOLO first, framework-neutral output).
2. Notebook 05 -> tracker and event contracts.
3. Notebook 06 -> HOG/DINOv2 embedder adapters + anomaly scorer.
4. Notebook 07 -> EO/IR fusion component.
5. Notebook 03 -> classifier component where it adds value after proposal generation.
6. Notebook 08 -> repeatable benchmark command, including COCO/UAVDT.
7. Notebook 09 -> retire notebook-only orchestration in favor of the CLI/package.

The target is a fresh-clone contract:

```text
git clone -> uv sync -> uv run pytest -> uv run pipeline-sentinel demo
```
