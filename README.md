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
- external benchmark staging for COCO and later UAVDT.

The ground-truth detector is deliberately a **test backend**, not a model. It lets us establish that
the software plumbing works before adding YOLO, MobileNet, DINOv2, tracking, and fusion.

## Quick start (Windows / PowerShell)

```powershell
git clone https://github.com/CCoffey2024/pipeline-sentinel.git
cd pipeline-sentinel

uv sync --group dev
uv run pytest
uv run ruff check .
uv run pipeline-sentinel demo --output outputs\demo
```

The demo produces:

```text
outputs/demo/
├── data/
│   ├── raw/        # deterministic EO/IR video + GT
│   └── interim/    # sampled frames + manifests
└── run/
    ├── alerts.csv
    ├── annotated_video.mp4
    └── run_manifest.json
```

Run the package against an arbitrary video for ETL only:

```powershell
uv run pipeline-sentinel extract input.mp4 `
  --frames-dir data\interim\frames\mission_001 `
  --manifest data\interim\mission_001_manifest.csv `
  --sensor-id EO_CAM_01 `
  --modality EO `
  --sample-every 2
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

See `docs/architecture.md` and `docs/migration-plan.md`.

## Notebooks

The notebooks remain valuable. They are the engineering/R&D record and the place for plots,
experiments, model comparisons, and explanations. They should increasingly import package code
instead of defining production logic in cells.

## COCO and other benchmarks

COCO is an external benchmark, not a runtime dependency. Dataset bytes are excluded from Git.
`prepare_coco_subset.py` creates a smaller repeatable COCO-format subset after the data is obtained.

Evaluation progression:

```text
synthetic integration test -> COCO generic benchmark -> UAVDT aerial/FMV benchmark
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
