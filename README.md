# Pipeline Sentinel

Pipeline Sentinel is a modular computer-vision prototype for turning video/sensor inputs into stable
runtime contracts, detections, temporal/event evidence, and human-facing alerts.

The project began as a sequence of learning notebooks. The application is being extracted from those
notebooks so model runtimes can be swapped without rewriting the pipeline.

> **Scope:** defensive sensing, detection, tracking, anomaly scoring, and human-facing alerts for a
> fictional pipeline corridor. No automated engagement or weapons logic.

## v0.2 milestone — first learned detector adapter

v0.1 established the software foundation. v0.2 adds the first real model-runtime boundary:

- `FrameRecord` remains the serializable ETL/manifest contract;
- `FrameContext` carries in-memory pixels between runtime components;
- the `Detector` protocol now consumes `FrameContext`;
- `GroundTruthDetector` remains the deterministic software test double;
- `YoloDetector` wraps Ultralytics behind the same `Detection` output contract;
- Ultralytics/Torch remain optional rather than core dependencies;
- `pipeline-sentinel run-yolo` exposes the learned backend through the CLI;
- framework-free fake-model tests validate YOLO result normalization without downloading weights;
- detections and mission alerts are explicitly separated.

The central design rule is still that `PipelineSentinel` consumes Pipeline Sentinel contracts, not
Ultralytics, Torch, or another vendor's result objects.

## Quick start — core application

```powershell
git clone https://github.com/CCoffey2024/pipeline-sentinel.git
cd pipeline-sentinel

uv sync --group dev
uv run ruff check .
uv run pytest
uv run pipeline-sentinel demo --output outputs\demo
```

The deterministic demo produces an annotated video, alerts CSV, and run manifest under the selected
output root.

Run the package against an arbitrary video for ETL only:

```powershell
uv run pipeline-sentinel extract input.mp4 `
  --frames-dir data\interim\frames\mission_001 `
  --manifest data\interim\mission_001_manifest.csv `
  --sensor-id EO_CAM_01 `
  --modality EO `
  --sample-every 2
```

## Optional YOLO backend

The learned detector is deliberately an optional install:

```powershell
uv sync --extra yolo --group dev
```

Then run YOLO through the same application pipeline:

```powershell
uv run pipeline-sentinel run-yolo .\input.mp4 `
  --output outputs\yolo `
  --model yolo26n.pt `
  --conf 0.25 `
  --device cpu
```

Use `--class-id` repeatedly to restrict inference to selected class IDs, for example:

```powershell
uv run pipeline-sentinel run-yolo .\input.mp4 `
  --class-id 0 `
  --class-id 2 `
  --class-id 5 `
  --class-id 7
```

When a named pretrained weight file is not already present locally, the model runtime may obtain it
on first use. Model weights (`*.pt`, `*.onnx`, TensorRT engines, etc.) are intentionally excluded from
Git.

### Detection is not alerting

The YOLO adapter emits normalized object observations such as `person`, `car`, or `truck`. Those
objects are drawn on the annotated video and counted in the run manifest, but they are **not**
automatically written as alerts. Mission semantics such as intrusion, loitering, or other temporal
behavior belong to the tracking/event/alert stages that follow detection.

The deterministic `GroundTruthDetector` carries `scenario_role` only so the software integration
path can continue to test known alert behavior.

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
video / stream
      |
      v
 FrameContext
      |
      v
 Detector contract
      |
      +---------- GroundTruthDetector   (test double)
      |
      +---------- YoloDetector          (optional learned backend)
      |
      v
 Detection contract
      |
      +---------- rendering / run metrics
      |
      +---------- future tracking -> events -> alert policy
      |
      +---------- future embeddings -> anomaly scoring
      |
      +---------- future EO/IR fusion
```

`pipeline.py` does not import Ultralytics or Torch. The framework-specific runtime stays inside the
YOLO adapter and is translated immediately into Pipeline Sentinel `Detection` objects.

Read more:

- `docs/architecture.md` — component boundaries and design rules.
- `docs/yolo-adapter.md` — first learned-runtime extraction and adapter mechanics.
- `docs/migration-plan.md` — Notebook 01–09 extraction map.
- `docs/testing.md` — local acceptance and CI strategy.

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

## External runtime licensing

Pipeline Sentinel does not vendor Ultralytics source code or pretrained weights. The optional YOLO
extra installs an external runtime package. Review the upstream runtime/model licensing terms before
using that backend in a commercial or otherwise distributed product.

## Next extraction milestones

1. Notebook 05 -> tracker and event contracts.
2. Notebook 06 -> HOG/DINOv2 embedder adapters + anomaly scorer.
3. Notebook 07 -> EO/IR fusion component.
4. Notebook 03 -> classifier component where it adds value after proposal generation.
5. Notebook 08 -> repeatable benchmark command, including COCO/UAVDT.
6. Notebook 09 -> retire notebook-only orchestration in favor of the CLI/package.

The software acceptance contract remains:

```text
git clone -> uv sync -> uv run ruff check . -> uv run pytest -> uv run pipeline-sentinel demo
```

The optional learned-detector acceptance path adds:

```text
uv sync --extra yolo --group dev -> pipeline-sentinel run-yolo <video>
```
