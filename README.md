# Pipeline Sentinel

Pipeline Sentinel is a modular computer-vision prototype for turning video/sensor inputs into stable
runtime contracts, detections, temporal/event evidence, and human-facing alerts.

The project began as a sequence of learning notebooks. The application is being extracted from those
notebooks so data sources and model runtimes can be swapped without rewriting the pipeline.

> **Scope:** defensive sensing, detection, tracking, anomaly scoring, and human-facing alerts for a
> fictional pipeline corridor. No automated engagement or weapons logic.

## v0.3 milestone — VisDrone becomes a first-class dataset

v0.1 established the software foundation. v0.2 added the first learned detector adapter. v0.3 moves
our learned-backend testing from arbitrary local videos to a repeatable aerial benchmark source:
**VisDrone2019-VID**.

Key changes:

- `FrameRecord` remains the serializable ETL/manifest contract;
- `FrameContext` remains the in-memory runtime contract;
- `PipelineSentinel.run_frames()` now accepts generic ordered frame streams, not only encoded MP4;
- `VisDroneDataset` discovers `sequences/` and `annotations/` under a VisDrone VID root;
- `VisDroneSequence` decodes the JPEG frame sequence directly into `FrameContext` objects;
- native VisDrone VID annotations are parsed into normalized XYXY ground truth;
- `pipeline-sentinel visdrone-info` validates and inventories a local dataset root;
- `pipeline-sentinel run-visdrone` runs the existing YOLO adapter directly over a VisDrone sequence;
- every run now writes `detections.csv` in addition to alerts/video/manifest;
- a normalized `ground_truth.csv` is emitted for VisDrone sequences that have annotations.

The runtime still depends on Pipeline Sentinel contracts rather than Ultralytics/Torch objects.

## Core application acceptance

```powershell
git clone https://github.com/CCoffey2024/pipeline-sentinel.git
cd pipeline-sentinel

uv sync --group dev
uv run ruff check .
uv run pytest
uv run pipeline-sentinel demo --output outputs\demo
```

The deterministic demo produces:

```text
annotated_video.mp4
detections.csv
alerts.csv
run_manifest.json
```

## Optional YOLO backend

Install the learned backend separately:

```powershell
uv sync --extra yolo --group dev
```

YOLO still works with an ordinary encoded video:

```powershell
uv run pipeline-sentinel run-yolo .\input.mp4 `
  --output outputs\yolo `
  --model yolo26n.pt `
  --conf 0.25 `
  --device cpu
```

But our preferred repeatable aerial test path is now VisDrone.

## VisDrone2019-VID workflow

Pipeline Sentinel expects the standard VisDrone VID layout:

```text
VisDrone2019-VID-val/
├── annotations/
│   ├── <sequence>.txt
│   └── ...
└── sequences/
    ├── <sequence>/
    │   ├── 0000001.jpg
    │   ├── 0000002.jpg
    │   └── ...
    └── ...
```

For the current Windows workstation the local roots are:

```text
D:\FMV\VisDrone\VisDrone2019-VID-train
D:\FMV\VisDrone\VisDrone2019-VID-val
```

These local paths are examples only; dataset bytes and machine-specific paths are not committed to
Git.

### 1. Inspect the validation split

```powershell
uv run pipeline-sentinel visdrone-info `
  "D:\FMV\VisDrone\VisDrone2019-VID-val"
```

This validates the root and prints the discovered sequence IDs, frame counts, and annotation status.

Use the **validation split first** for zero-shot model evaluation. The train split should be reserved
for model fitting/fine-tuning or training-time analysis so we do not quietly evaluate on data we
trained against.

### 2. Quick YOLO acceptance run on a real aerial sequence

If no `--sequence` is given, the first sequence in sorted order is used:

```powershell
uv run pipeline-sentinel run-visdrone `
  "D:\FMV\VisDrone\VisDrone2019-VID-val" `
  --output outputs\visdrone `
  --max-frames 300 `
  --model yolo26n.pt `
  --conf 0.25 `
  --device cpu
```

For an explicit sequence:

```powershell
uv run pipeline-sentinel run-visdrone `
  "D:\FMV\VisDrone\VisDrone2019-VID-val" `
  --sequence uav0000086_00000_v `
  --output outputs\visdrone `
  --max-frames 300 `
  --model yolo26n.pt `
  --device cpu
```

The sequence run is written under:

```text
outputs/visdrone/val/<sequence-id>/
├── annotated_video.mp4
├── detections.csv
├── ground_truth.csv
├── alerts.csv
└── run_manifest.json
```

`detections.csv` is the learned model output. `ground_truth.csv` is the normalized VisDrone annotation
record. The next benchmark step will compare these two explicitly rather than judging performance by
eye.

### 3. Process sparse frames for a fast experiment

```powershell
uv run pipeline-sentinel run-visdrone `
  "D:\FMV\VisDrone\VisDrone2019-VID-val" `
  --frame-step 5 `
  --max-frames 200 `
  --output outputs\visdrone-sparse `
  --device cpu
```

VisDrone VID is distributed as image sequences rather than encoded videos. `--fps` therefore sets a
**working cadence** for Pipeline Sentinel timestamps and the generated annotated MP4; it is not a
claim that we recovered the original acquisition frame rate from the JPEG files.

## VisDrone annotation contract

Pipeline Sentinel parses the native VID fields:

```text
frame_index,target_id,bbox_left,bbox_top,bbox_width,bbox_height,
score,object_category,truncation,occlusion
```

and adds normalized:

```text
label,x1,y1,x2,y2,ignored
```

The native VisDrone categories are preserved rather than pretending they are identical to COCO.
That matters because the class vocabularies do not line up exactly: for example VisDrone separates
`pedestrian` and `people`, while a generic COCO-pretrained YOLO model normally emits `person`.
Class mapping and formal quality metrics therefore belong in the benchmark layer, not inside the
runtime adapter.

## Detection is not alerting

A YOLO object observation such as `car`, `person`, or `truck` is a **detection**, not a mission alert.
YOLO observations are drawn and saved to `detections.csv`, but they are not automatically promoted to
intrusion/loitering alerts. Those semantics belong to tracking/event logic.

The deterministic `GroundTruthDetector` carries `scenario_role` only so the software integration
path can continue to test known alert behavior.

## Architecture

```text
encoded video -------------------+
                                 |
VisDrone JPEG sequence ----------+--> FrameContext stream
                                      |
                                      v
                                Detector contract
                                      |
                         +------------+------------+
                         |                         |
                 GroundTruthDetector          YoloDetector
                         |                         |
                         +------------+------------+
                                      |
                                      v
                                  Detection
                                      |
                         detections.csv / rendering
                                      |
                         future tracking -> events -> alerts
```

`pipeline.py` does not import Ultralytics or Torch. Dataset-specific layout knowledge lives in the
VisDrone adapter; model-specific result handling lives in the YOLO adapter.

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

Read more:

- `docs/architecture.md` — component boundaries and design rules.
- `docs/yolo-adapter.md` — learned-runtime adapter mechanics.
- `docs/visdrone.md` — VisDrone data/source adapter and benchmark workflow.
- `docs/migration-plan.md` — Notebook 01–09 extraction map.
- `docs/testing.md` — local acceptance and CI strategy.

## Benchmark progression

```text
synthetic integration
    -> COCO generic sanity check
    -> VisDrone aerial-video validation
    -> UAVDT aerial/FMV cross-dataset validation
    -> mission-specific held-out evidence
```

See `benchmarks/README.md`, `benchmarks/coco/README.md`, and `benchmarks/visdrone/README.md`.

## External runtime licensing

Pipeline Sentinel does not vendor Ultralytics source code or pretrained weights. The optional YOLO
extra installs an external runtime package. Review upstream runtime/model licensing terms before
using that backend in a commercial or otherwise distributed product.

## Next extraction milestones

1. Add repeatable VisDrone prediction-vs-ground-truth metrics and shared-class mapping.
2. Notebook 05 -> tracker and event contracts.
3. Notebook 06 -> HOG/DINOv2 embedder adapters + anomaly scorer.
4. Notebook 07 -> EO/IR fusion component.
5. Notebook 03 -> classifier component where it adds value after proposal generation.
6. Notebook 08 -> repeatable multi-dataset model bakeoff.
7. Notebook 09 -> retire notebook-only orchestration in favor of the CLI/package.
