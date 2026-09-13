# Pipeline Sentinel

Pipeline Sentinel is a modular computer-vision prototype for turning video/sensor inputs into stable
runtime contracts, detections, temporal/event evidence, and human-facing alerts.

The project began as a sequence of learning notebooks. The application is being extracted from those
notebooks so data sources and model runtimes can be swapped without rewriting the pipeline.

> **Scope:** defensive sensing, detection, tracking, anomaly scoring, and human-facing alerts for a
> fictional pipeline corridor. No automated engagement or weapons logic.

## v0.4 milestone — measure aerial detector quality

v0.1 established the software foundation. v0.2 added the first learned detector adapter. v0.3 made
VisDrone2019-VID a first-class aerial dataset. v0.4 adds the first repeatable prediction-vs-ground-
truth evaluator.

Key changes:

- `FrameRecord` remains the serializable ETL/manifest contract;
- `FrameContext` remains the in-memory runtime contract;
- `PipelineSentinel.run_frames()` accepts generic ordered frame streams, not only encoded MP4;
- `VisDroneDataset` discovers `sequences/` and `annotations/` under a VisDrone VID root;
- `VisDroneSequence` decodes JPEG frame sequences directly into `FrameContext` objects;
- `pipeline-sentinel run-visdrone` emits predictions plus ground truth scoped to the exact processed frames;
- `pipeline-sentinel evaluate-visdrone` performs deterministic one-to-one IoU matching;
- the evaluator uses a versioned shared COCO/VisDrone ontology instead of silently renaming runtime labels;
- benchmark outputs include per-class TP/FP/FN, precision, recall, F1, matched IoU, and auditable match rows;
- out-of-ontology predictions such as `sports ball` are reported separately instead of disappearing.

The runtime still depends on Pipeline Sentinel contracts rather than Ultralytics/Torch objects, and
the benchmark layer remains separate from model-provider result objects.

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

Our preferred repeatable aerial test path is VisDrone.

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

Use the **validation split first** for zero-shot model evaluation. The train split should be reserved
for model fitting/fine-tuning or training-time analysis so evaluation does not quietly use training
data.

### 2. Run YOLO on a real aerial sequence

If no `--sequence` is given, the first sequence in sorted order is used:

```powershell
uv run pipeline-sentinel run-visdrone `
  "D:\FMV\VisDrone\VisDrone2019-VID-val" `
  --output outputs\visdrone-acceptance `
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
  --output outputs\visdrone-acceptance `
  --max-frames 300 `
  --model yolo26n.pt `
  --device cpu
```

The sequence run is written under:

```text
outputs/visdrone-acceptance/val/<sequence-id>/
├── annotated_video.mp4
├── detections.csv
├── ground_truth.csv
├── alerts.csv
└── run_manifest.json
```

`detections.csv` is model output. `ground_truth.csv` is normalized VisDrone truth for exactly the
frames selected by `--frame-step` and `--max-frames`. The manifest records the detector model,
confidence threshold, NMS IoU, image size, device, and class filter used to produce the predictions.

### 3. Evaluate predictions against ground truth

```powershell
uv run pipeline-sentinel evaluate-visdrone `
  "outputs\visdrone-acceptance\val\uav0000086_00000_v" `
  --iou-threshold 0.50
```

The evaluator creates:

```text
benchmark/
├── benchmark_summary.json
├── class_metrics.csv
├── matches.csv
├── excluded_predictions.csv
└── excluded_ground_truth.csv
```

The shared ontology is explicit and versioned:

```text
COCO person       -> shared person <- VisDrone pedestrian + people
COCO bicycle      -> shared bicycle <- VisDrone bicycle
COCO motorcycle   -> shared motor   <- VisDrone motor
COCO car          -> shared car     <- VisDrone car
COCO truck        -> shared truck   <- VisDrone truck
COCO bus          -> shared bus     <- VisDrone bus
```

Matching is confidence-greedy and one-to-one within each frame and shared class. At the selected IoU
threshold, matched predictions are TP, unmatched predictions are FP, and unmatched ground truth is
FN. `matches.csv` preserves the evidence behind every scored row.

This is the **Pipeline Sentinel shared-class IoU benchmark**, not the official VisDrone AP/mAP
evaluator. v0.4 reports fixed-threshold precision/recall/F1. Ground-truth rows marked ignored are
excluded, but ignored-region overlap does not yet suppress predictions. Out-of-ontology labels are
reported separately and are not scored.

### 4. Process sparse frames for a fast experiment

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
claim that the original acquisition frame rate was recovered from the JPEG files.

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
Class mapping belongs in evaluation, not in the detector adapter.

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

saved detections.csv + ground_truth.csv
                 |
                 v
       shared-ontology evaluator
                 |
          TP / FP / FN evidence
                 |
   precision / recall / F1 / IoU
```

`pipeline.py` does not import Ultralytics or Torch. Dataset-specific layout knowledge lives in the
VisDrone adapter; model-specific result handling lives in the YOLO adapter; model-quality comparison
lives in the evaluation layer.

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
- `benchmarks/visdrone/README.md` — shared ontology and metric policy.

## Benchmark progression

```text
synthetic integration
    -> COCO generic sanity check
    -> VisDrone aerial-video shared-class IoU benchmark
    -> official-style AP/mAP and error slices
    -> UAVDT aerial/FMV cross-dataset validation
    -> mission-specific held-out evidence
```

See `benchmarks/README.md`, `benchmarks/coco/README.md`, and `benchmarks/visdrone/README.md`.

## External runtime licensing

Pipeline Sentinel does not vendor Ultralytics source code or pretrained weights. The optional YOLO
extra installs an external runtime package. Review upstream runtime/model licensing terms before
using that backend in a commercial or otherwise distributed product.

## Next extraction milestones

1. Extend VisDrone evaluation to AP/mAP plus small-object, occlusion, and truncation slices.
2. Notebook 05 -> tracker and event contracts.
3. Notebook 06 -> HOG/DINOv2 embedder adapters + anomaly scorer.
4. Notebook 07 -> EO/IR fusion component.
5. Notebook 03 -> classifier component where it adds value after proposal generation.
6. Notebook 08 -> repeatable multi-dataset model bakeoff.
7. Notebook 09 -> retire notebook-only orchestration in favor of the CLI/package.
