# Pipeline Sentinel architecture

## Architectural goal

Pipeline Sentinel is being refactored from a sequence of exploratory notebooks into a small,
installable application whose runtime depends on **contracts**, not on particular ML frameworks.

The central design rule is:

> A sensor source or model runtime may change without forcing unrelated downstream code to change.

That means OpenCV, YOLO, PyTorch, ONNX Runtime, TensorRT, DINOv2, or a future vendor runtime should
live behind adapters. The rest of the system consumes stable Pipeline Sentinel types.

## Runtime flow

```text
video / stream / still imagery
            |
            v
      Ingest Adapter
            |
            v
       FrameRecord
            |
            v
      Detector contract
            |
            v
        Detection
            |
            +-----------------------------+
            |                             |
            v                             v
        tracking                     embeddings
            |                             |
            v                             v
          events                    anomaly scoring
            |                             |
            +-------------+---------------+
                          |
                    EO/IR fusion
                          |
                          v
                     alert policy
                          |
                          v
               CSV / video / API / UI
```

Only the upper part of this diagram is implemented in v0.1. Tracking, anomaly scoring, fusion, and
production detector backends are the next extraction stages.

## Layers

### 1. Data contracts

`types.py` contains the canonical values exchanged between components. These types should be small,
serializable, and independent of frameworks.

Current examples:

- `FrameRecord`: provenance and frame-level metadata.
- `Detection`: normalized object-detection output using XYXY image coordinates.

Future contracts will likely include `Track`, `EmbeddingObservation`, `AnomalyScore`, and `Alert`.

A PyTorch tensor, Ultralytics result object, OpenCV capture handle, pandas row, or Roboflow-specific
object should not cross a component boundary unless the boundary explicitly exists for that type.

### 2. Adapters

Adapters translate external systems into Pipeline Sentinel contracts.

Current adapter:

```text
OpenCV video -> OpenCVVideoIngestAdapter -> FrameRecord manifest
```

Current detector test double:

```text
Ground-truth CSV -> GroundTruthDetector -> Detection objects
```

Planned detector adapter:

```text
YOLO runtime -> YoloDetector -> Detection objects
```

The important property is that `PipelineSentinel` receives the same `Detection` objects in either
case.

### 3. Domain components

Domain components operate on Pipeline Sentinel contracts rather than vendor objects. Tracking,
anomaly scoring, sensor fusion, and alert policy belong here.

For example, anomaly scoring should accept embeddings through an embedding contract; it should not
know whether those vectors came from HOG, DINOv2, or another representation model.

### 4. Orchestration

`pipeline.py` is intentionally thin. Its job is to coordinate components, preserve ordering, collect
artifacts, and surface errors. It should not contain model-specific preprocessing or training logic.

If `pipeline.py` ever needs code like:

```python
from ultralytics import YOLO
```

that is a warning that framework-specific behavior has leaked out of an adapter.

### 5. Interfaces

The CLI is the first application interface. A later REST API, Streamlit UI, desktop client, or edge
service should call the same package rather than reimplementing the pipeline.

```text
CLI --------+
REST API ---+--> PipelineSentinel --> components
UI ---------+
```

## Why the ground-truth detector exists

`GroundTruthDetector` is a **test double**, not an ML model. It gives the software stack a
known-correct backend so we can test video I/O, data contracts, orchestration, alert generation, and
artifact creation without conflating those failures with model-runtime, CUDA, or dependency
failures.

This gives us two separate questions:

```text
Does the application plumbing work?      -> GroundTruthDetector
Does the learned detector work well?     -> YOLO / future adapters + benchmark data
```

That separation is extremely useful when debugging real systems.

## Repository boundaries

```text
src/pipeline_sentinel/   production/runtime package
tests/                   automated correctness checks
notebooks/learning/      R&D and instructional record
benchmarks/              external evaluation definitions and tooling
scripts/                 developer/data-preparation utilities
config/                  version-controlled default behavior
docs/                    architecture, migration, validation notes
outputs/                 generated runtime artifacts; not committed
data/                    local data workspace; dataset bytes not committed
```

The runtime package must never require a notebook to have been executed first.

## Notebook relationship

The notebooks remain useful as the engineering lab book. They are the right place for plots,
confusion matrices, exploratory model comparisons, pedagogical explanations, and temporary
experiments.

They are not the source of truth for reusable application behavior.

The desired direction is:

```text
old pattern
notebook defines reusable function -> later notebook copies it

new pattern
package defines reusable function -> notebooks import and experiment with it
```

Notebook execution order, Jupyter kernel state, `sys.path` surgery, plotting calls, and hard-coded
project-root discovery must not be runtime requirements.

## Configuration rule

Runtime choices should progressively move from edited source code into explicit configuration.
Notebook switches such as `RUN_YOLO = False` or `RUN_DINOV2 = False` are useful during exploration;
a shipped application should express those choices through CLI/configuration while preserving safe
defaults.

## Evaluation layers

Pipeline Sentinel separates software correctness from model quality:

```text
unit tests
    -> Does a component obey its contract?

synthetic integration test
    -> Does the complete application execute predictably?

COCO subset
    -> Does a generic object detector behave sensibly on independent real imagery?

UAVDT / aerial benchmark
    -> Does performance survive the actual aerial/FMV domain shift?

mission-specific evaluation
    -> Does the system satisfy the operational requirement?
```

A high COCO score is not evidence of good FMV performance. It is one layer of evidence.

## Definition of a healthy architecture

A future model replacement should normally require edits inside one adapter, its configuration, and
its tests. If replacing YOLO forces changes throughout tracking, fusion, alerting, rendering, and
the CLI, the adapter boundary has failed.
