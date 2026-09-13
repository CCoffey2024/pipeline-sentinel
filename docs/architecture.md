# Pipeline Sentinel architecture

## Architectural goal

Pipeline Sentinel is being refactored from exploratory notebooks into an installable application
whose runtime depends on **contracts**, not on particular ML frameworks.

The central design rule is:

> A sensor source or model runtime may change without forcing unrelated downstream code to change.

OpenCV, Ultralytics YOLO, PyTorch, ONNX Runtime, TensorRT, DINOv2, or a future vendor runtime should
live behind adapters. The rest of the system consumes stable Pipeline Sentinel types.

## Runtime flow

```text
video / stream / still imagery
            |
            v
       FrameContext
            |
            v
      Detector contract
            |
       +----+----------------+
       |                     |
       v                     v
GroundTruthDetector      YoloDetector
(test double)           (optional runtime)
       |                     |
       +----------+----------+
                  |
                  v
              Detection
                  |
          +-------+----------------------+------------------+
          |                              |                  |
          v                              v                  v
       rendering                     tracking          embeddings
                                         |                  |
                                         v                  v
                                       events         anomaly scoring
                                         |                  |
                                         +---------+--------+
                                                   |
                                             EO/IR fusion
                                                   |
                                                   v
                                              alert policy
                                                   |
                                                   v
                                      CSV / video / API / UI
```

v0.2 implements the detector branch through normalized `Detection` objects. Tracking, anomaly
scoring, fusion, and production alert policy remain future extraction stages.

## Data contracts

### `FrameRecord`

`FrameRecord` is the **persisted ETL/provenance contract**. It contains identifiers, timestamps,
image paths, dimensions, sensor metadata, and the source path. It is designed to be serialized into
manifests and passed between batch-processing stages.

### `FrameContext`

`FrameContext` is the **runtime frame contract**. It carries:

- frame number;
- timestamp;
- in-memory NumPy image;
- optional source path;
- sensor identifier;
- modality.

This split is intentional. A persisted manifest should not contain a live NumPy image, while a real
object detector should not need to reopen a JPEG merely to access pixels that are already in memory.

### `Detection`

`Detection` is the framework-neutral object-observation contract. It contains:

- frame number;
- normalized XYXY integer coordinates;
- class label;
- confidence;
- source adapter;
- optional object/event metadata.

A PyTorch tensor, Ultralytics `Results` object, OpenCV capture handle, pandas row, or provider-specific
object must not leak past the adapter boundary.

Future contracts will likely include `Track`, `Event`, `EmbeddingObservation`, `AnomalyScore`, and
`Alert`.

## Detector contract evolution

The first v0.1 detector contract accepted only `frame_number` because the only backend was a
GroundTruthDetector that could look rows up in a CSV. That interface was sufficient for the test
double but insufficient for a learned model.

v0.2 evolves the contract to:

```python
class Detector(Protocol):
    name: str

    def detect(self, frame: FrameContext) -> list[Detection]:
        ...
```

This is a normal interface-evolution step. The important property is not that the first interface was
perfect; it is that the interface was isolated enough to change without rewriting unrelated
components.

## Adapter implementations

### Ground-truth test backend

```text
GT CSV + FrameContext.frame_number
            |
            v
   GroundTruthDetector
            |
            v
        Detection[]
```

The test backend deliberately ignores frame pixels. It remains useful because it gives integration
tests a known-correct detector path without network downloads, Torch, CUDA, or model uncertainty.

### YOLO learned backend

```text
FrameContext.image
        |
        v
Ultralytics model.predict(...)
        |
        v
framework Results / Boxes / tensors
        |
        v
     YoloDetector
        |
        v
Pipeline Sentinel Detection[]
```

Only `YoloDetector` understands Ultralytics result objects. `PipelineSentinel` does not import
Ultralytics or Torch.

The adapter also owns runtime-specific cleanup such as:

- tensor/array conversion;
- class-ID to label conversion;
- box clipping to image bounds;
- confidence normalization;
- model/runtime metadata.

This is the practical meaning of putting a framework behind an adapter.

## Optional dependencies

The core package remains deliberately lightweight:

```powershell
uv sync --group dev
```

The YOLO runtime is installed only when requested:

```powershell
uv sync --extra yolo --group dev
```

This keeps ordinary ETL, manifest, synthetic-data, and reference-pipeline tests independent of a
large ML runtime. Future ONNX or TensorRT backends can follow the same pattern rather than forcing
every deployment target to install every framework.

## Detection is not alerting

This boundary is important:

```text
object detection != mission event != alert
```

A generic detector can say:

```text
person at box X
car at box Y
```

It cannot, from that single observation alone, reliably say:

```text
loitering
intrusion
suspicious stop
crossed protected corridor
```

Those require temporal association, geometry, policy, or other context.

Therefore a YOLO `Detection` with no `scenario_role` is rendered and counted but does **not** become
an alert. The GroundTruthDetector carries scenario roles only so the deterministic integration test
can continue exercising alert artifact generation until the event/alert layers are extracted.

Notebook 05 is the natural next place to build that boundary properly:

```text
Detection -> Track -> Event -> Alert policy
```

## Domain components

Domain components operate on Pipeline Sentinel contracts rather than vendor objects. Tracking,
anomaly scoring, sensor fusion, and alert policy belong here.

For example, anomaly scoring should accept embeddings through an embedding contract; it should not
know whether vectors came from HOG, DINOv2, or another representation model.

## Orchestration

`pipeline.py` is intentionally thin. Its job is to coordinate components, preserve ordering, collect
artifacts, and surface errors. It should not contain model-specific preprocessing or training logic.

If `pipeline.py` ever needs code like:

```python
from ultralytics import YOLO
```

that is a warning that framework-specific behavior has leaked out of an adapter.

## Application interfaces

The CLI is the first application interface. A later REST API, desktop client, Streamlit UI, or edge
service should call the same package rather than reimplementing the pipeline.

```text
CLI --------+
REST API ---+--> PipelineSentinel --> components
UI ---------+
```

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

## Testing strategy for adapters

The YOLO adapter is unit-tested with an injected fake model object. This lets CI test the most
important contract behavior without installing Ultralytics, downloading weights, or requiring a GPU:

```text
fake framework result
        |
        v
    YoloDetector
        |
        v
normalized Detection objects
```

A separate workstation/runtime test can then answer the different question:

> Does the actual optional model runtime install and execute on this machine?

Keeping those questions separate dramatically improves failure diagnosis.

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
    -> Does performance survive aerial/FMV domain shift?

mission-specific evaluation
    -> Does the system satisfy the operational requirement?
```

A high COCO score is not evidence of good FMV performance. It is one layer of evidence.

## Definition of a healthy architecture

A future model replacement should normally require edits inside one adapter, its configuration, and
its tests. If replacing YOLO forces changes throughout tracking, fusion, alerting, rendering, and
the CLI, the adapter boundary has failed.
