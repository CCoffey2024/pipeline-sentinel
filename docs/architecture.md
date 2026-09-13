# Pipeline Sentinel architecture

## Architectural goal

Pipeline Sentinel is being refactored from exploratory notebooks into an installable application
whose runtime depends on **contracts**, not on particular datasets or ML frameworks.

The central design rule is:

> A sensor source, dataset layout, or model runtime may change without forcing unrelated downstream
> code to change.

OpenCV video capture, VisDrone image sequences, Ultralytics YOLO, PyTorch, ONNX Runtime, TensorRT,
DINOv2, or a future vendor runtime should live behind adapters. The rest of the system consumes
stable Pipeline Sentinel types.

## Runtime flow

```text
encoded video -------------------+
                                 |
VisDrone JPEG sequence ----------+--> FrameContext stream
                                 |
future stream / still source ----+
                                      |
                                      v
                                Detector contract
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
               GroundTruthDetector            YoloDetector
                  (test double)             (optional runtime)
                         |                         |
                         +------------+------------+
                                      |
                                      v
                                  Detection
                                      |
                 +--------------------+--------------------+
                 |                    |                    |
                 v                    v                    v
          detections.csv          tracking            embeddings
                                      |                    |
                                      v                    v
                                    events          anomaly scoring
                                      |                    |
                                      +---------+----------+
                                                |
                                           EO/IR fusion
                                                |
                                                v
                                           alert policy
                                                |
                                                v
                                   CSV / video / API / UI
```

v0.3 implements two real source families: encoded video and VisDrone-style image sequences. Both are
normalized into the same `FrameContext` runtime contract before model inference.

## Data contracts

### `FrameRecord`

`FrameRecord` is the **persisted ETL/provenance contract**. It contains identifiers, timestamps,
image paths, dimensions, sensor metadata, and the source path. It is designed to be serialized into
manifests and passed between batch-processing stages.

### `FrameContext`

`FrameContext` is the **runtime frame contract**. It carries frame number, timestamp, in-memory NumPy
image, optional source path, sensor identifier, and modality.

This split is intentional. A persisted manifest should not contain a live NumPy image, while a real
object detector should not need to reopen an image merely to access pixels that are already in
memory.

An encoded MP4 and a VisDrone JPEG directory therefore become equivalent downstream:

```text
VideoCapture frame ----+
                       +--> FrameContext --> Detector
JPEG file -------------+
```

### `Detection`

`Detection` is the framework-neutral object-observation contract. It contains normalized XYXY
coordinates, class label, confidence, source adapter, frame number, and optional object/event
metadata.

A PyTorch tensor, Ultralytics `Results` object, OpenCV capture handle, pandas row, or provider-specific
object must not leak past the adapter boundary.

Future contracts will likely include `Track`, `Event`, `EmbeddingObservation`, `AnomalyScore`, and
`Alert`.

## Source adapters and orchestration

### Encoded video

`PipelineSentinel.run_video()` remains the convenience entry point for ordinary video files. It uses
OpenCV to decode frames, turns them into `FrameContext` objects, then delegates to the generic frame
runtime.

### Generic frame streams

`PipelineSentinel.run_frames()` is the common orchestration path. It accepts any ordered iterable of
`FrameContext` objects. This is the important source-abstraction boundary added in v0.3.

It lets us add new source types without teaching the detector or downstream logic their storage
format.

### VisDrone

```text
VisDrone root
   |
   +-- sequences/<sequence-id>/*.jpg
   +-- annotations/<sequence-id>.txt
   |
   v
VisDroneDataset / VisDroneSequence
   |
   v
FrameContext stream + normalized ground truth
```

The VisDrone adapter owns dataset-specific knowledge:

- directory discovery;
- numeric frame ordering;
- JPEG decoding;
- native VID annotation parsing;
- VisDrone class names;
- XYWH -> XYXY conversion;
- ignored-region handling.

`YoloDetector` knows none of this.

## Detector contract

```python
class Detector(Protocol):
    name: str

    def detect(self, frame: FrameContext) -> list[Detection]:
        ...
```

The ground-truth test backend may ignore the pixel array and look up rows by frame number. A learned
backend uses the pixels. Both still return the same `Detection` contract.

## YOLO adapter

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

The adapter owns runtime-specific cleanup such as tensor/array conversion, class-ID to label
conversion, box clipping, confidence normalization, and model metadata.

## Dataset vocabulary stays outside model adapters

VisDrone and COCO do not use identical label vocabularies. For example, VisDrone separates
`pedestrian` and `people`, while a generic COCO-pretrained detector usually emits `person`.

The YOLO adapter therefore preserves the model's native output label. The VisDrone adapter preserves
the dataset's native annotation label. A benchmark layer will define the mapping explicitly.

This avoids hiding evaluation policy inside runtime code.

## Run artifacts

All runtime sources now produce a common artifact contract:

```text
annotated_video.mp4
detections.csv
alerts.csv
run_manifest.json
```

Dataset-backed runs may add source-specific evidence such as:

```text
ground_truth.csv
```

`detections.csv` is model output. `ground_truth.csv` is annotation truth. They remain separate.

## Optional dependencies

The core package remains deliberately lightweight:

```powershell
uv sync --group dev
```

The YOLO runtime is installed only when requested:

```powershell
uv sync --extra yolo --group dev
```

VisDrone itself is not a Python dependency. Its dataset bytes remain external local data supplied by
path.

## Detection is not alerting

```text
object detection != mission event != alert
```

A generic detector can say `person at box X` or `car at box Y`. It cannot, from one frame alone,
reliably assert loitering, intrusion, suspicious stop, or protected-corridor crossing.

Therefore generic YOLO detections are rendered and written to `detections.csv` but do not become
alerts unless a later event/policy layer supplies mission semantics.

Notebook 05 is the natural next place to build:

```text
Detection -> Track -> Event -> Alert policy
```

## Testing strategy

The project separates source-contract tests, model-adapter tests, and real-data evaluation:

```text
synthetic VisDrone fixture
    -> Does dataset discovery/annotation parsing work?

fake YOLO model
    -> Does model normalization obey the Detector contract?

PipelineSentinel.run_frames fixture
    -> Can generic frame streams execute end-to-end?

local VisDrone val + real YOLO weights
    -> Does the optional runtime execute on real aerial imagery?

benchmark evaluator
    -> How good are the predictions?
```

CI does not need benchmark downloads or model weights to prove the software contracts are healthy.

## Evaluation layers

```text
unit tests
    -> component correctness

synthetic integration
    -> deterministic application correctness

COCO subset
    -> generic detector sanity

VisDrone validation
    -> aerial-video performance and small-object behavior

UAVDT
    -> independent aerial/FMV cross-dataset shift

mission-specific held-out data
    -> operating-envelope evidence
```

A strong COCO result is not evidence of strong aerial-FMV performance. A strong VisDrone result is
also not proof of mission performance; it is a better, more relevant layer of evidence.

## Definition of a healthy architecture

A new dataset should normally require a source adapter plus its tests and benchmark documentation. A
new detector should normally require one detector adapter plus its tests and configuration.

If adding VisDrone forces YOLO internals to change, or replacing YOLO forces VisDrone parsing to
change, the boundaries have failed.
