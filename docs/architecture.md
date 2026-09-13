# Pipeline Sentinel architecture

## Architectural goal

Pipeline Sentinel is being refactored from exploratory notebooks into an installable application
whose runtime depends on **contracts**, not on particular datasets or ML frameworks.

The central design rule is:

> A sensor source, model runtime, tracker, or event algorithm may change without forcing unrelated
> downstream code to change.

OpenCV video capture, VisDrone image sequences, Ultralytics YOLO, a future ONNX/TensorRT detector,
ByteTrack, DeepSORT, DINOv2, or another provider runtime should live behind adapters. The rest of the
system consumes stable Pipeline Sentinel types.

## v0.5 runtime flow

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
                                      v
                                  Detection[]
                                      |
                                      v
                                Tracker contract
                                      |
                                      v
                                    Track[]
                                      |
                                      v
                           EventDetector contract
                                      |
                                      v
                                    Event[]
                                      |
                                      v
                             AlertPolicy contract
                                      |
                                      v
                                    Alert[]
                                      |
                 +--------------------+--------------------+
                 |                    |                    |
                 v                    v                    v
             CSV evidence       annotated video       future API/UI
```

This is the important semantic separation:

```text
detection != track != event != alert
```

A detector observes an object in one frame. A tracker establishes temporal identity. An event detector
uses track history or other evidence to infer a temporal condition. An alert policy decides whether
that event deserves a human-facing notification.

## Data contracts

### `FrameRecord`

Persisted ETL/provenance record for an extracted frame. It contains identifiers, timestamps, image
paths, dimensions, sensor metadata, and source path.

### `FrameContext`

In-memory runtime frame containing frame number, timestamp, NumPy image, source path, sensor ID, and
modality. Both encoded video and image-sequence sources normalize into this type before inference.

### `Detection`

Framework-neutral object observation containing label, confidence, XYXY box, source backend, frame
number, and optional metadata. Ultralytics/PyTorch result objects must not cross this boundary.

### `Track`

A detector observation associated with a stable runtime `track_id`. It carries the current box,
class, confidence, observation count (`hits`), and first-seen frame/time. It does not depend on the
internal state type of any tracking library.

### `Event`

Temporal or semantic evidence derived from tracks. Events have an explicit `event_type`, severity,
source component, optional track ID, and auditable metadata.

### `Alert`

A human-facing notification promoted from an event by an `AlertPolicy`. Alerts retain the originating
`event_id`, so the promotion decision can be audited.

## Detector contract

```python
class Detector(Protocol):
    name: str

    def detect(self, frame: FrameContext) -> list[Detection]:
        ...
```

`GroundTruthDetector` is a deterministic test double. `YoloDetector` is the optional learned runtime.
Both return the same `Detection` objects.

## Tracker contract

```python
class Tracker(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def update(self, frame: FrameContext, detections: list[Detection]) -> list[Track]:
        ...
```

v0.5 ships `IoUTracker`, a deterministic same-class greedy IoU tracker. It is intentionally small and
replaceable. A future ByteTrack or DeepSORT adapter should return the same `Track` contract, leaving
event code unchanged.

The baseline tracker is useful for integration, deterministic tests, and simple scenes. It is not a
claim of state-of-the-art multi-object tracking in dense aerial imagery.

## Event detector contract

```python
class EventDetector(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def update(self, frame: FrameContext, tracks: list[Track]) -> list[Event]:
        ...
```

v0.5 includes two implementations:

- `ScenarioRoleEventDetector` converts known synthetic/reference roles into events for deterministic
  end-to-end testing. Learned detectors do not receive these roles.
- `DwellEventDetector` is a configurable baseline rule that emits one warning event after a track has
  persisted for a minimum number of observations while remaining within a displacement threshold.

The dwell rule is intentionally described as a baseline rule, not mission-grade behavior analysis.

## Alert policy contract

```python
class AlertPolicy(Protocol):
    name: str

    def evaluate(self, event: Event) -> Alert | None:
        ...
```

`SeverityAlertPolicy` promotes events at or above a configured severity threshold. The deterministic
reference demo marks `normal_maintenance` as informational, so it appears in `events.csv` but not in
`alerts.csv`. That is a deliberate integration test of **event != alert**.

## Source and model adapters

`PipelineSentinel.run_video()` decodes ordinary video and delegates to `run_frames()`.
`PipelineSentinel.run_frames()` accepts any ordered `FrameContext` iterable, including VisDrone JPEG
sequences.

`YoloDetector` alone understands Ultralytics result objects. `VisDroneDataset` alone understands the
VisDrone directory/annotation format. The benchmark layer alone owns COCO-to-VisDrone ontology
mapping.

Those boundaries remain unchanged in v0.5.

## Run artifacts

A normal runtime run now produces:

```text
annotated_video.mp4
detections.csv
tracks.csv
events.csv
alerts.csv
run_manifest.json
```

Dataset-backed runs may add source evidence such as `ground_truth.csv`; benchmark evaluation adds its
own benchmark directory. Detection truth, runtime tracks, semantic events, and alerts remain separate
artifacts so each layer can be inspected independently.

The run manifest records component backends and counts including detector, tracker, event detector,
alert policy, detection rows, track observations, unique tracks, events, and alerts.

## Learned-detector event path

YOLO detections are tracked by default but still do **not** automatically become events or alerts.
For controlled experiments the CLI can enable the baseline dwell rule:

```powershell
uv run pipeline-sentinel run-yolo input.mp4 `
  --enable-dwell-events `
  --dwell-label person `
  --dwell-min-hits 30 `
  --dwell-max-displacement-px 40
```

This makes the semantic step explicit and configurable instead of hiding it in detector code.

## Testing strategy

CI uses deterministic fixtures rather than downloading model weights or benchmark datasets:

```text
synthetic source tests
    -> source and annotation contracts

fake detector tests
    -> detector normalization

IoU tracker tests
    -> stable IDs, class-aware association, expiration, reset

event/policy tests
    -> one-time events, dwell rules, severity promotion

end-to-end synthetic demo
    -> Detection -> Track -> Event -> Alert

local VisDrone + YOLO
    -> real aerial runtime and separate quality benchmark
```

## Definition of a healthy architecture

A new detector should require a detector adapter, not tracker rewrites. A new tracker should return
`Track` objects, not force event code to understand provider state. A new event algorithm should
consume stable runtime evidence, not Ultralytics boxes. A new alert policy should consume `Event`
objects, not raw detections.

If those substitutions force unrelated layers to change, the boundary has failed.
