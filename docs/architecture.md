# Pipeline Sentinel architecture

## Architectural goal

Pipeline Sentinel is being refactored from exploratory notebooks into an installable application
whose runtime depends on **contracts**, not on particular datasets or ML frameworks.

The central design rule is:

> A sensor source, model runtime, tracker, representation model, event algorithm, or fusion strategy
> may change without forcing unrelated downstream code to change.

OpenCV video capture, VisDrone image sequences, Ultralytics YOLO, a future ONNX/TensorRT detector,
ByteTrack, DeepSORT, DINOv2, and multisensor fusion all live behind explicit boundaries. The rest of
the system consumes stable Pipeline Sentinel types and persisted evidence artifacts.

## Single-sensor runtime flow

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
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
             AnomalyAnalyzer                     EventDetector
                    |                                   |
                    v                                   |
          AnomalyObservation[]                          |
                    |                                   |
                    v                                   |
          AnomalyEventDetector                          |
                    |                                   |
                    +-----------------+-----------------+
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

The semantic separation is now:

```text
detection != track != anomaly observation != event != alert
```

A detector observes an object in one frame. A tracker establishes temporal identity. An anomaly
analyzer may score the tracked appearance against a learned normal reference. Event logic decides
whether track history or persistent anomaly evidence represents a semantic condition. Alert policy
then decides whether a human operator should be notified.

## Multisensor late-fusion flow

Completed sensor runs remain independently auditable and may then be fused:

```text
EO run -> Event[] ----+
                      |
IR run -> Event[] ----+--> TemporalConsensusFuser --> fused Event[] --> AlertPolicy
                      |
other sensor Event[] -+
```

This is **late fusion**. v0.9 does not blend raw EO/IR pixels or provider-specific neural features.
The fusion layer consumes normalized semantic events and emits ordinary `Event` objects so downstream
alert policy remains unchanged.

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

### `AnomalyObservation`

One track-crop score relative to a persisted normal reference. It records the score, threshold,
anomaly decision, source analyzer, and metadata. It is evidence rather than an alert.

### `Event`

Temporal or semantic evidence derived from tracks, anomaly persistence, known synthetic roles, or
late multisensor fusion. Events have an explicit `event_type`, severity, source component, optional
track ID, and auditable metadata.

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

Pipeline Sentinel ships `IoUTracker`, a deterministic same-class greedy IoU tracker. It is
intentionally small and replaceable. A future ByteTrack or DeepSORT adapter should return the same
`Track` contract, leaving event and anomaly code unchanged.

## Representation / anomaly boundary

```python
class Embedder(Protocol):
    name: str

    def encode(self, images: Sequence[np.ndarray]) -> np.ndarray:
        ...
```

The optional DINOv2 adapter keeps Torch, devices, model loading, and preprocessing inside the
adapter. `TrackCropAnomalyAnalyzer` receives only NumPy embeddings and a persisted normal-reference
artifact.

The current Notebook-06-derived baseline uses cosine distance from the normal centroid with a fitted
normal quantile threshold. A persistence-gated anomaly event detector separates a single unusual
score from a semantic `visual_anomaly` event.

## Event and alert boundaries

`ScenarioRoleEventDetector` converts known synthetic/reference roles into events for deterministic
software tests. `DwellEventDetector` is a configurable baseline persistence rule for learned tracks.
`ConsecutiveAnomalyEventDetector` converts persistent anomaly observations into a semantic event.

All of them ultimately produce the same `Event` contract.

`SeverityAlertPolicy` promotes events at or above a configured severity threshold. The deterministic
reference demo marks `normal_maintenance` as informational, so it appears in `events.csv` but not in
`alerts.csv`.

## Fusion boundary

`TemporalConsensusFuser` consumes persisted events from two or more completed sensor runs. Matching
always requires distinct sensor IDs and equal event types, and may additionally require label
agreement and pixel-space IoU.

Pixel-space IoU is opt-in. A configured spatial threshold asserts that the source sensor products have
already been registered into a common geometry. With no spatial threshold, the fuser makes no pixel
registration claim and uses temporal + semantic evidence only.

The fused event confidence is deliberately left unset because source-model confidences are not
assumed to be calibrated to the same probability scale. Contributor confidences remain in the audit
artifact.

See `docs/sensor-fusion.md` for the operational contract.

## Source and model adapters

`PipelineSentinel.run_video()` decodes ordinary video and delegates to `run_frames()`.
`PipelineSentinel.run_frames()` accepts any ordered `FrameContext` iterable, including VisDrone JPEG
sequences.

`YoloDetector` alone understands Ultralytics result objects. `DinoV2Embedder` alone understands Torch
and DINOv2 model objects. `VisDroneDataset` alone understands the VisDrone directory/annotation
format. The benchmark layer alone owns COCO-to-VisDrone ontology mapping. The fusion layer consumes
Pipeline Sentinel evidence rather than any of those provider objects.

## Run artifacts

A single-sensor runtime run produces:

```text
annotated_video.mp4
detections.csv
tracks.csv
anomalies.csv
events.csv
alerts.csv
run_manifest.json
```

A config-driven production run additionally records effective configuration, structured lifecycle
logging, and run status.

A late-fusion run produces:

```text
fusion_events.csv
fusion_alerts.csv
fusion_contributors.csv
fusion_manifest.json
effective_fusion_config.json
```

Evidence stays layered so each inference and policy decision can be audited independently.

## Testing strategy

CI uses deterministic fixtures rather than downloading model weights or benchmark datasets:

```text
synthetic source tests
    -> source and annotation contracts

fake detector tests
    -> detector normalization

IoU tracker tests
    -> stable IDs, class-aware association, expiration, reset

anomaly tests
    -> reference fitting, serialization, scoring, persistence

event/policy tests
    -> semantic event generation and severity promotion

fusion tests
    -> time, label, sensor identity, optional spatial gating, audit artifacts

end-to-end synthetic demo
    -> Detection -> Track -> Event -> Alert

local VisDrone + YOLO
    -> real aerial runtime and separate quality benchmark
```

## Definition of a healthy architecture

A new detector should require a detector adapter, not tracker rewrites. A new tracker should return
`Track` objects, not force event code to understand provider state. A new representation model should
implement `Embedder` without changing anomaly policy. A new fusion strategy should consume normalized
sensor evidence rather than raw framework objects. A new alert policy should consume `Event` objects,
not raw detections.

If those substitutions force unrelated layers to change, the boundary has failed.
