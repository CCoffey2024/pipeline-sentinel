# Notebook-to-application migration plan

Pipeline Sentinel began as a deliberate learning sequence. The notebooks prove concepts, expose
failure modes, and make experiments easy to inspect. The application promotes only stable, reusable
behavior into `src/pipeline_sentinel/`.

## Status map

| Notebook | Primary purpose | Application destination | Migration status |
|---|---|---|---|
| 01 — ETL | ingest, frame contract, manifests, synthetic inputs | `types.py`, `ingest.py`, `manifest.py`, `synthetic.py` | **v0.1 extracted** |
| 02 — Classical CV baseline | cheap proposals / classical baseline | future `proposals.py`, benchmark utilities | notebook-only |
| 03 — HOG/SVM → MobileNet | crop classification and lightweight CNN comparison | future `classifiers.py` | notebook-only |
| 04 — Object detection | detector contract and optional YOLO backend | `detectors.py`, `yolo.py`, CLI | **v0.2 extracted** |
| 05 — Tracking and events | temporal association, event generation, alert promotion | `tracking.py`, `events.py`, `pipeline.py` | **v0.5 extracted** |
| 06 — DINOv2 anomaly detection | representation learning and anomaly scoring | future `embeddings.py`, `anomaly.py` | next domain extraction candidate |
| 07 — EO + IR fusion | modality alignment and fused evidence | future `fusion.py` | notebook-only |
| 08 — Model bakeoff | comparative evaluation | `benchmarks/`, `evaluation.py`, dataset adapters | **v0.3–v0.4 core extracted** |
| 09 — End-to-end demo | complete walkthrough | `pipeline.py`, CLI, integration tests | runtime orchestration extracted |

## Extraction rule

Do not copy a notebook cell merely because it exists. Promote code only when it represents a stable
contract, reusable algorithm/domain component, external-runtime adapter, validation rule, or shipped
orchestration behavior.

Plots, educational print statements, interactive visualizations, and one-off experiments remain in
notebooks unless the shipped application explicitly needs them.

## Standard migration workflow

```text
identify durable idea
    -> separate framework/data-source specifics
    -> define stable input/output contracts
    -> extract smallest reusable component
    -> unit test it
    -> integration test it behind PipelineSentinel
    -> keep notebook as explanation/experiment rather than second implementation
```

## Notebook 01 — ETL foundation

v0.1 established the first durable boundary: downstream code consumes canonical frame/provenance
contracts instead of caring whether data came from FMV, thermal video, still imagery, or another
source.

Promoted concepts include `FrameRecord`, ingest adapters, manifest validation, synthetic EO/IR data,
and the initial normalized `Detection` contract.

## Notebook 04 — detector adapter

v0.2 introduced `FrameContext` so learned detectors receive in-memory pixels while persisted
`FrameRecord` remains serializable provenance.

```python
class Detector(Protocol):
    name: str
    def detect(self, frame: FrameContext) -> list[Detection]: ...
```

`GroundTruthDetector` and `YoloDetector` satisfy the same interface. Ultralytics result/tensor objects
are normalized inside the adapter and do not leak into the application.

## Notebook 08 — benchmark plumbing and evaluator

v0.3 introduced generic frame-stream execution and first-class VisDrone image-sequence support. v0.4
added the shared-ontology IoU evaluator and auditable prediction-vs-ground-truth artifacts.

The benchmark work established a useful engineering rule: **runtime correctness and model quality are
separate concerns**. A detector can execute perfectly while performing poorly under aerial domain
shift.

The project now has enough benchmark evidence to continue application development without adding
additional datasets merely for breadth. Future benchmark work should be driven by a concrete model or
mission question.

## Notebook 05 — tracking and events extracted in v0.5

Notebook 05 supplied the temporal boundary that turns per-frame object observations into persistent
runtime state and then into semantic evidence.

The shipped flow is now:

```text
FrameContext
    -> Detector
    -> Detection[]
    -> Tracker
    -> Track[]
    -> EventDetector
    -> Event[]
    -> AlertPolicy
    -> Alert[]
```

### `Track`

A normalized track carries a stable runtime `track_id`, current box/class/confidence, observation
count, and first-seen frame/time. It contains no ByteTrack/DeepSORT/provider object.

### `IoUTracker`

The first tracker is deliberately deterministic and small: same-class greedy IoU matching with
configurable association threshold and expiration. It is a software baseline and replacement point,
not a claim of best multi-object tracking quality.

### `Event`

An event is explicit temporal/semantic evidence. It is not a detection and it is not yet a human
notification.

`ScenarioRoleEventDetector` preserves deterministic reference behavior without teaching learned
models synthetic ground-truth semantics. `DwellEventDetector` provides a real learned-detector path
for controlled persistence/low-displacement experiments.

### `Alert`

`SeverityAlertPolicy` promotes only events meeting policy threshold. The deterministic demo therefore
contains `normal_maintenance` in `events.csv` but not `alerts.csv`.

That gives the project an executable proof of:

```text
detection != track != event != alert
```

## Notebook 06 — next likely domain extraction

Representation generation and anomaly scoring should remain distinct:

```text
crop/image -> Embedder -> vector
reference vectors + vector -> AnomalyScorer -> score / decision
```

`HOGEmbedder` and `DinoV2Embedder` should eventually satisfy the same representation interface. The
anomaly scorer should operate on vectors rather than import DINOv2 directly.

The exact extraction should now be judged against the shippable application: add it only where it
creates runtime value rather than reproducing notebook material for completeness.

## Notebook 07 — fusion

EO/IR fusion becomes a domain component once track/event evidence and modality alignment rules are
stable. Sensor-specific preprocessing remains at the edge; fused evidence should use common contracts.

## Notebook 09 — end-to-end walkthrough

Notebook 09 remains useful as an explanatory walkthrough, but application execution no longer depends
on Jupyter:

```powershell
uv run pipeline-sentinel demo
uv run pipeline-sentinel run-yolo <video>
uv run pipeline-sentinel run-visdrone <dataset-root>
```

The notebook should increasingly become a thin consumer of package APIs rather than contain a second
copy of orchestration logic.

## What normally stays notebook-only

```text
matplotlib/image-grid setup
confusion-matrix display code
one-time debug prints
notebook root discovery
RUN_* teaching switches
ad hoc train/test experiments
cells whose purpose is explaining an intermediate concept
```

Those are not bad code; they simply serve a different purpose.

## Definition of done for a migrated component

A notebook component is migrated when its public inputs/outputs are explicit, provider-specific
objects do not leak across boundaries, automated tests cover behavior and failure modes, it runs
outside Jupyter, the deterministic end-to-end demo passes, and the notebook no longer acts as the
only implementation.
