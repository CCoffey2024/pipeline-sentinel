# Notebook-to-application migration plan

Pipeline Sentinel began as a deliberate learning sequence. The notebooks proved concepts, exposed
failure modes, and let us compare approaches interactively. The application preserves that record
while promoting only stable, reusable behavior into `src/pipeline_sentinel/`.

## Status map

| Notebook | Primary purpose | Application destination | Current migration status |
|---|---|---|---|
| 01 — ETL | ingest, frame contract, manifests, synthetic inputs | `types.py`, `ingest.py`, `manifest.py`, `synthetic.py` | **v0.1 extracted** |
| 02 — Classical CV baseline | cheap proposals / classical baseline | future `proposals.py`, benchmark utilities | notebook-only |
| 03 — HOG/SVM → MobileNet | crop classification and lightweight CNN comparison | future `classifiers.py` + classifier adapter | notebook-only |
| 04 — Object detection | detector adapter and optional YOLO backend | `detectors.py`, `yolo.py`, CLI | **v0.2 extracted** |
| 05 — Tracking and events | temporal association and event logic | future `tracking.py`, `events.py` | next domain extraction |
| 06 — DINOv2 anomaly detection | representation learning and distance-based anomaly scoring | future `embeddings.py`, `anomaly.py` | notebook-only |
| 07 — EO + IR fusion | modality alignment and fused evidence | future `fusion.py` | notebook-only |
| 08 — Model bakeoff | comparative evaluation | `benchmarks/`, dataset adapters, evaluation commands | **v0.3 VisDrone plumbing extracted** |
| 09 — End-to-end demo | complete walkthrough | `pipeline.py`, CLI, integration tests | runtime skeleton extracted |

## Extraction rule

Do not copy a notebook cell merely because it exists. Promote code only when it represents one of
these categories:

1. a stable data contract;
2. a reusable algorithm or domain component;
3. an adapter around an external framework, sensor, or data source;
4. validation required for reliable execution;
5. orchestration needed by the application.

Plots, educational print statements, one-off exploratory transforms, confusion-matrix rendering,
and benchmark-only visualization stay outside runtime code unless the shipped application explicitly
needs them.

## Standard migration workflow

```text
1. Identify the durable idea
        |
        v
2. Separate framework/data-source specifics from domain logic
        |
        v
3. Define or reuse input/output contracts
        |
        v
4. Extract the smallest reusable component
        |
        v
5. Unit-test the component
        |
        v
6. Integration-test it behind PipelineSentinel
        |
        v
7. Preserve/update notebook documentation
        |
        v
8. Keep plots/explanations in the notebook
```

The notebook is not deleted when extraction is complete. Its role changes from implementation to
experiment, explanation, and validation.

## Notebook 01 — ETL foundation

Notebook 01 supplied the first durable boundary: downstream code should consume canonical
frame/provenance contracts rather than care whether data came from FMV, thermal video, still imagery,
or another source.

The v0.1 package promoted:

```text
FrameRecord
Detection
OpenCVVideoIngestAdapter
manifest validation
synthetic EO/IR generation
```

## Notebook 04 — detector adapter

The detector lesson forced the first real interface evolution. v0.2 introduced `FrameContext` so a
learned detector can receive in-memory pixels while `FrameRecord` remains serializable provenance.

```python
class Detector(Protocol):
    name: str

    def detect(self, frame: FrameContext) -> list[Detection]:
        ...
```

Two implementations currently satisfy that interface:

```text
GroundTruthDetector -> Detection[]
YoloDetector        -> Detection[]
```

Ultralytics `Results`, `Boxes`, and tensor objects are converted inside `YoloDetector` and do not
cross into the rest of the application.

A learned detector emits object observations, not mission events:

```text
Detection != Event != Alert
```

That separation defines the Notebook 05 boundary.

## Notebook 08 — benchmark plumbing now underway

The first benchmark extraction originally staged COCO and UAVDT conceptually. v0.3 advances this by
making **VisDrone2019-VID** a first-class repeatable aerial source.

The important architectural change is not just "support another dataset." It is that the runtime now
accepts generic ordered frame streams:

```text
encoded video -> FrameContext stream --+
                                     |
VisDrone JPEG sequence -> FrameContext -+-> PipelineSentinel.run_frames()
                                     |
future source --------------------------+
```

`VisDroneDataset` and `VisDroneSequence` own dataset-specific layout and annotation knowledge.
`YoloDetector` remains unchanged.

The current benchmark ladder is:

```text
synthetic demo
    -> COCO generic sanity check
    -> VisDrone aerial-video validation
    -> UAVDT independent aerial/FMV cross-dataset validation
    -> mission-specific held-out data
```

v0.3 saves both `detections.csv` and normalized `ground_truth.csv` for VisDrone runs. The next
Notebook-08-style extraction is the formal evaluator that compares them through an explicit
COCO-to-VisDrone class-mapping policy.

## Notebook 05 — next domain extraction

Tracking and event logic should consume normalized detections and return normalized tracks/events.
The tracker must not care whether detections came from YOLO, ground truth, ONNX, or another future
backend.

```text
FrameContext
    |
    v
Detector -> Detection[]
               |
               v
            Tracker
               |
               v
             Track[]
               |
               v
          Event detector
               |
               v
             Event[]
               |
               v
          Alert policy
```

Tracking answers **which observations belong to the same object over time**. Event logic answers
**what that object's temporal/spatial behavior means**.

## Notebook 06 — embeddings and anomaly scoring

Representation generation and anomaly scoring remain distinct jobs:

```text
crop/image -> Embedder -> vector
reference vectors + vector -> AnomalyScorer -> score / decision
```

`HOGEmbedder` and `DinoV2Embedder` should satisfy the same representation interface. The anomaly
scorer should operate on vectors, not import DINOv2 directly.

## Notebook 07 — fusion

EO/IR fusion becomes a domain component after detections/tracks have a stable representation.
Sensor-specific preprocessing stays at the edge; fused evidence should be expressed through common
contracts downstream.

## Notebook 09 — end-to-end walkthrough

Notebook 09 remains useful as an explanatory walkthrough, but the application itself executes
without Jupyter:

```powershell
uv run pipeline-sentinel demo
uv run pipeline-sentinel run-visdrone <dataset-root>
```

Eventually Notebook 09 should become a thin consumer of package APIs rather than contain a second
copy of orchestration logic.

## What does not migrate

Examples of code that should usually remain notebook-only:

```text
matplotlib setup
interactive image grids
confusion-matrix display code
one-time debug prints
package/version printouts
notebook-specific root discovery
RUN_* teaching switches
ad hoc train/test experiments
cells whose only purpose is explaining an intermediate concept
```

Those are not bad code. They simply serve a different purpose.

## Definition of done for a migrated component

A notebook component is considered migrated when:

- its public inputs and outputs are explicit;
- framework- or dataset-specific objects do not leak across its boundary;
- automated tests cover normal behavior and at least one failure mode;
- it can be invoked outside Jupyter;
- preserved notebook material points toward the package implementation rather than becoming a
  second source of truth;
- the deterministic end-to-end reference demo still passes.
