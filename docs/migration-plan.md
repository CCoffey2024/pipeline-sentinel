# Notebook-to-application migration plan

Pipeline Sentinel began as a deliberate learning sequence. The notebooks proved concepts, exposed
failure modes, and let us compare approaches interactively. The application should preserve that
record while promoting only stable, reusable behavior into `src/pipeline_sentinel/`.

## Status map

| Notebook | Primary purpose | Application destination | Current migration status |
|---|---|---|---|
| 01 — ETL | ingest, frame contract, manifests, synthetic inputs | `types.py`, `ingest.py`, `manifest.py`, `synthetic.py` | **v0.1 extracted** |
| 02 — Classical CV baseline | cheap proposals / classical baseline | future `proposals.py`, benchmark utilities | notebook-only |
| 03 — HOG/SVM → MobileNet | crop classification and lightweight CNN comparison | future `classifiers.py` + classifier adapter | notebook-only |
| 04 — Object detection | detector adapter and optional YOLO backend | `detectors.py` | **next extraction** |
| 05 — Tracking and events | temporal association and event logic | future `tracking.py`, `events.py` | notebook-only |
| 06 — DINOv2 anomaly detection | representation learning and distance-based anomaly scoring | future `embeddings.py`, `anomaly.py` | notebook-only |
| 07 — EO + IR fusion | modality alignment and fused evidence | future `fusion.py` | notebook-only |
| 08 — Model bakeoff | comparative evaluation | `benchmarks/`, evaluation commands | scaffold only |
| 09 — End-to-end demo | complete walkthrough | `pipeline.py`, CLI, integration tests | v0.1 skeleton extracted |

## Extraction rule

Do not copy a notebook cell merely because it exists. Promote code only when it represents one of
these categories:

1. a stable data contract;
2. a reusable algorithm or domain component;
3. an adapter around an external framework, sensor, or data source;
4. validation required for reliable execution;
5. orchestration needed by the application.

Plots, educational print statements, one-off exploratory transforms, confusion-matrix rendering,
and benchmark-only code remain outside runtime code unless the shipped application explicitly needs
them.

## Migration workflow for each notebook

For each notebook we will follow the same sequence:

```text
1. Identify the durable idea
        |
        v
2. Separate framework-specific code from domain logic
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
7. Update the notebook to import package code
        |
        v
8. Keep plots/explanations in the notebook
```

The notebook is not deleted when extraction is complete. Its role changes from **implementation** to
**experiment / explanation / validation**.

## Notebook 01 — completed first pass

Notebook 01 already contained the strongest software boundary in the learning sequence: downstream
code should consume a canonical frame contract rather than care whether data came from FMV,
thermal video, still imagery, or another source.

The v0.1 package therefore promotes:

```text
FrameRecord
Detection
OpenCVVideoIngestAdapter
manifest validation
synthetic EO/IR generation
```

The notebook remains useful for inspecting frames, viewing the protected corridor, and explaining
why provenance matters.

## Notebook 04 — next production extraction

The next important step is to turn the detector lesson into a real adapter boundary.

Target shape:

```python
class Detector(Protocol):
    def detect(self, frame, frame_number, timestamp_s):
        ...

class GroundTruthDetector:
    ...

class YoloDetector:
    ...
```

Both backends return Pipeline Sentinel `Detection` objects. Ultralytics/YOLO result objects stay
inside `YoloDetector`.

Acceptance test:

```text
swap GroundTruthDetector -> YoloDetector
without modifying PipelineSentinel.run()
```

If that swap requires edits throughout the application, the interface is too weak.

## Notebook 05 — tracking and event logic

Tracking should consume normalized detections and return normalized tracks. The tracker should not
care whether detections came from YOLO, ground truth, ONNX, or a future backend.

Likely future contract:

```text
Detection -> Tracker -> Track
Track + time + region/policy -> Event
```

Event logic should be separable from tracker internals so different trackers can be evaluated.

## Notebook 06 — embeddings and anomaly scoring

Representation generation and anomaly scoring are distinct jobs and should remain distinct in the
application.

```text
crop/image -> Embedder -> vector
reference vectors + vector -> AnomalyScorer -> score / decision
```

`HOGEmbedder` and `DinoV2Embedder` should satisfy the same representation interface. The anomaly
scorer should operate on vectors, not import DINOv2 directly.

This preserves the control experiment from the notebook: a stronger representation has to improve
the mission metric rather than merely be newer.

## Notebook 07 — fusion

EO/IR fusion becomes a domain component after detections/tracks have a stable representation.
Sensor-specific preprocessing stays at the edge; fused evidence should be expressed through common
contracts downstream.

## Notebook 08 — benchmarks

The bakeoff notebook should evolve into repeatable benchmark commands and versioned evaluation
configuration. Notebook plots can remain a presentation layer over saved benchmark results.

Initial benchmark ladder:

```text
synthetic demo -> COCO subset -> UAVDT -> mission-specific held-out data
```

## Notebook 09 — end-to-end walkthrough

Notebook 09 remains valuable as an explanatory walkthrough, but the application itself should be
executable without Jupyter:

```powershell
uv run pipeline-sentinel demo
```

Eventually Notebook 09 should become a thin consumer of the package rather than contain a second
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
- framework-specific objects do not leak across its boundary;
- automated tests cover normal behavior and at least one failure mode;
- it can be invoked outside Jupyter;
- the notebook imports the package implementation instead of maintaining a divergent duplicate;
- the end-to-end demo still passes.

This keeps the learning material while preventing the classic R&D failure mode where the notebook
and the deployed implementation quietly become two different systems.
