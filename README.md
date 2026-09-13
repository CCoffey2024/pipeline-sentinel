# Pipeline Sentinel

Pipeline Sentinel is a modular computer-vision application for turning EO/IR video and image streams
into detections, temporal tracks, semantic events, and human-facing alerts.

The project began as a sequence of learning/R&D notebooks. Stable concepts are being promoted into an
installable Python package so model runtimes, source adapters, trackers, and event algorithms can be
replaced without rewriting the rest of the system.

> **Scope:** defensive sensing, detection, tracking, anomaly scoring, sensor fusion, and human-facing
> alerts for a fictional pipeline corridor. No automated engagement or weapons logic.

## v0.6 milestone — config-driven, auditable production runs

v0.1 established ingest and data contracts. v0.2 added the optional YOLO detector adapter. v0.3 made
VisDrone image sequences a repeatable aerial source. v0.4 added detector-quality evaluation. v0.5
added tracking, events, and alert policy. **v0.6 turns that runtime into a production-style execution
path** with validated configuration, run IDs, structured lifecycle logging, config snapshots, and
failure status artifacts.

The application flow remains:

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

The core rule is:

```text
detection != track != event != alert
```

A model saying “person here” is not the same thing as establishing that the same person persists over
time, inferring a temporal condition, or deciding that a human operator should be notified.

## First production-style run

Install the optional learned runtime, validate the checked-in production profile, then run one video:

```powershell
git clone https://github.com/CCoffey2024/pipeline-sentinel.git
cd pipeline-sentinel

uv sync --extra yolo --group dev

uv run pipeline-sentinel validate-config .\config\production.yaml

uv run pipeline-sentinel run .\input.mp4 `
  --config .\config\production.yaml
```

If `--output` is omitted, the application creates a unique directory under:

```text
outputs/runs/<UTC-timestamp>-<short-id>/
```

A completed run contains:

```text
annotated_video.mp4
detections.csv
tracks.csv
events.csv
alerts.csv
run_manifest.json
effective_config.json
run_log.jsonl
run_status.json
```

`effective_config.json` records the validated configuration actually used plus the SHA-256 of the
source YAML. `run_log.jsonl` records lifecycle events. `run_status.json` is written before inference
and ends as either `completed` or `failed`, so a failed run still leaves operational evidence.

The current `config/production.yaml` uses 960-pixel YOLO inference because the controlled VisDrone
comparison showed a large recall improvement over 640 without the larger precision penalty observed
at 1280. Dwell events remain disabled by default because deployment-specific behavior thresholds
should not be silently treated as universal policy.

See `docs/production-run.md` for the runtime contract and provenance details.

## Deterministic development acceptance

```powershell
uv sync --group dev
uv run ruff check .
uv run pytest
uv run pipeline-sentinel demo --output outputs\demo
```

The deterministic demo creates:

```text
outputs/demo/run/
├── annotated_video.mp4
├── detections.csv
├── tracks.csv
├── events.csv
├── alerts.csv
└── run_manifest.json
```

The reference demo deliberately produces a `normal_maintenance` **event** that is not promoted to an
alert, proving the event/policy boundary end to end.

## Architecture

```text
encoded video -------------------+
                                 |
image sequence ------------------+--> FrameContext
                                 |
future live source --------------+
                                      |
                                      v
                                  Detector
                                      |
                                  Detection[]
                                      |
                                      v
                                   Tracker
                                      |
                                    Track[]
                                      |
                                      v
                                EventDetector
                                      |
                                    Event[]
                                      |
                                      v
                                  AlertPolicy
                                      |
                                    Alert[]
                                      |
                     CSV / annotated video / future API/UI
```

Runtime components communicate through Pipeline Sentinel contracts rather than provider objects.
Ultralytics `Results`, Torch tensors, OpenCV capture handles, or future tracking-library state should
not leak across component boundaries.

## Runtime contracts

- `FrameRecord` — persisted ETL/provenance record.
- `FrameContext` — in-memory frame and pixels passed through runtime components.
- `Detection` — one framework-neutral object observation.
- `Track` — current observation associated with a stable runtime track ID.
- `Event` — temporal/semantic evidence derived from tracks.
- `Alert` — a human-facing notification promoted from an event by policy.

See `docs/architecture.md` for the component contracts and design rules.

## Baseline tracker

Pipeline Sentinel ships a deterministic `IoUTracker`:

- same-class one-to-one association;
- configurable IoU threshold;
- configurable track expiration after missed updates;
- stable integer track IDs within a run;
- reset between runs.

It is intentionally a baseline and adapter boundary. A future ByteTrack or DeepSORT implementation
should return the same `Track` objects so event logic does not change.

## Events and alert policy

Two event detectors currently exist:

- `ScenarioRoleEventDetector` — deterministic adapter for known synthetic/reference scenario roles.
  Learned detectors never receive those ground-truth semantics.
- `DwellEventDetector` — baseline persistence/low-displacement rule for controlled learned-detector
  experiments.

`SeverityAlertPolicy` promotes events at or above a configured severity threshold.

The dwell rule is useful software plumbing and a simple behavior baseline; it is **not** presented as
mission-grade loitering analytics.

## Lower-level YOLO development command

The config-driven `run` command is the application-facing entry point. `run-yolo` remains available
for development experiments where explicit flags are useful:

```powershell
uv run pipeline-sentinel run-yolo .\input.mp4 `
  --output outputs\yolo `
  --model yolo26n.pt `
  --conf 0.25 `
  --imgsz 960 `
  --device cpu
```

YOLO detections are tracked by default. Without an enabled event detector, `events.csv` and
`alerts.csv` are valid empty semantic artifacts rather than detections being silently promoted.

### Explicitly enable the baseline dwell rule

```powershell
uv run pipeline-sentinel run-yolo .\input.mp4 `
  --output outputs\yolo-dwell `
  --model yolo26n.pt `
  --imgsz 960 `
  --enable-dwell-events `
  --dwell-label person `
  --dwell-min-hits 30 `
  --dwell-max-displacement-px 40 `
  --device cpu
```

Tracker controls are also explicit:

```text
--tracker-iou 0.30
--tracker-max-missed 2
```

## VisDrone aerial validation

VisDrone2019-VID remains the current repeatable aerial acceptance/benchmark source. Dataset bytes are
not stored in Git.

Example local layout:

```text
D:\FMV\VisDrone\VisDrone2019-VID-val\
├── annotations\
└── sequences\
```

Inspect the split:

```powershell
uv run pipeline-sentinel visdrone-info `
  "D:\FMV\VisDrone\VisDrone2019-VID-val"
```

Run a 300-frame sequence sample:

```powershell
uv run pipeline-sentinel run-visdrone `
  "D:\FMV\VisDrone\VisDrone2019-VID-val" `
  --sequence uav0000086_00000_v `
  --output outputs\visdrone-acceptance `
  --max-frames 300 `
  --model yolo26n.pt `
  --imgsz 960 `
  --device cpu
```

A dataset-backed run adds `ground_truth.csv` while still producing the normal runtime artifacts:

```text
annotated_video.mp4
detections.csv
tracks.csv
events.csv
alerts.csv
ground_truth.csv
run_manifest.json
```

Detector-quality evaluation remains separate from tracking/event behavior:

```powershell
uv run pipeline-sentinel evaluate-visdrone `
  "outputs\visdrone-acceptance\val\uav0000086_00000_v" `
  --iou-threshold 0.50
```

The evaluator writes `benchmark_summary.json`, `class_metrics.csv`, auditable `matches.csv`, and
excluded-label evidence under `<run_dir>/benchmark/`.

The shared COCO/VisDrone ontology is explicit in the benchmark layer; runtime adapters preserve their
native labels.

## Testing philosophy

Pipeline Sentinel separates software correctness from model quality.

```text
unit / synthetic tests
    -> are the contracts and algorithms correct?

end-to-end deterministic demo
    -> does the complete application pipeline work?

real YOLO + local aerial data
    -> does the optional runtime execute correctly?

benchmark evaluator
    -> how good are the model predictions?
```

CI does not download model weights or benchmark datasets. It tests configuration validation, source
adapters, detector normalization, tracking, event generation, policy behavior, production lifecycle
artifacts, and orchestration with deterministic fixtures.

See `docs/testing.md`.

## Repository map

```text
pipeline-sentinel/
├── .github/workflows/       CI
├── benchmarks/              evaluation definitions and documentation
├── config/                  version-controlled defaults and production profile
├── data/                    local staging; large data ignored
├── docs/                    architecture, migration, testing, operational notes
├── notebooks/learning/      preserved R&D / instructional work
├── outputs/                 generated artifacts; ignored
├── scripts/                 developer / preparation utilities
├── src/pipeline_sentinel/   shipped application package
└── tests/                   deterministic automated tests
```

Useful documentation:

- `docs/architecture.md` — runtime boundaries and contracts.
- `docs/production-run.md` — config-driven execution, logging, and provenance.
- `docs/migration-plan.md` — notebook-to-application extraction map.
- `docs/testing.md` — CI and workstation acceptance.
- `docs/yolo-adapter.md` — learned detector adapter mechanics.
- `docs/visdrone.md` — current aerial source/benchmark workflow.
- `benchmarks/visdrone/README.md` — shared ontology and metric policy.

## Development direction

The project is no longer adding datasets merely to broaden the benchmark list. New data should be
added only when it answers a concrete engineering or mission question.

The next application-oriented milestones are:

1. extract the Notebook 06 embedder/anomaly-scoring boundary where it adds runtime value;
2. add EO/IR evidence fusion behind stable contracts;
3. build wheel/sdist release artifacts and a versioned release workflow;
4. add a service/API or operator UI only after the CLI/runtime contracts remain stable through those
   additions.

## External runtime licensing

Pipeline Sentinel does not vendor Ultralytics source code or pretrained weights. The optional YOLO
extra installs an external runtime package. Review upstream runtime/model licensing terms before
using that backend in commercial or distributed deployments.
