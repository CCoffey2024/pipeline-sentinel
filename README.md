# Pipeline Sentinel

Pipeline Sentinel is a modular computer-vision application for turning EO/IR video and image streams
into detections, tracks, anomaly evidence, semantic events, multisensor corroboration, and
human-facing alerts.

The project began as a sequence of learning/R&D notebooks. Stable ideas are promoted into an
installable Python package behind explicit contracts so source adapters, model runtimes, trackers,
representation models, fusion policies, and alert logic can change independently.

> **Scope:** defensive sensing, detection, tracking, anomaly scoring, sensor fusion, and analyst
> alerting for a fictional pipeline corridor. No automated engagement or weapons logic.

## Current development milestone — v0.9

```text
v0.1  ingest + data contracts
v0.2  optional YOLO detector adapter
v0.3  generic frame streams + VisDrone source
v0.4  repeatable detector evaluation
v0.5  tracking -> events -> alert policy
v0.6  config-driven production runs + provenance
v0.7  wheel/sdist release engineering
v0.8  DINOv2-backed anomaly-scoring boundary
v0.9  EO/IR semantic-event late fusion
```

The semantic rule is now:

```text
detection != track != anomaly observation != event != alert
```

A model saying “person here” is not the same thing as establishing temporal identity, deciding an
appearance is unusual, inferring a semantic condition, corroborating that condition with another
sensor, or deciding that an operator should be notified.

## Single-sensor runtime

```text
FrameContext
    -> Detector
    -> Detection[]
    -> Tracker
    -> Track[]
        |              \
        |               -> EventDetector
        v
    AnomalyAnalyzer
        |
    AnomalyObservation[]
        |
    AnomalyEventDetector
        |              /
        +-------------+
              |
           Event[]
              |
         AlertPolicy
              |
           Alert[]
```

Runtime components communicate through Pipeline Sentinel contracts rather than Ultralytics results,
Torch tensors, OpenCV handles, or other provider-specific objects.

## Production run

From a source checkout:

```powershell
git clone https://github.com/CCoffey2024/pipeline-sentinel.git
cd pipeline-sentinel

uv sync --extra yolo --group dev
uv run pipeline-sentinel validate-config .\config\production.yaml

uv run pipeline-sentinel run .\input.mp4 `
  --config .\config\production.yaml
```

If `--output` is omitted, a unique directory is created under:

```text
outputs/runs/<UTC-timestamp>-<short-id>/
```

A completed run contains:

```text
annotated_video.mp4
detections.csv
tracks.csv
anomalies.csv
events.csv
alerts.csv
run_manifest.json
effective_config.json
run_log.jsonl
run_status.json
```

The current production profile uses 960-pixel YOLO inference because the controlled VisDrone
comparison materially improved person recall over 640 without the larger precision penalty observed
at 1280. Dwell and anomaly events remain disabled until deployment-specific policy/reference evidence
is supplied.

See `docs/production-run.md`.

## Notebook 06 -> anomaly scoring

v0.8 extracted the reusable idea from the DINOv2 notebook without welding the application to DINOv2:

```text
track crop -> Embedder -> vector -> normal-reference scorer -> AnomalyObservation
```

`DinoV2Embedder` is an optional adapter. Torch model loading, preprocessing, devices, and tensors stay
inside that adapter. The rest of the application sees NumPy embeddings.

Build a normal-reference artifact from curated normal crops:

```powershell
uv sync --extra dinov2 --group dev

uv run pipeline-sentinel-fit-reference `
  .\references\normal-person-crops `
  --output .\references\person-dinov2-vits14.npz `
  --model dinov2_vits14 `
  --quantile 0.95
```

The baseline preserves the Notebook 06 method: cosine distance from a normal centroid with a threshold
fitted from the normal-score quantile. One high score is evidence, not an alert;
`ConsecutiveAnomalyEventDetector` can require repeated anomalous observations before emitting a
`visual_anomaly` event.

See `docs/anomaly-scoring.md`.

## Notebook 07 -> EO/IR late fusion

v0.9 promotes the stable late-fusion concept while making real-sensor assumptions explicit.
Independent EO and IR runs are processed normally first:

```text
EO run -> Event[] ----+
                      |
IR run -> Event[] ----+--> TemporalConsensusFuser --> fused Event[] --> AlertPolicy
                      |
other sensor Event[] -+
```

Validate the fusion profile:

```powershell
uv run pipeline-sentinel validate-fusion-config .\config\fusion.yaml
```

Fuse two or more completed runs:

```powershell
uv run pipeline-sentinel fuse-runs `
  .\outputs\runs\<eo-run-id> `
  .\outputs\runs\<ir-run-id> `
  --config .\config\fusion.yaml
```

A fusion run produces:

```text
fusion_events.csv
fusion_alerts.csv
fusion_contributors.csv
fusion_manifest.json
effective_fusion_config.json
```

The default strategy requires matching event types from distinct sensors within a configured time
window. Label agreement is configurable. Pixel-space IoU is optional and must only be enabled when
the source products are known to be registered into a common geometry.

Fused confidence is deliberately left unset in v0.9 because independent EO/IR model confidences and
anomaly scores are not assumed to be calibrated onto the same probability scale. Original values are
retained in `fusion_contributors.csv`.

See `docs/sensor-fusion.md`.

## Deterministic acceptance

```powershell
uv sync --group dev
uv run python scripts/check_release_version.py
uv run ruff check .
uv run pytest
uv build
uv run pipeline-sentinel demo --output outputs\demo
```

CI does not download detector weights, DINOv2 weights, or benchmark datasets. It tests contracts,
configuration, source adapters, detector normalization, tracking, anomaly scoring, event generation,
fusion, policy behavior, operational artifacts, release metadata, package building, and clean-wheel
installation with deterministic fixtures.

## Runtime contracts

- `FrameRecord` — persisted ETL/provenance record.
- `FrameContext` — in-memory frame/pixels passed through runtime components.
- `Detection` — framework-neutral per-frame object observation.
- `Track` — observation associated with a stable runtime track ID.
- `AnomalyObservation` — track appearance scored relative to a normal reference.
- `Event` — semantic/temporal evidence derived from tracks, anomaly persistence, or fusion.
- `Alert` — human-facing notification promoted from an event by policy.

See `docs/architecture.md`.

## Lower-level YOLO development command

The config-driven `run` command is the application-facing entry point. `run-yolo` remains available
for explicit experiments:

```powershell
uv run pipeline-sentinel run-yolo .\input.mp4 `
  --output outputs\yolo `
  --model yolo26n.pt `
  --conf 0.25 `
  --imgsz 960 `
  --device cpu
```

YOLO detections are tracked by default. Without an enabled event detector, semantic event and alert
artifacts remain valid and empty rather than silently treating detections as mission events.

## VisDrone aerial validation

Dataset bytes are not stored in Git. Example local run:

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

Evaluate saved detections separately from runtime behavior:

```powershell
uv run pipeline-sentinel evaluate-visdrone `
  "outputs\visdrone-acceptance\val\uav0000086_00000_v" `
  --iou-threshold 0.50
```

The shared evaluator reports fixed-IoU TP/FP/FN, precision, recall, F1, matched IoU, and auditable
match/exclusion artifacts. It is explicitly not presented as the official VisDrone AP evaluator.

## Release engineering

Versioned tags drive a release workflow that validates metadata, runs lint/tests, builds the wheel and
source distribution, generates SHA-256 checksums, installs the wheel into a clean environment,
smoke-tests packaged configuration/CLIs, and only then creates a GitHub Release.

The first tagged production package is `v0.7.0`; later development milestones retain the same release
gates.

See `docs/release.md`.

## Repository map

```text
pipeline-sentinel/
├── .github/workflows/       CI and tag-driven releases
├── benchmarks/              evaluation definitions and documentation
├── config/                  production and fusion profiles
├── data/                    local staging; large data ignored
├── docs/                    architecture, migration, testing, operational notes
├── notebooks/learning/      preserved R&D / instructional work
├── outputs/                 generated artifacts; ignored
├── scripts/                 developer/release/reference utilities
├── src/pipeline_sentinel/   shipped application package
└── tests/                   deterministic automated tests
```

Useful documentation:

- `docs/architecture.md` — runtime boundaries and contracts.
- `docs/production-run.md` — config-driven single-sensor execution and provenance.
- `docs/anomaly-scoring.md` — normal-reference and DINOv2 adapter boundary.
- `docs/sensor-fusion.md` — EO/IR late-fusion contract and assumptions.
- `docs/release.md` — release gates, installation, checksums, and rollback.
- `docs/migration-plan.md` — notebook-to-application extraction map.
- `docs/testing.md` — CI and workstation acceptance.
- `docs/visdrone.md` — aerial validation workflow.

## Development direction

The notebook-derived analytical layers are now largely represented in the runtime. The next major
step should be application delivery rather than another dataset or model experiment: stabilize v0.9,
then add a small service/API boundary and operator-facing presentation on top of the same evidence
contracts.

## External runtime licensing

Pipeline Sentinel does not vendor Ultralytics or DINOv2 source/weights. Optional runtimes and model
weights remain external dependencies. Review their upstream licenses and model terms before commercial
or distributed deployment.
