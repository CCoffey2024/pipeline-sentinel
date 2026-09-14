# Pipeline Sentinel

Pipeline Sentinel is a modular computer-vision application for turning EO/IR video and image streams
into detections, tracks, anomaly evidence, semantic events, multisensor corroboration, and
human-facing alerts.

The project began as a sequence of learning/R&D notebooks. Stable ideas are promoted into an
installable Python package behind explicit contracts so source adapters, model runtimes, trackers,
representation models, fusion policies, delivery layers, and alert logic can change independently.

> **Scope:** defensive sensing, detection, tracking, anomaly scoring, sensor fusion, and analyst
> alerting for a fictional pipeline corridor. No automated engagement or weapons logic.

## Current development milestone — v0.12 MVP candidate

```text
v0.1   ingest + data contracts
v0.2   optional YOLO detector adapter
v0.3   generic frame streams + VisDrone source
v0.4   repeatable detector evaluation
v0.5   tracking -> events -> alert policy
v0.6   config-driven production runs + provenance
v0.7   wheel/sdist release engineering
v0.8   DINOv2-backed anomaly-scoring boundary
v0.9   EO/IR semantic-event late fusion
v0.10  local service/API + operator-facing web console
v0.11  read-in-place image folders + VisDrone/UAVDT operator sources
v0.12  unified video/image ingest + MVP packaging/acceptance hardening
```

The semantic rule remains:

```text
detection != track != anomaly observation != event != alert
```

The application-delivery rule is equally important:

```text
UI != analytics
API != analytics
job orchestration != analytics
source adapter != copied dataset
```

The browser console and HTTP service consume the same production APIs and evidence artifacts as the
CLI rather than creating a second implementation of the vision pipeline.

## Operator console

Pipeline Sentinel can be driven as an operator application instead of as a set of notebook or CLI
steps.

From a source checkout on Windows, double-click:

```text
start-operator.cmd
```

The development launcher synchronizes the operator shell plus the optional YOLO backend used by the
current production profile, starts the local service, and opens the browser. The equivalent command
is:

```powershell
uv sync --extra operator --extra yolo --group dev
uv run pipeline-sentinel-operator --open-browser
```

The default console is:

```text
http://127.0.0.1:8765/
```

From the UI an operator can:

- select one supported encoded video or one-or-more still-image frames;
- mix still-image resolutions and orientations using automatic aspect-preserving letterboxing;
- point at a large local image folder and read it in place;
- use native VisDrone or UAVDT image-sequence layouts;
- assign a sensor ID and EO, IR, or OTHER modality independently of file type;
- set working FPS for generic still-image sequences;
- start a production analysis run with one button;
- watch queued/running/completed/failed jobs;
- inspect summary counts and alerts;
- open/download evidence artifacts and annotated video;
- select completed sensor runs and start EO/IR late fusion.

The default operator workspace is:

```text
outputs/operator/
├── jobs/              persisted job state
├── uploads/           browser-uploaded media
├── runtime-configs/   job-specific validated runtime identity
├── runs/              production-run evidence
└── fusions/           multisensor fusion evidence
```

Large local image sources are read in place and are not mirrored into the workspace.

The v0.12 MVP service is deliberately local-first. It binds to loopback by default and has no
authentication or multi-user security model. A non-loopback bind is refused unless `--allow-remote`
is explicitly supplied. Local-filesystem source APIs are disabled on non-loopback binds. Do not expose
this version directly to an untrusted network.

See `docs/operator-console.md` and `docs/mvp-acceptance.md`.

## Install a release artifact

Core package:

```powershell
python -m pip install .\pipeline_sentinel-0.12.0-py3-none-any.whl
```

Operator application shell:

```powershell
python -m pip install ".\pipeline_sentinel-0.12.0-py3-none-any.whl[operator]"
pipeline-sentinel-operator --open-browser
```

The `operator` extra intentionally does **not** install Ultralytics. The current production detector is
an optional YOLO provider. To use it, opt in separately:

```powershell
python -m pip install ".\pipeline_sentinel-0.12.0-py3-none-any.whl[operator,yolo]"
```

The optional Ultralytics runtime and its model weights are not covered by Pipeline Sentinel's
Apache-2.0 license; their upstream license terms remain in force. See `THIRD_PARTY_NOTICES.md`.

DINOv2 remains a separate optional extra because anomaly scoring is disabled in the shipped
production profile until a fitted normal-reference artifact is supplied.

Model weights and datasets are not vendored into the release artifact. The configured detector model
may need to be obtained by its upstream runtime on first use.

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

## Production run from the CLI

The CLI remains useful for automation and development:

```powershell
uv sync --extra yolo --group dev
uv run pipeline-sentinel validate-config .\config\production.yaml

uv run pipeline-sentinel run .\input.mp4 `
  --config .\config\production.yaml
```

A completed production run contains:

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

v0.8 extracted the reusable DINOv2 notebook idea without welding the application to DINOv2:

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

One high anomaly score remains evidence rather than an alert. Persistence rules can convert repeated
anomaly evidence into a `visual_anomaly` event before alert policy is applied.

See `docs/anomaly-scoring.md`.

## Notebook 07 -> EO/IR late fusion

v0.9 promotes semantic-event late fusion while making real-sensor assumptions explicit:

```text
EO run -> Event[] ----+
                      |
IR run -> Event[] ----+--> TemporalConsensusFuser --> fused Event[] --> AlertPolicy
                      |
other sensor Event[] -+
```

The CLI fusion path remains available:

```powershell
uv run pipeline-sentinel fuse-runs `
  .\outputs\runs\<eo-run-id> `
  .\outputs\runs\<ir-run-id> `
  --config .\config\fusion.yaml
```

Fusion produces:

```text
fusion_events.csv
fusion_alerts.csv
fusion_contributors.csv
fusion_manifest.json
effective_fusion_config.json
```

The default strategy requires matching event types from distinct sensors within a configured time
window. Label agreement is configurable. Pixel-space IoU is optional and must only be enabled when
the source products are registered into a common geometry.

Fused confidence remains unset because independent EO/IR detector confidence and anomaly evidence are
not assumed to be calibrated onto one probability scale. Original values remain available in
`fusion_contributors.csv`.

See `docs/sensor-fusion.md`.

## Operator API

The web console uses the same documented HTTP API available at `/docs`:

```text
GET  /api/health
GET  /api/jobs
GET  /api/jobs/{job_id}
POST /api/jobs/media
POST /api/local-sources/inspect
POST /api/jobs/local-sequence
POST /api/jobs/fusion
GET  /api/jobs/{job_id}/alerts
GET  /api/jobs/{job_id}/events
GET  /api/jobs/{job_id}/artifacts
GET  /api/jobs/{job_id}/artifacts/{artifact_name}
```

`POST /api/jobs/run` remains available as the backward-compatible encoded-video upload endpoint.

The default worker count is one. This protects a workstation from accidentally starting several
GPU-heavy model stacks concurrently merely because an operator clicked a button several times.

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
fusion, operator-job orchestration, HTTP routes, policy behavior, operational artifacts, release
metadata, package building, clean-wheel installation, and Windows workstation compatibility with
deterministic fixtures.

The release-candidate manual test sequence is in `docs/mvp-acceptance.md`.

## Runtime contracts

- `FrameRecord` — persisted ETL/provenance record.
- `FrameContext` — in-memory frame/pixels passed through runtime components.
- `Detection` — framework-neutral per-frame object observation.
- `Track` — observation associated with a stable runtime track ID.
- `AnomalyObservation` — track appearance scored relative to a normal reference.
- `Event` — semantic/temporal evidence derived from tracks, anomaly persistence, or fusion.
- `Alert` — human-facing notification promoted from an event by policy.

See `docs/architecture.md`.

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
source distribution, generates SHA-256 checksums, installs the wheel into clean environments,
smoke-tests packaged configuration/CLIs/UI assets/operator dependencies, and only then creates a
GitHub Release. Normal CI also exercises the deterministic test suite and clean operator-wheel install
on Windows.

See `docs/release.md`.

## Repository map

```text
pipeline-sentinel/
├── .github/workflows/       CI and tag-driven releases
├── benchmarks/              evaluation definitions and documentation
├── config/                  production and fusion profiles
├── data/                    local staging; large data ignored
├── docs/                    architecture, migration, testing, operational notes
├── LICENSE                  Apache License 2.0 for Pipeline Sentinel
├── THIRD_PARTY_NOTICES.md   external dependency/model licensing notes
├── notebooks/learning/      preserved R&D / instructional work
├── outputs/                 generated artifacts; ignored
├── scripts/                 developer/release/reference utilities
├── src/pipeline_sentinel/   shipped analytics + service package
├── start-operator.cmd       Windows source-checkout launcher
└── tests/                   deterministic automated tests
```

Useful documentation:

- `docs/mvp-acceptance.md` — v0.12 release-candidate and external tester checklist.
- `docs/operator-console.md` — local service, UI, job workspace, API, and security boundary.
- `docs/architecture.md` — runtime boundaries and contracts.
- `docs/production-run.md` — config-driven single-sensor execution and provenance.
- `docs/anomaly-scoring.md` — normal-reference and DINOv2 adapter boundary.
- `docs/sensor-fusion.md` — EO/IR late-fusion contract and assumptions.
- `docs/release.md` — release gates, installation, checksums, and rollback.
- `docs/migration-plan.md` — notebook-to-application extraction map.
- `docs/testing.md` — CI and workstation acceptance.
- `docs/visdrone.md` — aerial validation workflow.

## MVP boundaries and development direction

v0.12 is meant to be a shippable **testing MVP**, not the final operator platform. The next source
boundary should be first-class live RTSP/USB/network-camera ingestion using the existing lazy
`FrameContext` contract. Other post-MVP work includes retention controls, richer run inspection,
browser-compatible video delivery across more codecs, authentication before remote use, and possibly
a desktop wrapper if one-click workstation launch is worth formalizing.

Generic image-folder ingest does not perform raw/radiometric thermal calibration, and unusual
proprietary camera containers/codecs may still fail at the underlying decode layer. See
`docs/mvp-acceptance.md` for the explicit test boundary and known limitations.

## License and external runtimes

Pipeline Sentinel's own source code is licensed under the Apache License 2.0. Optional third-party
model runtimes, weights, and datasets retain their upstream licenses. In particular, the standard
`operator` extra intentionally excludes Ultralytics; the `yolo` extra is an explicit opt-in provider
for users who choose to accept its applicable upstream terms.

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
