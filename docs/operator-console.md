# Operator console and local service

Pipeline Sentinel v0.10 adds a workstation-oriented application-delivery layer on top of the existing
runtime. The computer-vision pipeline remains a Python package; the operator console is a thin local
service that queues runs, exposes status and evidence through HTTP, and serves a bundled browser UI.

## Why a service layer

The runtime already had stable contracts and auditable artifacts, but an operator should not need to
assemble CLI flags or browse output folders manually. The service turns those package APIs into a
small local control plane:

```text
browser operator console
        |
        v
FastAPI service
        |
        +--> persistent job registry
        |
        +--> bounded worker queue
        |
        +--> production run API
        |
        +--> multisensor fusion API
        |
        +--> artifact / alert API
        |
        v
existing Pipeline Sentinel package
```

The service does not duplicate detector, tracker, anomaly, event, fusion, or alert logic. It calls the
same application APIs used by the CLI.

## Install

From a source checkout:

```powershell
uv sync --extra operator --group dev
```

The `operator` extra installs FastAPI, Uvicorn, multipart upload support, and the existing optional
YOLO runtime. DINOv2 remains a separate optional extra because anomaly scoring is disabled in the
shipped production profile until a fitted normal-reference artifact is supplied.

From a release wheel:

```powershell
python -m pip install ".\pipeline_sentinel-0.10.0-py3-none-any.whl[operator]"
```

## Start the console

From the repository on Windows, `start-operator.cmd` is intended to be double-clicked. It synchronizes
the operator dependencies, starts the local service, and opens the default browser.

The equivalent command is:

```powershell
uv run pipeline-sentinel-operator --open-browser
```

The default URL is:

```text
http://127.0.0.1:8765/
```

The OpenAPI interface is available at `/docs`.

## Operator workflow

### Start one sensor run

Choose a local video file in the browser, select `EO`, `IR`, or `OTHER`, give the sensor a stable ID,
and click **Start analysis**.

The browser uploads the source into the managed workspace and queues the normal config-driven
production runtime. The default service uses one analytic worker. This is intentional: a browser
should not be able to start several GPU-heavy model stacks concurrently merely because an operator
double-clicked a button.

### Inspect evidence

The job table shows queued, running, completed, and failed work. Selecting a completed job exposes:

- run summary counts;
- annotated video when the browser can decode the generated MP4;
- detections, tracks, anomaly observations, events, and alerts as normal artifacts;
- effective configuration, run log, status, and manifest;
- a human-facing alert table in the console.

The browser UI never changes the meaning of those artifacts. It is only another consumer of the
runtime evidence contract.

### Fuse EO and IR evidence

Completed sensor runs appear in the **Fuse Completed Runs** panel. Select at least two runs with
distinct sensor IDs and click **Fuse selected runs**. The service calls the same v0.9 late-fusion API
used by `pipeline-sentinel fuse-runs` and creates its own fusion job and artifacts.

## Workspace

The default managed workspace is:

```text
outputs/operator/
├── jobs/              persisted operator job records
├── uploads/           browser-uploaded source videos
├── runtime-configs/   effective sensor-identity config used for each run
├── runs/              production run evidence
└── fusions/           late-fusion evidence
```

Job metadata is persisted as JSON. If the service restarts while a job is queued or running, that job
is marked failed with an `InterruptedJob` error rather than silently pretending it completed.

Uploaded sources are retained in v0.10 so an operator can reconstruct exactly which media produced a
run. Automated retention/cleanup policy belongs in a later deployment milestone.

## Runtime identity overrides

The operator selects `sensor_id` and `modality` in the UI. The service loads the validated production
profile, preserves all detector/tracker/anomaly/event policy, writes a job-specific runtime YAML, and
changes only the runtime sensor identity. Relative anomaly-reference paths are first resolved to an
absolute path so moving the effective YAML into the operator workspace cannot silently change its
meaning.

## API surface

The first API intentionally remains small:

```text
GET  /api/health
GET  /api/jobs
GET  /api/jobs/{job_id}
POST /api/jobs/run                  multipart video + sensor identity
POST /api/jobs/fusion               completed run job IDs
GET  /api/jobs/{job_id}/alerts
GET  /api/jobs/{job_id}/events
GET  /api/jobs/{job_id}/artifacts
GET  /api/jobs/{job_id}/artifacts/{artifact_name}
```

The browser console uses this same public API; there is no private UI-only execution path.

## Security boundary

v0.10 is a **local workstation service**, not an internet-facing deployment. It has no user
authentication, authorization, TLS termination, or multi-tenant isolation. Therefore:

- the default bind is `127.0.0.1`;
- a non-loopback bind is refused unless `--allow-remote` is explicitly supplied;
- arbitrary server filesystem paths cannot be submitted through the HTTP API;
- uploaded files are restricted to supported video extensions;
- artifact download is restricted to paths recorded for that job and inside the job output directory.

An authenticated network service, reverse proxy, role model, and deployment packaging are separate
milestones. Do not expose the v0.10 console directly to an untrusted network.

## Operational controls

Useful launcher options:

```powershell
pipeline-sentinel-operator `
  --workspace D:\PipelineSentinel\operator `
  --port 8765 `
  --max-workers 1 `
  --max-upload-gib 8 `
  --open-browser
```

Increasing `--max-workers` should be a deliberate hardware decision because model runtimes can hold
large CPU/GPU allocations independently.

## Design rule

The operator UI is deliberately downstream of the analytic package:

```text
UI != analytics
API != analytics
job orchestration != analytics
```

A future desktop shell, remote service, or mission-specific UI should be able to replace the v0.10
browser console without changing detector/tracker/anomaly/fusion code.
