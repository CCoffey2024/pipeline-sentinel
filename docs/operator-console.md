# Operator console and local service

Pipeline Sentinel v0.10 introduced the workstation-oriented service/UI layer. v0.11 extends that
control plane with **read-in-place image-sequence sources** so operators can use large local imagery
collections without uploading, copying, transcoding, or subsetting them merely to satisfy the UI.

## Why a service layer

The runtime remains a Python package. The browser is only a control/evidence surface:

```text
browser operator console
        |
        v
FastAPI service
        |
        +--> persistent job registry
        +--> bounded worker queue
        +--> encoded-video run API
        +--> local image-sequence run API
        +--> multisensor fusion API
        +--> artifact / alert API
        |
        v
existing Pipeline Sentinel package
```

The service does not duplicate detector, tracker, anomaly, event, fusion, or alert logic.

## Start the console

From a source checkout on Windows, double-click `start-operator.cmd`, or run:

```powershell
uv sync --extra operator --group dev
uv run pipeline-sentinel-operator --open-browser
```

The default URL is `http://127.0.0.1:8765/`. OpenAPI documentation is available at `/docs`.

## Source modes

### Encoded video

The v0.10 video form remains available. Browser-selected `.mp4`, `.mov`, `.avi`, `.mkv`, and `.m4v`
inputs are copied into the managed upload workspace because browsers provide their bytes rather than a
trusted server-side filesystem path.

### Local image folder

The **Local Image Sequence** panel can point directly at a directory of ordered image files. Supported
extensions include JPEG, PNG, BMP, and TIFF. The folder is read in place. Source imagery is never
copied into `outputs/operator`.

Generic folders use natural filename ordering and an ordinal runtime frame number. A declared working
FPS supplies timestamps because ordinary still images do not carry reliable sequence acquisition
timing.

### VisDrone2019-VID

Point the console at the existing VisDrone VID dataset root, for example:

```text
D:\FMV\VisDrone\VisDrone2019-VID-val
```

The service discovers the existing `sequences/` directory and populates the sequence selector. It
reuses the first-class VisDrone adapter already present in Pipeline Sentinel; no MP4 conversion is
performed and native image paths/frame numbers remain the source provenance.

### UAVDT

Point the console at an original UAVDT root containing:

```text
UAVDT/
├── UAV-benchmark-M/
│   ├── M0203/
│   │   ├── img000001.jpg
│   │   └── ...
│   └── ...
└── UAV-benchmark-MOTD_v1.0/
    └── GT/                 # optional for runtime discovery
```

The UAVDT adapter discovers sequences under `UAV-benchmark-M`, preserves `imgNNNNNN` frame indices,
and records available GT paths as metadata without parsing/copying those annotations during ordinary
operator inference.

## Data movement and memory policy

Large source media is treated as **immutable external input**, not workspace material.

```text
source disk / dataset
        |
        | decode one selected frame
        v
FrameContext
        |
        v
detector -> tracker -> anomaly/events -> render
        |
        +--> streaming CSV evidence
        +--> annotated output video
```

For local image-sequence runs:

- source JPEG/PNG/TIFF files stay in their original directories;
- Pipeline Sentinel does not create temporary frame copies;
- the runtime decodes one selected frame at a time;
- detector/tracker/event work proceeds before the next source frame is decoded;
- evidence CSV rows are written incrementally instead of accumulating complete run tables in RAM;
- only necessary outputs (evidence/config/status plus the annotated video) are written to the run
  directory;
- `frame_step` and `max_frames` are sampling controls, not preprocessing jobs that create derivative
  image folders.

A sorted list of filenames is materialized briefly for deterministic sequence ordering. That is small
metadata compared with image bytes and avoids loading the image collection itself into memory.

## Local-source API

```text
POST /api/local-sources/inspect
POST /api/jobs/local-sequence
```

Inspection reads directory metadata only. A local-sequence job accepts source type, root path,
sequence ID, sensor identity, modality, working FPS, frame step, and optional frame cap.

The generated run manifest records `imagery_access=read_in_place`, `imagery_copied=false`, source root,
sequence ID, and sampling controls.

## Job/evidence workflow

The job table shows queued, running, completed, and failed work. Selecting a completed run exposes
summary counts, annotated video, normal CSV evidence, effective configuration, run log, status, and
manifest. Completed sensor runs can still be selected for v0.9 temporal-consensus fusion.

The default service uses one analytic worker. This prevents repeated UI clicks from instantiating
multiple heavyweight model stacks concurrently on a workstation.

## Workspace

The managed workspace remains:

```text
outputs/operator/
├── jobs/              persisted operator job records
├── uploads/           browser-uploaded encoded videos only
├── runtime-configs/   validated per-job sensor identity
├── runs/              production evidence
└── fusions/           late-fusion evidence
```

Read-in-place image datasets are **not** mirrored under this tree.

## Security boundary

The console remains a local workstation service, not an internet-facing deployment. It has no user
authentication, authorization, or TLS termination.

- default bind: `127.0.0.1`;
- a non-loopback bind requires explicit `--allow-remote`;
- **local filesystem source APIs are disabled automatically on a non-loopback bind**, even when remote
  binding is explicitly enabled;
- browser-uploaded files remain extension/size constrained;
- artifact download remains constrained to recorded files inside each job output directory.

Allowing read-in-place paths is reasonable for a loopback desktop control plane because the operator
already has local filesystem access. It would be unsafe to expose that capability to remote clients
without authentication and path authorization, so v0.11 does not do so.

## Design rule

```text
UI != analytics
API != analytics
job orchestration != analytics
source adapter != copied dataset
```

A later desktop shell can add native Windows folder pickers without changing the local-source or
analytic contracts introduced here.
