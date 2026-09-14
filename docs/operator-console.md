# Operator console and local service

Pipeline Sentinel v0.10 introduced the workstation-oriented service/UI layer. v0.11 added
**read-in-place image-sequence sources**. v0.12 unifies browser media and local imagery behind one
operator-facing **Sensor Ingest** control so file format no longer determines which part of the UI an
operator must use.

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
        +--> unified browser media run API
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

From an installed release wheel:

```powershell
python -m pip install ".\pipeline_sentinel-0.12.0-py3-none-any.whl[operator]"
pipeline-sentinel-operator --open-browser
```

The default URL is `http://127.0.0.1:8765/`. OpenAPI documentation is available at `/docs`.

## Sensor ingest

The console exposes one **Start Analysis Run** card with two source modes. Modality and sensor ID are
shared metadata because EO/IR describes the sensor, not the file container.

### Media files

Browser-selected media accepts one encoded video or one-or-more still images.

Supported video extensions in v0.12 are:

```text
.mp4 .mov .avi .mkv .m4v
```

Supported still-image extensions are:

```text
.jpg .jpeg .png .bmp .tif .tiff
```

A browser submission is one logical run. Do not mix video and image files in the same submission.
Selected still images are copied into one managed upload directory and processed as an ordered image
sequence using natural filename ordering. The working FPS supplies runtime timestamps for generic
still images.

Browser-selected media is copied because browsers provide file bytes rather than a trusted local
server-side path.

### Local image folder

For large image collections, switch to **Local folder / dataset** and choose `Image folder`. Point the
console directly at a directory of ordered image files. The folder is read in place; source imagery
is never copied into `outputs/operator`.

Generic folders use natural filename ordering and an ordinal runtime frame number. A declared working
FPS supplies timestamps because ordinary still images do not carry reliable sequence acquisition
timing.

### VisDrone2019-VID

Point the local source at an existing VisDrone VID dataset root, for example:

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

## Operator API

The primary v0.12 endpoints are:

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

`POST /api/jobs/run` remains as a backward-compatible encoded-video endpoint.

Local-source inspection reads directory metadata only. A local-sequence job accepts source type, root
path, sequence ID, sensor identity, modality, working FPS, frame step, and optional frame cap.

The generated run manifest records `imagery_access=read_in_place`, `imagery_copied=false`, source root,
sequence ID, and sampling controls for read-in-place sources.

## Job/evidence workflow

The job table shows queued, running, completed, and failed work. Selecting a completed run exposes
summary counts, annotated video, normal CSV evidence, effective configuration, run log, status, and
manifest. Completed sensor runs can be selected for temporal-consensus fusion.

The default service uses one analytic worker. This prevents repeated UI clicks from instantiating
multiple heavyweight model stacks concurrently on a workstation.

## Workspace

The managed workspace is:

```text
outputs/operator/
├── jobs/              persisted operator job records
├── uploads/           browser-uploaded media only
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
without authentication and path authorization, so the MVP does not do so.

## MVP boundaries

The v0.12 MVP is intended for controlled workstation testing. Live RTSP/USB/network camera feeds are
not yet first-class source adapters, raw/radiometric thermal calibration is not performed by the
generic image-folder path, and unusual proprietary codecs may still fail at the underlying decode
layer.

See `docs/mvp-acceptance.md` for the release-candidate test procedure and known limitations.

## Design rule

```text
UI != analytics
API != analytics
job orchestration != analytics
source adapter != copied dataset
```

A later desktop shell or live-stream adapter can reuse the same frame/runtime contracts without
changing detector, tracking, event, fusion, or evidence code.
