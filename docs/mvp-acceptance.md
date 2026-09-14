# Pipeline Sentinel v0.12 MVP acceptance

This document defines the **software MVP acceptance boundary** for external workstation testing.
It is intentionally narrower than operational validation of any detector, anomaly model, tracker,
or sensor deployment.

## MVP purpose

A tester should be able to install one release artifact, launch the local operator application,
provide ordinary EO/IR media, run the packaged analytic pipeline, inspect job status and interactive
results, review annotated imagery, and retrieve normal Pipeline Sentinel evidence artifacts without
opening a notebook or editing source code.

The MVP is successful when the application is installable, repeatable, understandable, and produces
auditable outputs. It is not a claim of mission-grade model performance.

For the shortest external-test path, start with `TESTER-QUICKSTART.md`.

## Supported operator inputs

### Browser-selected media

The unified **Media files** control accepts:

- one encoded video: `.mp4`, `.mov`, `.avi`, `.mkv`, or `.m4v`;
- one or more still frames: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, or `.tiff`.

A browser submission is one logical run: choose one video **or** an image sequence, not a mixture of
video and still images. Uploaded image frames are naturally ordered by filename.

### Read-in-place local imagery

The **Local folder / dataset** control supports:

- a generic image folder;
- VisDrone2019-VID;
- UAVDT.

This path is preferred for large image collections because source imagery remains in place and is
streamed one frame at a time instead of being copied into the operator workspace.

EO, IR, and OTHER are sensor modalities, not file types. The same ingest path can represent EO or IR
media when the underlying encoded file/image format is supported.

## Install the release candidate on Windows

Python 3.12 on 64-bit Windows is the tested MVP target. Create a clean virtual environment. The
Apache-licensed operator shell is the `operator` extra; this acceptance procedure also opts into the
separate YOLO provider because the current production profile uses it:

```powershell
py -3.12 -m venv .venv-mvp
.\.venv-mvp\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".\pipeline_sentinel-0.12.0-py3-none-any.whl[operator,yolo]"

pipeline-sentinel --version
pipeline-sentinel-operator --version
```

Both commands should report `0.12.0`.

The `operator` extra itself does not install Ultralytics. The `yolo` extra is an explicit optional
provider and remains subject to its upstream license terms. Model weights are not vendored in the
release artifact; the configured model may need to be obtained by its upstream runtime on first use.
See `THIRD_PARTY_NOTICES.md`.

## Launch acceptance

Start the application:

```powershell
pipeline-sentinel-operator --open-browser
```

Expected result:

- browser opens to `http://127.0.0.1:8765/`;
- health indicator becomes green;
- version reports `0.12.0`;
- **Sensor Ingest** offers `Media files` and `Local folder / dataset`;
- job queue initially loads without an error.

The MVP is local-first. Do not expose it directly to an untrusted network. There is no authentication
or TLS termination in this release.

## Test A — encoded video

1. Select **Media files**.
2. Choose one short supported EO or IR video.
3. Set a meaningful Sensor ID and modality.
4. Start analysis.
5. Confirm the job transitions through queued/running to completed or presents a useful failure.
6. Select the completed job.

Expected evidence includes:

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

The generated MP4 may not decode natively in Chrome on every workstation. That is not by itself a
failed run; use the codec-safe annotated-frame player in the completed-run view when necessary.

## Test B — uploaded still-image sequence

1. Select **Media files**.
2. Choose several sequential JPEG/PNG/TIFF frames in one selection.
3. Set the working FPS used to convert frame order to runtime timestamps.
4. Start analysis.
5. Confirm the run completes and frame count is plausible.
6. Review the annotated imagery and interactive results.

The input images should be processed in natural filename order.

## Test C — large local image folder

1. Select **Local folder / dataset**.
2. Choose `Image folder`.
3. Enter an absolute local directory containing supported image frames.
4. Click **Inspect source**.
5. Confirm the reported frame count is plausible.
6. Start analysis.

Expected behavior:

- source images are read in place;
- source files are not copied into `outputs/operator`;
- the run writes only normal run evidence and annotated output;
- sampling controls (`frame_step`, `max_frames`) do not create derivative frame directories.

## Test D — interactive results and annotated playback

For a completed run, verify:

1. **Overview** shows plausible class and detections-per-frame summaries.
2. **Detections**, **Tracks**, **Anomalies**, **Events / Alerts**, and **Downloads** remain selectable and do not reset while the queue continues background polling.
3. The codec-safe annotated-frame browser can step through frames.
4. Play/Pause, Previous/Next, speed controls, Loop, scrubber, and timestamp display work.
5. Bounding boxes/captions use stable non-white per-class colors and remain readable over the imagery.

If native Chrome playback of `annotated_video.mp4` fails, the codec-safe player is the supported
v0.12 in-console review path.

## Test E — persistence and evidence

After at least one completed run:

1. stop the operator service cleanly with `Ctrl+C`;
2. restart it using the same workspace;
3. confirm completed jobs remain visible;
4. confirm completed artifacts and interactive results are still accessible.

If the service is stopped while a job is active, that job should be marked failed on restart rather
than silently appearing completed.

## Test F — safe cleanup

For a finished run:

1. note whether the input was browser-uploaded or read in place;
2. click **Delete run & files**;
3. confirm the job disappears and generated evidence is removed;
4. for browser-uploaded media, confirm the managed uploaded copy is removed;
5. for read-in-place imagery, confirm the original source files remain untouched.

Queued/running jobs must not be deletable.

## Test G — multisensor fusion

If two completed runs have distinct sensor IDs:

1. select both completed runs in **Fuse Completed Runs**;
2. start fusion;
3. confirm a fusion job is created;
4. inspect the fusion artifacts.

Expected fusion outputs include:

```text
fusion_events.csv
fusion_alerts.csv
fusion_contributors.csv
fusion_manifest.json
effective_fusion_config.json
```

Fusion validates the software path. Meaningful EO/IR corroboration still depends on appropriate
sensor timing, event semantics, and deployment-specific assumptions.

## Automated release gates

Before a v0.12 release is tagged, CI must pass:

- release/version consistency checks;
- Ruff lint;
- deterministic pytest suite;
- wheel and source-distribution build;
- SHA-256 manifest generation;
- clean core-wheel installation;
- packaged configuration/UI/source-adapter smoke tests;
- Windows test-suite execution;
- clean Windows installation of the wheel with `[operator]`;
- verification that `[operator]` does not install Ultralytics;
- import/route smoke test of the installed operator service.

The tag-driven release workflow repeats lint/tests/build and clean-installs both the core wheel and
operator extra before creating a GitHub Release.

## Known MVP limitations

These are intentional boundaries, not hidden claims:

- live RTSP/USB/network camera feeds are not yet first-class source adapters;
- the service is local-first and has no user authentication;
- browser-selected media is copied into the managed operator workspace;
- uploaded/local image sequences currently derive runtime time from filename order plus working FPS
  unless a dataset-specific adapter provides stronger timing semantics;
- unusual proprietary camera containers/codecs may not be decodable by the packaged OpenCV/FFmpeg
  runtime even when a similar extension is common;
- native Chrome playback of the generated `mp4v` evidence MP4 is not guaranteed; use the codec-safe
  annotated-frame player for in-console review;
- raw/radiometric thermal calibration is not performed by the generic image-folder adapter;
- DINOv2 anomaly scoring remains disabled by default until a deployment provides a fitted normal
  reference artifact;
- software acceptance does not establish detector/tracker/anomaly accuracy for a particular mission.

## Tester feedback to capture

For every failed or confusing run, record:

- Pipeline Sentinel version;
- Windows version and Python version;
- CPU/GPU and available memory;
- source type, extension/container, approximate resolution, and frame count/duration;
- EO/IR modality and sensor ID used;
- exact operator error message;
- `run_status.json` and `run_log.jsonl` when created;
- whether the failure occurred during ingest, model loading, inference, rendering, playback/review,
  cleanup, or artifact access.

That information is sufficient to separate packaging/runtime defects from model-quality or
source-codec problems during the MVP test cycle.
