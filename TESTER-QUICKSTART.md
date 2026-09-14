# Pipeline Sentinel v0.12.0 — External Tester Quick Start

Pipeline Sentinel v0.12.0 is a **local workstation testing MVP**. It is intended to let an external tester install the packaged application, run ordinary video or image-sequence inputs, inspect the annotated output and interactive results, and report packaging/runtime problems.

It is not a claim of mission-grade model performance.

## Recommended test environment

- Windows 10 or Windows 11
- Python 3.12 (64-bit)
- Chrome or another modern Chromium-based browser
- Internet access for the initial Python package install and, when using the optional YOLO provider, any upstream model-weight retrieval required by that runtime

Pipeline Sentinel itself is Apache-2.0 licensed. The optional Ultralytics YOLO provider and its model weights retain their own upstream license terms. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## 1. Create a clean environment

Open PowerShell in the folder containing the release wheel and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Install Pipeline Sentinel

The current production profile uses the optional YOLO provider, so external acceptance testing should install both the operator shell and YOLO extra:

```powershell
python -m pip install ".\pipeline_sentinel-0.12.0-py3-none-any.whl[operator,yolo]"
```

Confirm the package version:

```powershell
pipeline-sentinel --version
pipeline-sentinel-operator --version
```

Both should report `0.12.0`.

## 3. Start the Operator Console

```powershell
pipeline-sentinel-operator --open-browser
```

The console should open at:

```text
http://127.0.0.1:8765/
```

The health indicator should turn green.

The service is intentionally local-first. Do not expose this MVP directly to an untrusted network.

## 4. Run a simple test

Use **Sensor Ingest** and choose either:

- **Media files** — one encoded video, or one-or-more still images; or
- **Local folder / dataset** — a local image folder, VisDrone sequence, or UAVDT sequence read in place.

For a first test, use a short video or a folder with roughly 50–300 sequential JPEG images.

Set a Sensor ID, choose EO / IR / OTHER as appropriate, and start the analysis.

A healthy run should transition:

```text
queued -> running -> completed
```

## 5. Review the output

Select the completed run. The initial dashboard should provide:

- frame, detection, track, representation, anomaly, event, and alert counts;
- detections by class;
- detection density across frames;
- interactive Detections, Tracks, Anomalies, Events / Alerts, and Downloads tabs;
- an annotated imagery review surface;
- run storage information and **Delete run & files** cleanup.

### Annotated playback

The generated evidence MP4 currently uses a codec that Chrome may not decode directly on every workstation. This is a known MVP limitation, not a failed analytic run.

When native browser video playback is unavailable, use **Browse annotated frames (codec-safe)**. The built-in player supports:

- Play / Pause
- Previous / Next frame
- 0.25x / 0.5x / 1x / 2x speed
- Loop
- frame scrubber
- current / total timestamp

This player uses Pipeline Sentinel's own frame decoder and JPEG delivery, so it remains usable even when Chrome cannot play the MP4 container's video codec.

Bounding boxes and captions use stable per-class colors to make object classes easier to distinguish during review.

## 6. Stop cleanly

Close the browser tab, then return to the terminal running Pipeline Sentinel and press:

```text
Ctrl+C
```

If Windows asks:

```text
Terminate batch job (Y/N)?
```

answer `Y` and press Enter.

## 7. What to report

For anything that fails or feels confusing, please capture:

- Pipeline Sentinel version
- Windows version
- Python version
- CPU/GPU and available memory
- source type and file extension/container
- approximate resolution and frame count/duration
- exact error message shown in the Operator Console or terminal
- whether the problem occurred during ingest, model loading, inference, playback/review, or cleanup
- `run_status.json` and `run_log.jsonl` from the run when available

Feedback on usability is just as useful as outright failures: confusing controls, slow interactions, hard-to-read annotations, unclear errors, or surprising behavior are all worth reporting.

## Known MVP boundaries

This release intentionally does not yet provide first-class RTSP/USB/network-camera ingest, authentication for remote multi-user deployment, radiometric thermal calibration for generic image folders, or guaranteed decoding of proprietary camera codecs.

DINOv2 track representations are enabled in the development profile and may add first-run model
download and inference time. DINOv2 anomaly scoring and mission-specific dwell-event rules remain
disabled until deployment-specific reference data and policy thresholds are supplied.

For the complete acceptance procedure, see `docs/mvp-acceptance.md` in the source repository.
