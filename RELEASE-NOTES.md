# Pipeline Sentinel v0.13.0

Pipeline Sentinel v0.13.0 is a local-workstation external testing release for EO/IR video and
ordered imagery. It is intended to test installation, ingest, model execution, evidence generation,
and operator usability. It is not a claim of mission-grade model performance.

## Highlights

- Unified browser-selected video and still-image ingestion.
- Read-in-place image folders, VisDrone2019-VID, and UAVDT sequences.
- Mixed image sizes and orientations normalized by aspect-preserving letterboxing.
- Ultralytics YOLO as the fast first-stage detector.
- Sampled DINOv2 ViT-S/14 representations over mature tracked-object crops.
- Interactive detections, tracks, representations, anomalies, events/alerts, and downloads.
- Codec-safe annotated-frame review when Chrome cannot decode the evidence MP4.
- Streamed CSV/JSON evidence plus compact little-endian float32 representation vectors.

Representation change is descriptive within-track cosine-distance evidence. It is not an anomaly
score, event, or alert. Anomaly scoring remains disabled until a deployment supplies a fitted normal
reference.

## Windows installation

Download the wheel, `SHA256SUMS.txt`, and `TESTER-QUICKSTART.md` from this release. In a clean Python
3.12 environment, install the complete external-test profile:

```powershell
python -m pip install ".\pipeline_sentinel-0.13.0-py3-none-any.whl[operator,yolo,dinov2]"
pipeline-sentinel-operator --open-browser
```

The initial installation and first run require internet access for dependencies and upstream model
weights. DINOv2 uses CUDA when available through PyTorch and otherwise runs on CPU.

## Acceptance evidence

The v0.13 pipeline completed real-media tests on the primary Lenovo workstation using both browser
file selection and Local Folder ingestion, including mixed-size imagery and the YOLO-to-DINOv2
second stage. Automated tests use deterministic model doubles and do not download model weights.

## Known boundaries

- The service binds to loopback and has no authentication; do not expose it to an untrusted network.
- Live RTSP/USB/network cameras are not yet first-class source adapters.
- DINOv2 can be substantially slower on CPU-only machines.
- Model weights and datasets are not included in the release.
- Native Chrome playback of the generated MP4 is not guaranteed; use the codec-safe frame player.
- Generic image-folder ingest does not perform radiometric thermal calibration.

See `TESTER-QUICKSTART.md`, `LICENSE`, and `THIRD_PARTY_NOTICES.md` before testing.
