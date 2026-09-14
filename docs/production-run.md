# Production-style runtime

Pipeline Sentinel uses a single config-driven entry point for repeatable operational runs. The purpose is to stop treating a long list of CLI flags as the runtime configuration and make every run self-describing and auditable.

## Primary command

Install the learned-model runtime required by the configuration, then run one encoded EO/IR video:

```powershell
uv sync --extra yolo --extra dinov2 --group dev

uv run pipeline-sentinel run .\input.mp4 `
  --config .\config\production.yaml
```

The shipped development profile uses YOLO for detection and DINOv2 for track representations. A
detector/tracker-only profile can omit the `dinov2` extra when both representation and anomaly stages
are disabled.

If `--output` is omitted, Pipeline Sentinel creates:

```text
outputs/runs/<UTC-timestamp>-<short-id>/
```

An exact destination can be supplied with `--output`.

## Validate configuration without inference

```powershell
uv run pipeline-sentinel validate-config .\config\production.yaml
```

Validation rejects unknown keys as well as invalid probabilities, image sizes, tracker settings, anomaly settings, severities, modalities, and unsupported backends. This is deliberate: a misspelled production setting should fail loudly instead of silently falling back to a default.

## Production profile

`config/production.yaml` is the version-controlled runtime profile. It currently selects:

```text
YOLO detector
  -> yolo26n.pt
  -> confidence 0.25
  -> NMS IoU 0.70
  -> inference size 960

IoU tracker
  -> association IoU 0.30
  -> expire after 2 missed updates

DINOv2 representations
  -> ViT-S/14 over sampled mature track crops
  -> every 15 hits, bounded to 16 crops per frame
  -> descriptive evidence, not anomaly labels

DINOv2 anomaly scoring
  -> disabled by default
  -> requires a fitted normal-reference .npz artifact when enabled

Dwell event rule
  -> disabled by default

Severity alert policy
  -> warning or higher
```

The 960 inference size is intentional. In the current VisDrone validation slice, 960 materially improved person recall over 640 while avoiding the larger precision penalty observed at 1280. That benchmark result is evidence for the current default, not a claim that 960 is universally optimal.

Dwell and anomaly-event policies remain conservative by default because persistence thresholds and definitions of normal activity are deployment policy, not safe universal inference defaults.

## Representation configuration

The representation stage is independent of anomaly scoring and needs no fitted normal reference:

```yaml
representation:
  enabled: true
  backend: dinov2
  model: dinov2_vits14
  device: null
  labels: [person, car, truck, bus, motorcycle, bicycle]
  min_track_hits: 3
  sample_every_n_hits: 15
  max_per_frame: 16
  pad_px: 6
  min_crop_size: 12
```

See `docs/representations.md` for workload controls and vector interpretation.

## Anomaly configuration

Notebook 06 is promoted behind an optional runtime boundary. An example enabled configuration is:

```yaml
anomaly:
  enabled: true
  backend: dinov2
  reference_path: ../references/person-dinov2-vits14.npz
  model: dinov2_vits14
  device: cpu
  labels:
    - person
  min_track_hits: 3
  pad_px: 6
  min_crop_size: 12
  event_min_consecutive: 3
  event_severity: warning
```

The configured reference must have been fitted with the same embedder identity. Relative reference paths resolve relative to the YAML file. See `docs/anomaly-scoring.md` for reference fitting and interpretation.

## Run artifact contract

A completed production-style run contains:

```text
annotated_video.mp4
detections.csv
tracks.csv
representations.csv
representation_embeddings.f32
representation_manifest.json
anomalies.csv
events.csv
alerts.csv
run_manifest.json
effective_config.json
run_log.jsonl
run_status.json
```

The representation CSV indexes normalized vectors stored row-major in the float32 data file. Its JSON
manifest records the vector shape and storage contract. `anomalies.csv` is always present and is empty
when anomaly scoring is disabled. When enabled, it records scored track observations separately from
semantic events and human-facing alerts.

`effective_config.json` is the parsed configuration actually used by the application plus the SHA-256 of the source YAML. `run_manifest.json` contains model/runtime output counts and embeds the run ID, configuration provenance, Python version, executable, and platform metadata.

`run_log.jsonl` is an append-only machine-readable lifecycle log. A normal run contains `run_started` followed by `run_completed`. A failed run records `run_failed` with exception type and message.

`run_status.json` is the simple orchestration status record. It is written as `running` before model inference begins and is rewritten as `completed` or `failed`. This means a failed production run still leaves useful operational evidence even when normal model artifacts could not be completed.

## Why this boundary matters

The lower-level commands remain useful for development and benchmarking:

```text
run-yolo
run-visdrone
evaluate-visdrone
demo
pipeline-sentinel-fit-reference
```

The `run` command is different. It is the stable application-facing entry point. Detector,
representation, tracker, anomaly, event, alert, sensor, and modality choices come from validated
configuration rather than requiring an operator to reconstruct a long experimental command line.

The intended progression is:

```text
notebook experiment
    -> component contract
    -> deterministic tests
    -> config-driven runtime
    -> auditable run artifacts
    -> packaged/released application
```
