# Production-style runtime

Pipeline Sentinel v0.6 adds a single config-driven entry point for repeatable operational runs. The
purpose is to stop treating a long list of CLI flags as the runtime configuration and make every run
self-describing and auditable.

## Primary command

Install the optional learned-model runtime, then run one encoded EO/IR video:

```powershell
uv sync --extra yolo --group dev

uv run pipeline-sentinel run .\input.mp4 `
  --config .\config\production.yaml
```

If `--output` is omitted, Pipeline Sentinel creates:

```text
outputs/runs/<UTC-timestamp>-<short-id>/
```

An exact destination can be supplied with `--output`.

## Validate configuration without inference

```powershell
uv run pipeline-sentinel validate-config .\config\production.yaml
```

Validation rejects unknown keys as well as invalid probabilities, image sizes, tracker settings,
severities, modalities, and unsupported backends. This is deliberate: a misspelled production
setting should fail loudly instead of silently falling back to a default.

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

Dwell event rule
  -> disabled by default

Severity alert policy
  -> warning or higher
```

The 960 inference size is intentional. In the current VisDrone validation slice, 960 materially
improved person recall over 640 while avoiding the larger precision penalty observed at 1280. That
benchmark result is evidence for the current default, not a claim that 960 is universally optimal.

Dwell events remain disabled by default because persistence thresholds are deployment policy, not a
safe universal inference default.

## Run artifact contract

A completed production-style run contains:

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

`effective_config.json` is the parsed configuration actually used by the application plus the SHA-256
of the source YAML. `run_manifest.json` contains model/runtime output counts and embeds the run ID,
configuration provenance, Python version, executable, and platform metadata.

`run_log.jsonl` is an append-only machine-readable lifecycle log. A normal run contains
`run_started` followed by `run_completed`. A failed run records `run_failed` with exception type and
message.

`run_status.json` is the simple orchestration status record. It is written as `running` before model
inference begins and is rewritten as `completed` or `failed`. This means a failed production run still
leaves useful operational evidence even when normal model artifacts could not be completed.

## Why this boundary matters

The existing lower-level commands remain useful for development and benchmarking:

```text
run-yolo
run-visdrone
evaluate-visdrone
demo
```

The `run` command is different. It is the stable application-facing entry point. Model, tracker,
event, alert, sensor, and modality choices come from validated configuration rather than requiring an
operator to reconstruct a long experimental command line.

The intended progression is now:

```text
notebook experiment
    -> component contract
    -> deterministic tests
    -> config-driven runtime
    -> auditable run artifacts
    -> packaged/released application
```
