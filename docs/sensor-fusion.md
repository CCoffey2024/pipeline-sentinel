# EO/IR late sensor fusion

Pipeline Sentinel v0.9 promotes the stable concept from the Notebook 07 learning step into a
shippable application boundary: **independent sensor processing followed by late fusion of normalized
semantic evidence**.

The learning progression described Notebook 07 as aligned synthetic EO/IR streams plus late fusion.
The production implementation deliberately keeps the reusable idea and removes the synthetic-only
assumption that two image planes always share identical pixels.

## Architecture

```text
EO stream                              IR stream
   |                                      |
   v                                      v
Detector / tracker / anomaly          Detector / tracker / anomaly
   |                                      |
   v                                      v
Event[]                                Event[]
   |                                      |
   +---------------+  +------------------+
                   |  |
                   v  v
             TemporalConsensusFuser
                   |
                   v
              fused Event[]
                   |
                   v
              AlertPolicy
                   |
                   v
              fused Alert[]
```

This is **late fusion**. Pipeline Sentinel does not blend EO and IR pixels, concatenate neural
features, share Torch/Ultralytics provider objects, or pretend that model confidence values from two
sensors are automatically calibrated onto the same probability scale.

## Why fuse events instead of raw pixels?

Raw-pixel and feature fusion require stronger sensor-specific assumptions: registration, calibration,
field-of-view transforms, resampling policy, radiometry, timing synchronization, and often
model-specific tensors. Those concerns should not leak through the rest of the application.

Semantic events are already portable application evidence. Fusing them preserves the existing rule:

```text
detection != track != anomaly score != event != alert
```

The fuser therefore corroborates evidence; it does not create a new detector.

## Input contract

`fuse-runs` consumes **completed Pipeline Sentinel run directories**. Each input must contain:

```text
run_manifest.json
events.csv
```

`tracks.csv` is also read when available so event-linked track boxes can support optional spatial
matching.

Each input manifest must provide a distinct `sensor_id`. Typical EO/IR input manifests might contain:

```text
sensor_id: EO_CAM_01    modality: EO
sensor_id: IR_CAM_01    modality: IR
```

Using the same sensor ID twice is rejected because sensor consensus cannot be established by counting
two files from the same logical source as two independent sensors.

## Matching policy

The default `temporal_consensus` strategy requires:

- the same semantic `event_type`;
- distinct sensor IDs;
- timestamps within `max_time_delta_s`;
- at least `min_sensors` independent sensors;
- equal labels when `require_matching_label: true`.

The default profile is:

```yaml
strategy: temporal_consensus
max_time_delta_s: 0.50
min_sensors: 2
require_matching_label: true
spatial_iou_threshold: null
alerts:
  minimum_severity: warning
```

A contributor event is consumed at most once by the deterministic greedy matcher. This keeps the
output auditable and prevents one event from artificially corroborating several fused events.

## Spatial agreement is explicit

The synthetic learning streams were aligned, but real EO and IR products may not be. Pipeline
Sentinel therefore makes pixel-space agreement opt-in.

When:

```yaml
spatial_iou_threshold: null
```

fusion uses temporal + semantic evidence only. No pixel-registration claim is made.

When a numeric threshold is configured, for example:

```yaml
spatial_iou_threshold: 0.30
```

the contributing events must have track boxes at their event frames and those boxes must satisfy the
configured IoU threshold. **Only enable this when the sensor products are known to be registered into
a common pixel geometry.**

Future adapters can add calibrated coordinate transforms or world-coordinate association behind the
same fusion boundary without changing downstream alert policy.

## Run it

Validate the fusion profile:

```powershell
uv run pipeline-sentinel validate-fusion-config .\config\fusion.yaml
```

Fuse two completed runs:

```powershell
uv run pipeline-sentinel fuse-runs `
  .\outputs\runs\<eo-run-id> `
  .\outputs\runs\<ir-run-id> `
  --config .\config\fusion.yaml
```

An exact destination can be supplied with `--output`. Otherwise a unique directory is created under
`outputs/fusion/`.

The command accepts more than two run directories as well; `min_sensors` controls how many distinct
sources must corroborate an event.

## Fusion artifacts

A completed fusion produces:

```text
fusion_events.csv
fusion_alerts.csv
fusion_contributors.csv
fusion_manifest.json
effective_fusion_config.json
```

`fusion_events.csv` is the normalized fused semantic evidence. `fusion_alerts.csv` contains only
fused events promoted by the standard severity policy.

`fusion_contributors.csv` is the audit trail. For every fused event it records the contributing run,
sensor, modality, original event ID, timestamp, track ID, label, confidence, optional track box, and
time offset from the fused event timestamp.

`fusion_manifest.json` records the input run manifests and hashes, strategy, sensor count, evidence
counts, configuration provenance, and the assumptions the fuser did and did not make.

## Confidence policy

Pipeline Sentinel intentionally sets the fused event's `confidence` to `null` in v0.9.

A YOLO confidence from EO, a DINOv2 distance-derived anomaly, and an IR-model confidence are not
necessarily calibrated measurements of the same quantity. Averaging or multiplying them would look
scientific while making an unsupported probability claim. The original contributor confidences remain
in the audit artifact and metadata.

A future calibrated fusion policy can be added behind the same contract once calibration evidence
exists.

## What v0.9 does not claim

- temporal proximity alone does not prove physical identity;
- spatial IoU is valid only for registered products;
- event consensus does not repair poor single-sensor detectors;
- fused severity is policy evidence, not a probability;
- this is not pixel-level or feature-level EO/IR fusion.

The purpose of this milestone is to make multisensor corroboration explicit, replaceable, testable,
and auditable without welding the application to one registration library or one model family.
