# Anomaly scoring

Pipeline Sentinel v0.8 promotes the stable idea from Notebook 06 into the application runtime without welding the package to DINOv2.

## Runtime boundary

```text
FrameContext
    -> Detector
    -> Detection[]
    -> Tracker
    -> Track[]
    -> AnomalyAnalyzer
    -> AnomalyObservation[]
    -> AnomalyEventDetector
    -> Event[]
    -> AlertPolicy
    -> Alert[]
```

A visual anomaly score is evidence, not an alert. The default anomaly event rule requires repeated anomalous observations for the same track before producing a `visual_anomaly` event. The normal alert policy then decides whether that event is human-facing.

## What came from Notebook 06

The notebook used DINOv2 as a representation engine rather than a detector:

1. curate examples of normal activity;
2. encode the image crops into embeddings;
3. compute the centroid of the normal embedding cloud;
4. score an observation by cosine distance from that centroid;
5. use a high quantile of normal-reference scores as the anomaly threshold.

The production code preserves that method while separating the representation backend from the scoring and event logic.

## Contracts

`Embedder` accepts image crops and returns a two-dimensional NumPy embedding matrix. Framework-specific tensors and model objects remain inside the adapter.

`AnomalyReference` is the persisted normal-activity model. It stores the embedder identity, centroid, fitted threshold, normal sample count, quantile, and reference-build metadata.

`TrackCropAnomalyAnalyzer` crops tracked objects from the current frame, embeds eligible crops, scores them against the reference centroid, and emits `AnomalyObservation` records.

`ConsecutiveAnomalyEventDetector` converts persistent anomalous evidence into a semantic event. This keeps the rule `anomaly observation != event != alert` explicit.

## Optional DINOv2 runtime

DINOv2 is an optional heavyweight dependency:

```powershell
uv sync --extra yolo --extra dinov2 --group dev
```

The standard installation does not import Torch. The `DinoV2Embedder` loads Torch lazily and loads the requested DINOv2 model through Torch Hub on first use. First use therefore requires network access or an existing Torch Hub cache.

xFormers is not required by Pipeline Sentinel's adapter contract.

## Build a normal reference

Start with a curated directory of **normal** object crops. These are training/calibration examples for the anomaly reference, not test examples.

```text
references/normal-person-crops/
├── crop-0001.jpg
├── crop-0002.jpg
└── ...
```

Build the reference artifact:

```powershell
uv run --extra dinov2 pipeline-sentinel-fit-reference `
  .\references\normal-person-crops `
  --output .\references\person-dinov2-vits14.npz `
  --model dinov2_vits14 `
  --device cpu `
  --quantile 0.95
```

The builder recursively discovers supported images, fails on unreadable inputs, embeds them in batches, fits the normal centroid, and records a SHA-256 digest over the source crop collection in the artifact metadata.

A real deployment should curate normal examples across relevant sensor geometries, seasons, illumination, weather, authorized object types, and operating conditions. A reference that represents only one narrow condition will treat ordinary domain variation as anomalous.

## Enable runtime anomaly scoring

Anomaly scoring is disabled in the shipped production profile. To enable it, point the configuration at a fitted reference:

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

Relative `reference_path` values are resolved relative to the YAML configuration file, not the process working directory.

The embedder identity saved in the reference artifact must match the configured runtime embedder. This prevents accidentally scoring DINOv2 vectors against a reference produced by a different representation.

## Output

Every run now writes `anomalies.csv`, even when anomaly scoring is disabled. When enabled, each row records:

```text
frame_number
timestamp_s
track_id
label
score
threshold
margin
is_anomaly
source
sensor_id
modality
source_path
metadata
```

The run manifest separately reports anomaly observations and the number flagged as anomalous. Event and alert artifacts remain separate.

## Interpretation

The score is cosine distance from the learned normal centroid. A higher score means the crop is less similar to the fitted normal reference in embedding space. It does **not** identify what the object is or why it is unusual.

For example, an anomalous person crop might differ because of pose, equipment, viewpoint, lighting, occlusion, sensor artifacts, or truly unusual activity. This is why anomaly evidence is passed through temporal event logic rather than automatically becoming an alert.

## Evaluation discipline

Normal-reference crops used to fit the centroid or threshold must not be reused as evidence that the anomaly model generalizes. Hold out separate normal and anomalous/probe observations for evaluation.

Useful future evaluation measures include false-positive rate on held-out normal activity, detection rate on defined probe conditions, score distributions by sensor/season/domain, and time-to-event for persistence-gated anomaly events.

The DINOv2 backend is an adapter, not the anomaly architecture itself. A future embedding model should be able to replace it without changing tracking, event, alert, logging, or output contracts.
