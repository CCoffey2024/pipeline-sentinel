# DINOv2 representation evidence

Pipeline Sentinel v0.13 keeps YOLO as the fast first-stage detector and uses DINOv2 as an optional
second-stage representation model. DINOv2 does not replace detection, tracking, anomaly policy, or
alerting.

```text
frame -> YOLO detections -> IoU tracks -> sampled track crops -> DINOv2 embeddings
```

## Why this is a separate stage

YOLO answers **what and where** for each frame. DINOv2 produces a semantic appearance vector for a
tracked crop. Keeping those responsibilities separate lets Pipeline Sentinel use the vectors later
for retrieval, clustering, re-identification experiments, reference fitting, or change analysis
without treating every visual difference as an operational anomaly.

`representation_change` is therefore descriptive evidence only. It is `1 - cosine_similarity`
between the current vector and the preceding sampled vector for the same runtime track. It is not an
anomaly score, event, or alert.

## Workload controls

The production profile enables `dinov2_vits14` with bounded sampling:

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

- `min_track_hits` avoids spending ViT work on unstable one-frame tracks.
- `sample_every_n_hits` controls how often each mature track is embedded.
- `max_per_frame` bounds a crowded frame's DINOv2 batch.
- `labels` limits representation work to useful object classes; omit it to allow every class.
- `device: null` selects CUDA when Torch reports it available, otherwise CPU.

The crop adapter accepts grayscale, BGR, or BGRA imagery and letterboxes each crop to the model input
without changing its aspect ratio. The first DINOv2 run may need network access to populate the
Torch Hub model cache. Later runs can use that local cache.

## Evidence artifacts

Every run creates three representation artifacts, even when the stage is disabled or emits no rows:

```text
representations.csv
representation_embeddings.f32
representation_manifest.json
```

`representations.csv` is the searchable row index. It includes `embedding_row`, frame/time, track,
class, crop bounds, track maturity, vector dimension, vector norm, and similarity/change relative to
the track's preceding sample.

`representation_embeddings.f32` stores L2-normalized vectors as contiguous little-endian float32
rows. If the manifest reports dimension `D`, vector row `R` begins at byte offset `R * D * 4`.

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

run = Path("outputs/run")
index = pd.read_csv(run / "representations.csv")
manifest = json.loads((run / "representation_manifest.json").read_text())
dimension = manifest["embedding_dimension"]
vectors = np.fromfile(run / "representation_embeddings.f32", dtype="<f4")
vectors = vectors.reshape(len(index), dimension)
```

The manifest records whether the stage was enabled, analyzer/embedder identity, effective sampling
and crop configuration, observation and represented-track counts, embedding dimension/dtype/order,
and the associated index/data paths. The same counts appear in `run_manifest.json` and the Operator
Console summary.

## Operator behavior

The Sensor Ingest card enables **DINOv2 track representations** by default. Clear the checkbox for a
detector/tracker-only speed run. Completed jobs expose a Representations metric and tab, while the
raw index, vectors, and manifest remain available from Downloads.

Representation inference adds work after YOLO and tracking. Runtime cost depends on device, object
count, crop cadence, and `max_per_frame`; it is deliberately not performed for every detection in
every frame.

## Relationship to anomaly scoring

Anomaly scoring remains separately configured and requires a fitted normal-reference artifact. A
representation-only run needs no reference. If both stages use the same DINOv2 model/device, the
runtime reuses the loaded embedder instance, while each analyzer retains its own sampling and
semantic contract.
