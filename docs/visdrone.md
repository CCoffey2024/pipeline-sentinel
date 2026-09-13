# VisDrone integration

Pipeline Sentinel uses VisDrone2019-VID as its first repeatable aerial-video dataset integration.
The goal is to replace ad hoc local-video testing with a named dataset whose image sequences and
annotations can be revisited, benchmarked, and compared across model versions.

## Why VisDrone belongs behind a dataset adapter

VisDrone VID is not distributed as one MP4 per clip. Each clip is an image directory under
`sequences/`, with a matching text annotation file under `annotations/`.

That means the application should not force the dataset into an encoded-video abstraction merely to
reuse OpenCV `VideoCapture`. Instead:

```text
VisDrone sequence directory
        |
        v
VisDroneSequence.iter_frames()
        |
        v
FrameContext stream
        |
        v
PipelineSentinel.run_frames()
```

The detector sees the same `FrameContext` contract whether the pixels came from an MP4 or a JPEG
sequence.

## Local dataset roots

Current workstation examples:

```text
D:\FMV\VisDrone\VisDrone2019-VID-train
D:\FMV\VisDrone\VisDrone2019-VID-val
```

These are documentation examples, not package defaults. Machine-specific absolute paths and dataset
bytes stay outside Git.

## Expected directory structure

```text
VisDrone2019-VID-val/
├── annotations/
│   ├── uav...txt
│   └── ...
└── sequences/
    ├── uav.../
    │   ├── 0000001.jpg
    │   ├── 0000002.jpg
    │   └── ...
    └── ...
```

`VisDroneDataset` validates the `sequences/` directory and associates annotations by matching the
sequence directory name to `<sequence-id>.txt`.

## Native VID annotation format

Pipeline Sentinel parses ten fields per row:

```text
frame_index,
target_id,
bbox_left,
bbox_top,
bbox_width,
bbox_height,
score,
object_category,
truncation,
occlusion
```

The runtime normalizes the box to XYXY:

```text
x1 = bbox_left
y1 = bbox_top
x2 = bbox_left + bbox_width
y2 = bbox_top + bbox_height
```

and maps VisDrone category IDs to the native labels:

```text
0  ignored_region
1  pedestrian
2  people
3  bicycle
4  car
5  van
6  truck
7  tricycle
8  awning_tricycle
9  bus
10 motor
11 others
```

Ignored-region rows and rows with non-positive evaluation score are marked `ignored=True` in the
normalized ground truth.

## Train versus validation

The current workflow intentionally uses the validation split first:

```text
train -> fitting / fine-tuning / training-time analysis
val   -> zero-shot and held-out development evaluation
```

Once we fine-tune a model on VisDrone train, train performance must not be presented as independent
evidence of generalization.

## Timing

The dataset supplies image frames, not an encoded container with recoverable FPS metadata. Pipeline
Sentinel therefore requires a declared working cadence through `--fps`.

That cadence is used for:

- generated timestamps;
- the annotated MP4 frame rate;
- later time-based algorithms that need an explicit working assumption.

The run manifest records a timing note so a benchmark result does not silently imply that the chosen
FPS was recovered from the source dataset.

## Run artifacts

A VisDrone sequence run produces:

```text
annotated_video.mp4
detections.csv
ground_truth.csv
alerts.csv
run_manifest.json
```

`detections.csv` and `ground_truth.csv` are deliberately separate. Runtime model output should never
overwrite or masquerade as annotation truth.

## COCO versus VisDrone classes

A generic pretrained YOLO model typically uses the COCO class vocabulary. VisDrone uses a different
vocabulary. Some classes align closely (`car`, `truck`, `bus`, `bicycle`), some need translation
(`motorcycle` versus VisDrone `motor`), and some are semantically ambiguous (`person` versus
VisDrone's separate `pedestrian` and `people`).

The runtime therefore preserves both vocabularies. Formal class mapping belongs in benchmark code so
we can version and audit the evaluation decision rather than hiding it inside `YoloDetector`.

## Next benchmark step

The next VisDrone-specific engineering step is a repeatable evaluator that consumes:

```text
detections.csv + ground_truth.csv + class-mapping policy
```

and reports at minimum:

```text
IoU-matched precision / recall
per-class counts
false positives / false negatives
small-object slices
occlusion / truncation slices
latency or throughput metadata
```

Later we can add official-style AP/mAP evaluation once the class mapping and task definition are
explicit.
