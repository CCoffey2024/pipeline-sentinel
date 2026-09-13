# VisDrone benchmark

VisDrone2019-VID is Pipeline Sentinel's first repeatable aerial-video benchmark source.

Dataset bytes are **not** stored in this repository. The runtime reads a local train/val root
supplied on the CLI.

## Current workstation paths

```text
D:\FMV\VisDrone\VisDrone2019-VID-train
D:\FMV\VisDrone\VisDrone2019-VID-val
```

## Validation run

```powershell
uv sync --extra yolo --group dev

uv run pipeline-sentinel visdrone-info `
  "D:\FMV\VisDrone\VisDrone2019-VID-val"

uv run pipeline-sentinel run-visdrone `
  "D:\FMV\VisDrone\VisDrone2019-VID-val" `
  --output outputs\visdrone-acceptance `
  --max-frames 300 `
  --model yolo26n.pt `
  --conf 0.25 `
  --device cpu
```

The selected sequence produces a model prediction table and a normalized ground-truth table under
`outputs/visdrone-acceptance/val/<sequence-id>/`. Ground truth is scoped to exactly the frames
selected by `--frame-step` and `--max-frames`.

The run manifest also records the model name, confidence threshold, NMS IoU, image size, device, and
optional class filter so the benchmark can preserve the inference conditions that produced the
predictions.

## Shared-class evaluator

Run the deterministic evaluator on one saved VisDrone run directory:

```powershell
uv run pipeline-sentinel evaluate-visdrone `
  "outputs\visdrone-acceptance\val\uav0000086_00000_v" `
  --iou-threshold 0.50
```

The evaluator writes:

```text
benchmark/
├── benchmark_summary.json
├── class_metrics.csv
├── matches.csv
├── excluded_predictions.csv
└── excluded_ground_truth.csv
```

`matches.csv` contains the frame-by-frame TP/FP/FN evidence used to calculate the summary. This is
important for debugging model behavior rather than treating precision and recall as magic numbers.

## Shared ontology v1

Runtime labels are never silently renamed. Mapping happens only in the benchmark layer.

| Model / COCO label | Shared class | VisDrone label(s) |
| --- | --- | --- |
| `person` | `person` | `pedestrian`, `people` |
| `bicycle` | `bicycle` | `bicycle` |
| `motorcycle` | `motor` | `motor` |
| `car` | `car` | `car` |
| `truck` | `truck` | `truck` |
| `bus` | `bus` | `bus` |

The human mapping deliberately collapses VisDrone's `pedestrian` and `people` categories into the
single COCO `person` class. That loses some source-dataset semantics, so the mapping is versioned as
`pipeline-sentinel-visdrone-shared-v1` and is recorded in the benchmark summary.

Model predictions such as `sports ball` and VisDrone categories such as `van`, `tricycle`, and
`awning_tricycle` are outside this shared ontology. They are reported separately instead of being
silently forced into an unrelated class.

## Metric policy

The v1 evaluator performs confidence-greedy, one-to-one matching independently for each frame and
shared class. A prediction is a true positive when its best unmatched ground-truth box reaches the
configured IoU threshold; otherwise it is a false positive. Unmatched ground-truth boxes become
false negatives.

It reports per-class and micro-aggregate precision, recall, F1, TP, FP, FN, and mean IoU of matched
true positives.

This benchmark is intentionally called the **Pipeline Sentinel shared-class IoU benchmark**. It is
**not** the official VisDrone AP/mAP evaluator. In v1, ground-truth rows marked ignored are removed
from scoring, but predictions overlapping ignored regions are not neutralized. Metrics are computed
at one fixed IoU threshold rather than sweeping confidence to calculate AP. Those distinctions stay
explicit so we do not overstate what the numbers mean.

## Evaluation policy

Use validation sequences for zero-shot/pretrained-model comparisons. Use training sequences only for
fitting, fine-tuning, calibration, hard-negative mining, or other training-time analysis.

## Why this comes before UAVDT in the current project

Both are useful aerial datasets, but VisDrone is now locally available with train/validation
annotations and gives Pipeline Sentinel a clean first target for dataset discovery, image-sequence
execution, saved predictions, and benchmark plumbing. UAVDT remains valuable as a later independent
cross-dataset/domain-shift check.
