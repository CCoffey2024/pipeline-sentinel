# VisDrone benchmark

VisDrone2019-VID is Pipeline Sentinel's first repeatable aerial-video benchmark source.

Dataset bytes are **not** stored in this repository. The runtime reads a local train/val root
supplied on the CLI.

## Current workstation paths

```text
D:\FMV\VisDrone\VisDrone2019-VID-train
D:\FMV\VisDrone\VisDrone2019-VID-val
```

## First validation run

```powershell
uv sync --extra yolo --group dev

uv run pipeline-sentinel visdrone-info `
  "D:\FMV\VisDrone\VisDrone2019-VID-val"

uv run pipeline-sentinel run-visdrone `
  "D:\FMV\VisDrone\VisDrone2019-VID-val" `
  --output outputs\visdrone `
  --max-frames 300 `
  --model yolo26n.pt `
  --conf 0.25 `
  --device cpu
```

The selected sequence produces a model prediction table and a normalized ground-truth table under
`outputs/visdrone/val/<sequence-id>/`.

## Evaluation policy

Use validation sequences for zero-shot/pretrained-model comparisons. Use training sequences only for
fitting, fine-tuning, calibration, hard-negative mining, or other training-time analysis.

A future evaluator will version the COCO-to-VisDrone class mapping explicitly rather than silently
renaming model outputs in runtime code.

## Why this comes before UAVDT in the current project

Both are useful aerial datasets, but VisDrone is now locally available with train/validation
annotations and gives Pipeline Sentinel a clean first target for dataset discovery, image-sequence
execution, saved predictions, and benchmark plumbing. UAVDT remains valuable as a later independent
cross-dataset/domain-shift check.
