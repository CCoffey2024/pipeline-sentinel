# YOLO adapter extraction

This note documents the first transition from a deterministic software test double to a real learned
model runtime in Pipeline Sentinel.

The important lesson is not YOLO itself. The important lesson is **where YOLO is allowed to exist in
the application**.

## Before v0.2

The v0.1 pipeline had one detector implementation:

```text
frame_number
     |
     v
GroundTruthDetector
     |
     v
Detection[]
```

That was sufficient for software integration testing because a ground-truth CSV can retrieve rows by
frame number alone.

It was not sufficient for a learned detector. A learned detector needs the image pixels.

## Interface evolution

Rather than change every caller to pass several loose arguments, v0.2 introduces one runtime object:

```text
FrameContext
├── frame_number
├── timestamp_s
├── image
├── source_path
├── sensor_id
└── modality
```

The detector contract is now:

```python
class Detector(Protocol):
    name: str

    def detect(self, frame: FrameContext) -> list[Detection]:
        ...
```

Both the reference backend and the learned backend implement that same contract.

The ground-truth implementation uses only `frame.frame_number`. The YOLO implementation uses
`frame.image`. The orchestrator does not need special-case logic for either backend.

## What the YOLO adapter owns

External model runtimes have their own APIs and data structures. Ultralytics returns result objects
containing box objects, confidence tensors, class tensors, and model-specific metadata.

Those objects stop at the adapter boundary.

```text
Ultralytics result
      |
      |  framework-specific
      v
+------------------+
|   YoloDetector   |
+------------------+
      |
      |  Pipeline Sentinel contract
      v
Detection
```

`YoloDetector` is responsible for:

- invoking the external model runtime;
- passing confidence, NMS IoU, image-size, device, and class filters;
- converting tensors/arrays to NumPy-compatible values;
- resolving class IDs to labels;
- clipping boxes to the source frame;
- discarding invalid boxes;
- attaching model/class metadata;
- returning only Pipeline Sentinel `Detection` objects.

Downstream tracking, rendering, event logic, fusion, and alerting should never need to know what an
Ultralytics `Results` object is.

## Why model injection exists

`YoloDetector` accepts an optional `model=` object primarily to make the adapter testable.

Production use:

```python
detector = YoloDetector(model_name="yolo26n.pt")
```

Unit-test use:

```python
detector = YoloDetector(model=fake_model)
```

The fake model exposes a `predict()` method and returns result-like objects. This lets CI test the
translation boundary without:

- installing Torch;
- installing Ultralytics;
- downloading model weights;
- requiring CUDA;
- relying on changing model predictions.

This is dependency injection in a very practical form: the application component receives the thing
it depends on rather than constructing an untestable global runtime inside every method.

## Optional dependency boundary

YOLO is not part of the core install:

```powershell
uv sync --group dev
```

Install it explicitly when needed:

```powershell
uv sync --extra yolo --group dev
```

This means a lightweight ETL or test deployment does not inherit the full learned-model software
stack.

If the extra is missing, requesting `run-yolo` produces a targeted installation message instead of
breaking import of the entire Pipeline Sentinel package.

## Running the adapter

```powershell
uv run pipeline-sentinel run-yolo .\input.mp4 `
  --output outputs\yolo `
  --model yolo26n.pt `
  --conf 0.25 `
  --iou 0.70 `
  --imgsz 640 `
  --device cpu
```

For a CUDA-capable environment the device can be changed without changing application source code.
The model can likewise be changed to another compatible weight file or exported runtime supported by
the external adapter.

## Why YOLO detections do not create alerts yet

The ground-truth integration backend knows synthetic scenario labels such as:

```text
normal_maintenance
intrusion_vehicle
dismount_loiter
```

A generic learned detector does not.

It returns object observations such as:

```text
person
car
truck
```

Automatically converting every detected object into an alert would mix three distinct jobs:

```text
Detection
    -> what object is visible now?

Tracking / event logic
    -> what has that object been doing over time?

Alert policy
    -> does that behavior meet a threshold that an operator should see?
```

v0.2 therefore renders YOLO detections and records detection counts, but only detections carrying
explicit event semantics become alerts. Notebook 05 will give those temporal/event semantics a real
production boundary.

## What this buys us later

A future detector can satisfy exactly the same contract:

```text
YoloDetector
OnnxDetector
TensorRTDetector
OpenVinoDetector
VendorDetector
        |
        v
   Detection[]
```

If we later export a trained model to ONNX or TensorRT, the rest of Pipeline Sentinel should not be
rewritten. We add or replace the adapter.

That is the practical payoff of the "bolt-on adapter" idea: framework replacement is localized
instead of becoming a system-wide refactor.
