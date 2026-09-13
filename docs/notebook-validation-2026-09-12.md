# Notebook validation notes — 2026-09-12

The original Pipeline Sentinel learning sequence was validated before application extraction began.

Executed successfully in the build environment:

```text
01 Pipeline Sentinel ETL
02 Classical CV Baseline
03 Lightweight Classifier (HOG/SVM path; MobileNet optional)
04 Object Detection (adapter + ground-truth path; YOLO optional)
05 Tracking and Events
06 DINOv2 Anomaly Detection (HOG control path; DINOv2 optional)
07 EO + IR Sensor Fusion
08 Model Bakeoff scaffold
09 End-to-End Demo
```

The heavier backends were intentionally optional so the CPU baseline could be validated without
surprise model downloads or GPU/toolchain failures:

```text
Notebook 03: RUN_MOBILENET
Notebook 04: RUN_YOLO
Notebook 06: RUN_DINOV2
```

Generated validation artifacts included the deterministic EO/IR source videos, alert CSV output,
and an annotated Pipeline Sentinel demo video.

This historical notebook validation is separate from package/CI validation. As components are
migrated, the package tests become the source of truth for reusable runtime behavior while the
notebooks remain the R&D and instructional record.
