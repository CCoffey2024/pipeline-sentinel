# UAVDT benchmark staging

UAVDT is the planned **domain-relevant aerial/FMV benchmark** after the generic COCO sanity check.

This directory intentionally contains documentation only for now. Dataset bytes should live under
ignored local paths such as:

```text
benchmarks/uavdt/source/
benchmarks/uavdt/subset/
```

When the detector adapter is ready, the benchmark layer should normalize UAVDT annotations into the
same evaluation contract used for other detector backends rather than embedding UAVDT-specific
assumptions into `PipelineSentinel` itself.

The key purpose of this benchmark is to expose the domain shift that COCO cannot: small targets,
aerial viewpoint, frame-to-frame correlation, camera motion, compression, occlusion, and variable
target pixel footprint.
