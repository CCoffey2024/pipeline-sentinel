# Pipeline Sentinel benchmarks

Benchmarks are deliberately separate from the runtime package.

The purpose of this directory is to answer **model-quality and domain-generalization questions**
without making benchmark datasets or provider SDKs dependencies of the application itself.

## Evaluation ladder

```text
1. Unit tests
   software contracts and failure handling

2. Synthetic Pipeline Sentinel demo
   deterministic end-to-end integration

3. COCO subset
   independent real-world generic object detection

4. UAVDT / aerial-FMV benchmark
   domain-relevant aerial video evaluation

5. Mission-specific held-out data
   actual operating-envelope evidence
```

Each layer answers a different question. Passing one layer does not imply passing the next.

In particular, COCO is useful for generic detector sanity checks but it is not an aerial-FMV
benchmark. A detector can perform well on ordinary ground-level photographs and still fail badly on
small, oblique, compressed, motion-blurred targets in airborne video.

## Repository policy

Dataset bytes are not committed to Git. Benchmark directories contain documentation, preparation
scripts, configuration, and small metadata fixtures only.

Downloaded/exported datasets should live in ignored local paths such as:

```text
benchmarks/coco/source/
benchmarks/coco/subset/
benchmarks/uavdt/source/
```

Benchmark results that are small, interpretable, and useful for regression history may later be
committed under a dedicated `results/` directory. Large rendered videos and raw prediction dumps
should remain build artifacts.

## Reproducibility

A benchmark run should eventually record at least:

```text
model/backend name
model version or weights hash
source dataset/version
subset seed / selection rules
class mapping
runtime device
software version / git commit
confidence and NMS thresholds
latency summary
mission-relevant quality metrics
```

The long-term goal is that Notebook 08 becomes a visualization/analysis client of these saved,
repeatable benchmark outputs instead of being the only place the bakeoff can be reproduced.
