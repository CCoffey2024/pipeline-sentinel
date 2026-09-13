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
   independent real-world generic object-detection sanity check

4. VisDrone2019-VID validation
   first repeatable aerial-video benchmark and image-sequence integration

5. UAVDT
   independent aerial/FMV cross-dataset and domain-shift evaluation

6. Mission-specific held-out data
   actual operating-envelope evidence
```

Each layer answers a different question. Passing one layer does not imply passing the next.

In particular, COCO is useful for generic detector sanity checks but it is not an aerial-FMV
benchmark. VisDrone is now the first local aerial validation dataset wired directly into the runtime,
while UAVDT remains useful as an independent cross-dataset check.

## Repository policy

Dataset bytes are not committed to Git. Benchmark directories contain documentation, preparation
scripts, configuration, and small metadata fixtures only.

Current local VisDrone examples:

```text
D:\FMV\VisDrone\VisDrone2019-VID-train
D:\FMV\VisDrone\VisDrone2019-VID-val
```

These paths are documentation examples only and are not package defaults.

Benchmark results that are small, interpretable, and useful for regression history may later be
committed under a dedicated `results/` directory. Large rendered videos and raw prediction dumps
should remain build artifacts.

## Reproducibility

A benchmark run should eventually record at least:

```text
model/backend name
model version or weights hash
source dataset/version
split and sequence IDs
subset or frame-sampling rules
class-mapping policy
runtime device
software version / git commit
confidence and NMS thresholds
latency summary
mission-relevant quality metrics
```

The long-term goal is that Notebook 08 becomes a visualization/analysis client of these saved,
repeatable benchmark outputs instead of being the only place the bakeoff can be reproduced.
