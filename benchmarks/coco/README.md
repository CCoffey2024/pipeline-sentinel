# COCO benchmark staging

Pipeline Sentinel uses COCO as a **generic object-detection sanity benchmark**, not as evidence of
aerial/FMV mission performance.

A convenient source is the Microsoft COCO export hosted on Roboflow Universe:

`https://universe.roboflow.com/microsoft/coco`

You may also use an official/local COCO-format export. The preparation script is intentionally
provider-neutral and only expects standard COCO JSON plus the referenced image files.

## Recommended first subset

The initial Pipeline Sentinel subset focuses on classes that overlap our detector-learning use case:

```text
person
car
truck
bus
motorcycle
bicycle
airplane
```

The default maximum is 2,000 images with deterministic seed `42`. That is large enough to catch
obvious detector problems without turning every development iteration into a full COCO benchmark.

## Expected local layout

The source data itself is ignored by Git:

```text
benchmarks/coco/
├── README.md
├── source/
│   ├── annotations/
│   │   └── instances_val2017.json
│   └── images/
└── subset/
```

The exact source directories may differ. Pass their paths explicitly to the script.

## Build the subset

Example from PowerShell:

```powershell
uv run python scripts/prepare_coco_subset.py `
  benchmarks/coco/source/annotations/instances_val2017.json `
  benchmarks/coco/source/images `
  benchmarks/coco/subset `
  --max-images 2000 `
  --seed 42
```

The output is:

```text
benchmarks/coco/subset/
├── annotations.json
├── subset_manifest.json
└── images/
```

`subset_manifest.json` records the requested categories, seed, selected image count, and annotation
count so a result can be traced back to the preparation choices.

## Why a subset instead of the whole dataset?

During application development we want a repeatable regression set that runs quickly enough to use
often. Full benchmark runs can be scheduled later when a detector candidate is ready for serious
comparison.

## Important interpretation rule

COCO answers something like:

> Can this detector recognize ordinary instances of these generic object classes on independent
> real imagery?

It does **not** answer:

> Will this detector perform reliably on EO/IR airborne FMV at the target size, altitude, sensor
> geometry, compression, weather, and illumination relevant to Pipeline Sentinel?

That second question requires aerial/FMV data such as UAVDT and, ultimately, properly held-out
mission-representative data.
