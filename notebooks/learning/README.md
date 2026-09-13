# Pipeline Sentinel learning notebooks

These notebooks are the **R&D / instructional record** that preceded the installable application.
They are intentionally preserved rather than rewritten into production-style scripts.

Their role is changing:

```text
before extraction
notebook = implementation + experiment + explanation

after extraction
package  = reusable implementation
notebook = experiment + explanation + visualization
```

The application must never depend on a notebook having been executed first.

## Notebook inventory

| Notebook | Topic | Repository status |
|---|---|---|
| 01 | ETL, frame contracts, OpenCV ingest, manifests | historical learning snapshot preserved here; v0.1 package extraction complete |
| 02 | Classical CV baseline | remains in the local learning workspace; add when reconciled |
| 03 | HOG/SVM → MobileNet lightweight classification | repaired code path preserved as a repository-cleaned snapshot |
| 04 | Object detection / detector adapter / optional YOLO | local notebook not yet archived here; **v0.2 runtime extraction complete** |
| 05 | Tracking and events | remains in the local learning workspace; next production extraction target |
| 06 | DINOv2 representations and anomaly scoring | fixed historical snapshot preserved here |
| 07 | EO + IR sensor fusion | remains in the local learning workspace |
| 08 | Model bakeoff | remains in the local learning workspace |
| 09 | End-to-end demo | remains in the local learning workspace; orchestration is moving to the CLI |

Only notebooks actually available in the project archive have been committed. We do **not** create
placeholder `.ipynb` files for missing local notebooks because that would make the repository look
more complete than it is.

A notebook can therefore be "extracted" into production code before its historical `.ipynb` file is
archived here. Notebook 04 is currently the example: its durable detector-adapter idea now lives in
the package even though the local teaching notebook itself has not been copied into this repository.

## Preserved versions

Notebook 01 is the ETL lesson from which the initial application contracts were extracted.

Notebook 03 preserves the repaired restart-safe HOG/SVM and MobileNet code path, including raw
ImageNet prediction inspection and the frozen-feature linear probe. Its repository copy trims some
redundant teaching/plot cells so the durable learning path is easier to review in source control.

Notebook 06 is the repaired DINOv2 anomaly-detection lesson with the HOG control experiment and
optional DINOv2 backend.

These are **learning snapshots**, not the canonical runtime implementation. Some historical imports
still reflect the pre-package folder layout. As each lesson is migrated, the notebook should be
reconciled to import `pipeline_sentinel` package components instead of preserving duplicate runtime
logic.

## Notebook 04 production counterpart

The production concepts learned in Notebook 04 now live in:

```text
src/pipeline_sentinel/types.py       FrameContext + Detection contracts
src/pipeline_sentinel/detectors.py   Detector protocol + GroundTruthDetector
src/pipeline_sentinel/yolo.py        optional YoloDetector adapter
src/pipeline_sentinel/pipeline.py    backend-neutral orchestration
```

See `docs/yolo-adapter.md` for the engineering walkthrough.

## Rules for future notebooks

A learning notebook may contain plots, teaching prose, experimental parameters, sanity-check
displays, and model comparisons. It should increasingly import production components from
`pipeline_sentinel`.

A learning notebook should not become the only location of a stable data contract, runtime
orchestration, framework adapter, reusable validator, or deployment configuration.

Once a notebook component is promoted into the package, the notebook should import it rather than
maintain a second copy indefinitely.

## Notebook environment

Jupyter is intentionally **not** a core application dependency. The production CLI should not
install a notebook server just to run Pipeline Sentinel.

For now, use the existing learning environment for these snapshots. When we resume active notebook
work inside this repository, the clean next step is a separate optional `notebooks` dependency group
containing Jupyter, matplotlib, scikit-learn, scikit-image, Torch/torchvision, and other teaching
requirements.

## Data policy

The notebooks may reference local synthetic or external benchmark data. Raw datasets, generated
video, model weights, caches, and benchmark image bytes are excluded from Git. The notebooks should
be able to explain where data comes from without turning the repository into a dataset mirror.
