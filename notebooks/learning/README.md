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
| 01 | ETL, frame contracts, OpenCV ingest, manifests | preserved here |
| 02 | Classical CV baseline | remains in the local learning workspace; add when reconciled |
| 03 | HOG/SVM → MobileNet lightweight classification | repaired version preserved here |
| 04 | Object detection / detector adapter / optional YOLO | remains in the local learning workspace; next extraction target |
| 05 | Tracking and events | remains in the local learning workspace |
| 06 | DINOv2 representations and anomaly scoring | fixed version preserved here |
| 07 | EO + IR sensor fusion | remains in the local learning workspace |
| 08 | Model bakeoff | remains in the local learning workspace |
| 09 | End-to-end demo | remains in the local learning workspace; orchestration is moving to the CLI |

Only notebooks actually available in the project archive have been committed. We do **not** create
placeholder `.ipynb` files for missing local notebooks because that would make the repository look
more complete than it is.

## Preserved versions

The preserved Notebook 03 is the restart-safe repaired version that includes the MobileNet output
inspection step. Notebook 06 is the repaired DINOv2 anomaly-detection lesson. Notebook 01 is the
ETL lesson from which the initial application contracts were extracted.

## Rules for future notebooks

A learning notebook may:

- contain plots, teaching prose, experimental parameters, and sanity-check displays;
- compare alternative models or methods;
- use small convenience cells for exploration;
- import production components from `pipeline_sentinel`.

A learning notebook should not become the only location of:

- a stable data contract;
- runtime orchestration;
- framework adapters used by the application;
- reusable validation logic;
- deployment configuration.

Once a notebook component is promoted into the package, the notebook should import it rather than
maintain a second copy indefinitely.

## Running notebooks

From the repository root:

```powershell
uv sync --group dev
uv run jupyter lab
```

Jupyter is not currently a core runtime dependency. If you want the repo itself to manage the
notebook environment, we can add a separate optional `notebooks` dependency group later rather than
forcing Jupyter onto application users.

## Data policy

The notebooks may reference local synthetic or external benchmark data. Raw datasets, generated
video, model weights, caches, and benchmark image bytes are excluded from Git. The notebooks should
be able to explain where data comes from without turning the repository into a dataset mirror.
