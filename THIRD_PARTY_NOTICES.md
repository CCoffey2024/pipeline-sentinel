# Third-party notices

Pipeline Sentinel is licensed under the Apache License 2.0. That license applies to Pipeline Sentinel's own source code and does **not** relicense third-party software, model runtimes, model weights, or datasets.

Pipeline Sentinel depends on or can optionally integrate with the following projects. Their own license terms remain in force.

| Component | Role | Upstream license / note |
| --- | --- | --- |
| NumPy | Core numerical runtime | BSD-3-Clause |
| OpenCV / opencv-python-headless | Image and video I/O/processing | OpenCV and Python packaging include their own permissive and third-party license notices; preserve those notices when redistributing their binaries |
| pandas | Tabular evidence processing | BSD-3-Clause |
| PyYAML | Configuration parsing | MIT |
| FastAPI | Optional operator HTTP service | MIT |
| python-multipart | Optional operator upload parsing | Apache-2.0 |
| Uvicorn | Optional operator ASGI server | BSD-3-Clause |
| PyTorch | Optional DINOv2 runtime | BSD-style license; installed distribution may include additional third-party notices |
| DINOv2 | Optional representation-model provider | Standard DINOv2 code/models used by Pipeline Sentinel are upstream Meta artifacts and retain their own license terms |
| Ultralytics YOLO / `ultralytics-opencv-headless` | Optional detector provider | **Not covered by Pipeline Sentinel's Apache-2.0 license.** The upstream package is offered under Ultralytics' applicable open-source and/or commercial terms. Users who install the `yolo` extra are responsible for complying with those terms. |

## Optional YOLO backend

The default Pipeline Sentinel and `operator` installations do not install Ultralytics. The YOLO integration is an optional provider boundary:

```text
pip install "pipeline-sentinel[yolo]"
```

No Ultralytics model weights are bundled in the Pipeline Sentinel wheel or source distribution.

For development from this repository, `start-operator.cmd` installs both the `operator` and `yolo` extras so the current YOLO-backed test workflow continues to work. That convenience does not change the license of either project.

## Models and datasets

Model weights and datasets are not part of the Pipeline Sentinel Apache-2.0 grant unless explicitly stated otherwise. VisDrone, UAVDT, COCO imagery, DINOv2 weights, Ultralytics weights, and other externally obtained artifacts retain their upstream terms. Pipeline Sentinel release artifacts should not vendor those assets without a separate license review.

This file is an engineering notice, not legal advice. When redistributing third-party binary packages, preserve the license and notice files supplied by those packages.
