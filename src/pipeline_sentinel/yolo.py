from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .types import Detection, FrameContext


class YoloDependencyError(RuntimeError):
    """Raised when the optional Ultralytics runtime is requested but unavailable."""


def _to_numpy(value: Any) -> np.ndarray:
    """Convert common tensor/array objects to NumPy without importing a framework."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _class_name(names: Mapping[int, str] | Sequence[str] | None, class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, class_id))
    if names is not None and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


class YoloDetector:
    """Ultralytics YOLO adapter that emits Pipeline Sentinel ``Detection`` objects.

    Ultralytics is imported lazily so the core package remains usable without Torch or model
    runtimes. Tests can inject a model-like object through ``model`` and exercise the adapter without
    installing Ultralytics or downloading weights.
    """

    name = "yolo"

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        *,
        confidence: float = 0.25,
        iou: float = 0.70,
        imgsz: int = 640,
        device: str | None = None,
        class_ids: Sequence[int] | None = None,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if class_ids is not None and any(class_id < 0 for class_id in class_ids):
            raise ValueError("class_ids cannot contain negative values")

        self.model_name = model_name
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        self.class_ids = tuple(class_ids) if class_ids is not None else None

        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise YoloDependencyError(
                    "The YOLO backend is optional. Install it with "
                    "`uv sync --extra yolo --group dev`."
                ) from exc
            model = YOLO(model_name)

        self.model = model

    def detect(self, frame: FrameContext) -> list[Detection]:
        predict_kwargs: dict[str, Any] = {
            "source": frame.image,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device
        if self.class_ids is not None:
            predict_kwargs["classes"] = list(self.class_ids)

        results = self.model.predict(**predict_kwargs)
        detections: list[Detection] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy = _to_numpy(boxes.xyxy)
            confidences = _to_numpy(boxes.conf).reshape(-1)
            class_ids = _to_numpy(boxes.cls).reshape(-1)
            names = getattr(result, "names", None)
            if names is None:
                names = getattr(self.model, "names", None)

            for coords, confidence, raw_class_id in zip(
                xyxy,
                confidences,
                class_ids,
                strict=True,
            ):
                class_id = int(raw_class_id)
                x1 = max(0, min(frame.width - 1, int(round(float(coords[0])))))
                y1 = max(0, min(frame.height - 1, int(round(float(coords[1])))))
                x2 = max(0, min(frame.width - 1, int(round(float(coords[2])))))
                y2 = max(0, min(frame.height - 1, int(round(float(coords[3])))))
                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    Detection(
                        frame_number=frame.frame_number,
                        label=_class_name(names, class_id),
                        confidence=float(confidence),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        source=self.name,
                        metadata={
                            "class_id": class_id,
                            "model": self.model_name,
                        },
                    )
                )

        return detections
