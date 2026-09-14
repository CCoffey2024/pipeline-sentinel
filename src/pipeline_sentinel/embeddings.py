from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import cv2
import numpy as np


class Embedder(Protocol):
    """Turn image crops into framework-neutral feature vectors."""

    name: str

    def encode(self, images: Sequence[np.ndarray]) -> np.ndarray: ...


class DinoV2DependencyError(RuntimeError):
    """Raised when the optional DINOv2 runtime is unavailable."""


class DinoV2Embedder:
    """Lazy DINOv2 representation adapter.

    The rest of Pipeline Sentinel sees only NumPy embeddings. Torch models, devices, and hub
    objects remain inside this adapter boundary. The model is loaded on first encode so anomaly
    support can remain disabled without importing Torch.
    """

    def __init__(
        self,
        *,
        model_name: str = "dinov2_vits14",
        device: str | int | None = None,
        image_size: int = 224,
        model: Any | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.model_name = model_name
        self.name = f"dinov2:{model_name}"
        self.requested_device = device
        self.image_size = int(image_size)
        self._model = model
        self._torch: Any | None = None
        self.device: str | None = None

    def _ensure_runtime(self) -> None:
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise DinoV2DependencyError(
                    "DINOv2 support requires the optional 'dinov2' dependency: "
                    "uv sync --extra dinov2"
                ) from exc
            self._torch = torch

        torch = self._torch
        if self.device is None:
            requested = self.requested_device
            if requested is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = str(requested)

        if self._model is None:
            try:
                self._model = torch.hub.load(
                    "facebookresearch/dinov2",
                    self.model_name,
                    pretrained=True,
                )
            except Exception as exc:
                raise DinoV2DependencyError(
                    "Could not load DINOv2 model. First use requires network access or an existing "
                    "Torch Hub cache for the requested model."
                ) from exc

        if hasattr(self._model, "eval"):
            self._model.eval()
        if hasattr(self._model, "to"):
            self._model.to(self.device)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("DINOv2 input must be a non-empty numpy array")
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError("DINOv2 input must have shape HxW, HxWx3, or HxWx4")

        resized = cv2.resize(
            rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        array = resized.astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        return np.transpose(array, (2, 0, 1))

    def encode(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_runtime()
        torch = self._torch
        batch_np = np.stack([self._prepare(image) for image in images], axis=0)
        batch = torch.from_numpy(batch_np).to(self.device)
        with torch.no_grad():
            output = self._model(batch)

        if isinstance(output, dict):
            if "x_norm_clstoken" not in output:
                raise RuntimeError("DINOv2 model output did not contain x_norm_clstoken")
            output = output["x_norm_clstoken"]
        elif isinstance(output, (tuple, list)):
            output = output[0]
        if getattr(output, "ndim", None) != 2:
            raise RuntimeError("DINOv2 model output must be a 2-D embedding matrix")
        return output.detach().cpu().numpy().astype(np.float32, copy=False)
