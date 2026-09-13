from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_FRAME_COLUMNS = {
    "frame_id",
    "frame_number",
    "timestamp_s",
    "image_path",
    "width",
    "height",
    "sensor_id",
    "modality",
    "source_path",
}


class ManifestValidationError(ValueError):
    """Raised when a frame manifest violates the Pipeline Sentinel data contract."""


def validate_frame_manifest(manifest: pd.DataFrame, *, check_files: bool = True) -> None:
    """Validate the canonical frame manifest."""

    missing_columns = REQUIRED_FRAME_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ManifestValidationError(
            f"Manifest missing required columns: {sorted(missing_columns)}"
        )
    if manifest.empty:
        raise ManifestValidationError("Manifest contains no frames.")
    if not manifest["frame_id"].is_unique:
        raise ManifestValidationError("frame_id values must be unique.")
    if (manifest["frame_number"] < 0).any():
        raise ManifestValidationError("frame_number cannot be negative.")
    if (manifest["width"] <= 0).any() or (manifest["height"] <= 0).any():
        raise ManifestValidationError("Frame width and height must be positive.")
    if not manifest["timestamp_s"].is_monotonic_increasing:
        raise ManifestValidationError("timestamp_s must be monotonically increasing.")

    if check_files:
        missing = [str(p) for p in manifest["image_path"] if not Path(p).exists()]
        if missing:
            preview = missing[:3]
            raise ManifestValidationError(f"Manifest references missing frame files: {preview}")
