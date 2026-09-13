from pathlib import Path

import pandas as pd
import pytest

from pipeline_sentinel.manifest import ManifestValidationError, validate_frame_manifest


def _valid_manifest(tmp_path: Path) -> pd.DataFrame:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"test")
    return pd.DataFrame(
        [
            {
                "frame_id": "f000",
                "frame_number": 0,
                "timestamp_s": 0.0,
                "image_path": str(frame),
                "width": 320,
                "height": 180,
                "sensor_id": "EO_TEST",
                "modality": "EO",
                "source_path": "demo.mp4",
            }
        ]
    )


def test_manifest_accepts_valid_contract(tmp_path: Path) -> None:
    validate_frame_manifest(_valid_manifest(tmp_path))


def test_manifest_rejects_missing_required_column(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path).drop(columns=["sensor_id"])

    with pytest.raises(ManifestValidationError, match="missing required columns"):
        validate_frame_manifest(manifest)


def test_manifest_rejects_missing_frame_file(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    manifest.loc[0, "image_path"] = str(tmp_path / "does_not_exist.jpg")

    with pytest.raises(ManifestValidationError, match="missing frame files"):
        validate_frame_manifest(manifest)
