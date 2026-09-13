from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import pandas as pd

from .manifest import validate_frame_manifest
from .types import FrameRecord, Modality


class VideoIngestAdapter(Protocol):
    def extract(
        self,
        video_path: Path,
        frames_dir: Path,
        *,
        sensor_id: str,
        modality: Modality,
    ) -> pd.DataFrame: ...


class OpenCVVideoIngestAdapter:
    """Decode video through OpenCV and emit the canonical frame manifest."""

    def __init__(self, sample_every_n: int = 1, jpeg_quality: int = 95) -> None:
        if sample_every_n < 1:
            raise ValueError("sample_every_n must be >= 1")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.sample_every_n = sample_every_n
        self.jpeg_quality = jpeg_quality

    def extract(
        self,
        video_path: Path,
        frames_dir: Path,
        *,
        sensor_id: str,
        modality: Modality,
    ) -> pd.DataFrame:
        video_path = Path(video_path).resolve()
        frames_dir = Path(frames_dir).resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            cap.release()
            raise RuntimeError(f"Video reports invalid FPS ({fps}): {video_path}")

        frames_dir.mkdir(parents=True, exist_ok=True)
        records: list[FrameRecord] = []
        frame_number = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_number % self.sample_every_n == 0:
                    height, width = frame.shape[:2]
                    frame_id = f"{video_path.stem}_{frame_number:06d}"
                    image_path = frames_dir / f"{frame_id}.jpg"
                    wrote = cv2.imwrite(
                        str(image_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                    if not wrote:
                        raise RuntimeError(f"OpenCV failed to write frame: {image_path}")

                    records.append(
                        FrameRecord(
                            frame_id=frame_id,
                            frame_number=frame_number,
                            timestamp_s=frame_number / fps,
                            image_path=image_path,
                            width=width,
                            height=height,
                            sensor_id=sensor_id,
                            modality=modality,
                            source_path=video_path,
                        )
                    )
                frame_number += 1
        finally:
            cap.release()

        if not records:
            raise RuntimeError(f"Video opened but yielded no sampled frames: {video_path}")

        manifest = pd.DataFrame(record.to_dict() for record in records)
        validate_frame_manifest(manifest)
        return manifest
