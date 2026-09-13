from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _box(width: int, height: int, cx: float, cy: float, bw: float, bh: float) -> tuple[int, int, int, int]:
    return (
        max(0, int((cx - bw / 2) * width)),
        max(0, int((cy - bh / 2) * height)),
        min(width - 1, int((cx + bw / 2) * width)),
        min(height - 1, int((cy + bh / 2) * height)),
    )


def generate_demo_video(
    video_path: Path,
    ground_truth_path: Path,
    *,
    modality: str = "EO",
    seed: int = 7,
    frame_count: int = 120,
    fps: float = 10.0,
    size: tuple[int, int] = (640, 360),
) -> pd.DataFrame:
    """Generate deterministic EO/IR-like video plus annotations for integration testing."""

    modality = modality.upper()
    if modality not in {"EO", "IR"}:
        raise ValueError("modality must be 'EO' or 'IR'")
    if frame_count < 1 or fps <= 0:
        raise ValueError("frame_count and fps must be positive")

    width, height = size
    video_path = Path(video_path).resolve()
    ground_truth_path = Path(ground_truth_path).resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open synthetic video writer: {video_path}")

    rng = np.random.default_rng(seed + (100 if modality == "IR" else 0))
    rows: list[dict[str, object]] = []

    def add(frame: np.ndarray, i: int, object_id: int, label: str, role: str, box: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = box
        value = (225, 225, 225) if modality == "IR" else (80, 190, 220)
        if label == "aerial_object":
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), max(3, (x2 - x1) // 2), value, -1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), value, -1)
        rows.append({
            "frame_number": i,
            "timestamp_s": i / fps,
            "label": label,
            "object_id": object_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "scenario_role": role,
            "modality": modality,
        })

    try:
        for i in range(frame_count):
            base_value = 40 if modality == "IR" else 80
            frame = np.full((height, width, 3), base_value, dtype=np.uint8)
            noise = rng.normal(0, 4, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            corridor_y = int(height * 0.66)
            cv2.line(frame, (0, corridor_y), (width - 1, corridor_y), (120, 120, 120), 3)

            if 10 <= i < min(50, frame_count):
                p = (i - 10) / 40
                add(frame, i, 1, "vehicle", "normal_maintenance", _box(width, height, 0.10 + 0.34 * p, 0.60, 0.12, 0.10))
            if 55 <= i < min(105, frame_count):
                p = (i - 55) / 50
                add(frame, i, 2, "vehicle", "intrusion_vehicle", _box(width, height, 0.92 - 0.48 * p, 0.71, 0.12, 0.10))
            if 70 <= i < min(116, frame_count):
                add(frame, i, 3, "person", "dismount_loiter", _box(width, height, 0.54, 0.60, 0.035, 0.10))
            if 85 <= i < frame_count:
                p = (i - 85) / max(1, frame_count - 85)
                add(frame, i, 4, "aerial_object", "small_uas_like", _box(width, height, 0.18 + 0.56 * p, 0.20, 0.035, 0.055))
            if 95 <= i < frame_count:
                add(frame, i, 5, "smoke", "smoke_anomaly", _box(width, height, 0.72, 0.55, 0.18, 0.22))

            writer.write(frame)
    finally:
        writer.release()

    gt = pd.DataFrame(rows)
    gt.to_csv(ground_truth_path, index=False)
    return gt
