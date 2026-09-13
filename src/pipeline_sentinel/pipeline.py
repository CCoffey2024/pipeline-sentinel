from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from .detectors import Detector

NORMAL_ROLES = {"normal_maintenance"}


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    annotated_video: Path
    alerts_csv: Path
    run_manifest: Path


class PipelineSentinel:
    """Thin orchestration layer around interchangeable pipeline components."""

    def __init__(self, detector: Detector) -> None:
        self.detector = detector

    def run_video(self, video_path: Path, output_dir: Path) -> RunArtifacts:
        video_path = Path(video_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Video metadata is invalid.")

        annotated_video = output_dir / "annotated_video.mp4"
        writer = cv2.VideoWriter(
            str(annotated_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open video writer: {annotated_video}")

        alert_rows: list[dict[str, object]] = []
        frame_number = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                for detection in self.detector.detect(frame_number):
                    is_alert = detection.scenario_role not in NORMAL_ROLES
                    x1, y1, x2, y2 = detection.xyxy
                    box_value = (255, 255, 255) if is_alert else (180, 180, 180)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_value, 3 if is_alert else 2)
                    text = f"{detection.label} {detection.scenario_role or ''}".strip()
                    cv2.putText(
                        frame,
                        text,
                        (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        box_value,
                        1,
                        cv2.LINE_AA,
                    )

                    if is_alert:
                        alert_rows.append({
                            "frame_number": frame_number,
                            "timestamp_s": frame_number / fps,
                            "label": detection.label,
                            "object_id": detection.object_id,
                            "scenario_role": detection.scenario_role,
                            "confidence": detection.confidence,
                            "detector": detection.source,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        })

                writer.write(frame)
                frame_number += 1
        finally:
            cap.release()
            writer.release()

        alerts_csv = output_dir / "alerts.csv"
        pd.DataFrame(alert_rows).to_csv(alerts_csv, index=False)

        run_manifest = output_dir / "run_manifest.json"
        run_manifest.write_text(
            json.dumps({
                "pipeline_sentinel_version": "0.1.0",
                "input_video": str(video_path),
                "detector_backend": self.detector.name,
                "frames_processed": frame_number,
                "alerts_emitted": len(alert_rows),
                "artifacts": {
                    "annotated_video": str(annotated_video),
                    "alerts_csv": str(alerts_csv),
                },
            }, indent=2),
            encoding="utf-8",
        )

        return RunArtifacts(annotated_video, alerts_csv, run_manifest)
