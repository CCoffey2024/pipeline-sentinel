from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from . import __version__
from .detectors import Detector
from .types import FrameContext, Modality

NORMAL_ROLES = {"normal_maintenance"}
DETECTION_COLUMNS = [
    "frame_number",
    "timestamp_s",
    "label",
    "object_id",
    "scenario_role",
    "confidence",
    "detector",
    "sensor_id",
    "modality",
    "source_path",
    "x1",
    "y1",
    "x2",
    "y2",
]
ALERT_COLUMNS = DETECTION_COLUMNS.copy()


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    annotated_video: Path
    detections_csv: Path
    alerts_csv: Path
    run_manifest: Path


class PipelineSentinel:
    """Thin orchestration layer around interchangeable pipeline components."""

    def __init__(self, detector: Detector) -> None:
        self.detector = detector

    def run_frames(
        self,
        frames: Iterable[FrameContext],
        output_dir: Path,
        *,
        render_fps: float = 30.0,
        source_name: str | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> RunArtifacts:
        """Run the pipeline over any ordered stream of ``FrameContext`` objects.

        This is the common runtime used by both encoded videos and image-sequence datasets such as
        VisDrone. ``render_fps`` controls only the generated annotated MP4; it does not claim to be
        original acquisition timing unless the source provides that timing.
        """

        if render_fps <= 0:
            raise ValueError("render_fps must be positive")

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        iterator = iter(frames)
        first = next(iterator, None)
        if first is None:
            raise ValueError("frame source yielded zero frames")

        width, height = first.width, first.height
        annotated_video = output_dir / "annotated_video.mp4"
        writer = cv2.VideoWriter(
            str(annotated_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(render_fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {annotated_video}")

        detection_rows: list[dict[str, object]] = []
        alert_rows: list[dict[str, object]] = []
        frames_processed = 0
        first_frame_number = first.frame_number
        last_frame_number = first.frame_number

        try:
            for frame in chain([first], iterator):
                if frame.width != width or frame.height != height:
                    raise ValueError(
                        "all frames in one run must have the same dimensions; "
                        f"expected {width}x{height}, got {frame.width}x{frame.height}"
                    )

                rendered = frame.image.copy()
                for detection in self.detector.detect(frame):
                    role = detection.scenario_role
                    is_alert = bool(role) and role not in NORMAL_ROLES
                    x1, y1, x2, y2 = detection.xyxy
                    box_value = (255, 255, 255) if is_alert else (180, 180, 180)
                    cv2.rectangle(rendered, (x1, y1), (x2, y2), box_value, 3 if is_alert else 2)

                    text = f"{detection.label} {detection.confidence:.2f}"
                    if role:
                        text = f"{text} {role}"
                    cv2.putText(
                        rendered,
                        text,
                        (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        box_value,
                        1,
                        cv2.LINE_AA,
                    )

                    row = {
                        "frame_number": detection.frame_number,
                        "timestamp_s": frame.timestamp_s,
                        "label": detection.label,
                        "object_id": detection.object_id,
                        "scenario_role": role,
                        "confidence": detection.confidence,
                        "detector": detection.source,
                        "sensor_id": frame.sensor_id,
                        "modality": frame.modality,
                        "source_path": str(frame.source_path) if frame.source_path else None,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                    detection_rows.append(row)
                    if is_alert:
                        alert_rows.append(row.copy())

                writer.write(rendered)
                frames_processed += 1
                last_frame_number = frame.frame_number
        finally:
            writer.release()

        detections_csv = output_dir / "detections.csv"
        pd.DataFrame(detection_rows, columns=DETECTION_COLUMNS).to_csv(detections_csv, index=False)

        alerts_csv = output_dir / "alerts.csv"
        pd.DataFrame(alert_rows, columns=ALERT_COLUMNS).to_csv(alerts_csv, index=False)

        metadata = dict(run_metadata or {})
        run_manifest = output_dir / "run_manifest.json"
        run_manifest.write_text(
            json.dumps(
                {
                    "pipeline_sentinel_version": __version__,
                    "input_source": source_name
                    or (str(first.source_path) if first.source_path else "frame_sequence"),
                    "sensor_id": first.sensor_id,
                    "modality": first.modality,
                    "detector_backend": self.detector.name,
                    "frames_processed": frames_processed,
                    "first_frame_number": first_frame_number,
                    "last_frame_number": last_frame_number,
                    "detections_emitted": len(detection_rows),
                    "alerts_emitted": len(alert_rows),
                    "render_fps": float(render_fps),
                    "metadata": metadata,
                    "artifacts": {
                        "annotated_video": str(annotated_video),
                        "detections_csv": str(detections_csv),
                        "alerts_csv": str(alerts_csv),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return RunArtifacts(annotated_video, detections_csv, alerts_csv, run_manifest)

    def run_video(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        sensor_id: str = "VIDEO_01",
        modality: Modality = "EO",
    ) -> RunArtifacts:
        video_path = Path(video_path).resolve()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            cap.release()
            raise RuntimeError("Video metadata is invalid: FPS must be positive.")

        def contexts() -> Iterable[FrameContext]:
            frame_number = 0
            try:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    yield FrameContext(
                        frame_number=frame_number,
                        timestamp_s=frame_number / fps,
                        image=image,
                        source_path=video_path,
                        sensor_id=sensor_id,
                        modality=modality,
                    )
                    frame_number += 1
            finally:
                cap.release()

        return self.run_frames(
            contexts(),
            output_dir,
            render_fps=fps,
            source_name=str(video_path),
            run_metadata={"source_type": "encoded_video"},
        )
