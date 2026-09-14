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
from .anomaly import AnomalyAnalyzer
from .detectors import Detector
from .events import AlertPolicy, AnomalyEventDetector, EventDetector, SeverityAlertPolicy
from .tracking import IoUTracker, Tracker
from .types import Alert, AnomalyObservation, Detection, Event, FrameContext, Modality, Track

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
TRACK_COLUMNS = [
    "frame_number",
    "timestamp_s",
    "track_id",
    "label",
    "confidence",
    "hits",
    "duration_s",
    "first_frame_number",
    "first_timestamp_s",
    "scenario_role",
    "tracker",
    "sensor_id",
    "modality",
    "source_path",
    "x1",
    "y1",
    "x2",
    "y2",
]
ANOMALY_COLUMNS = [
    "frame_number",
    "timestamp_s",
    "track_id",
    "label",
    "score",
    "threshold",
    "margin",
    "is_anomaly",
    "source",
    "sensor_id",
    "modality",
    "source_path",
    "metadata",
]
EVENT_COLUMNS = [
    "event_id",
    "frame_number",
    "timestamp_s",
    "event_type",
    "severity",
    "track_id",
    "label",
    "confidence",
    "source",
    "message",
    "metadata",
]
ALERT_COLUMNS = [
    "alert_id",
    "event_id",
    "frame_number",
    "timestamp_s",
    "alert_type",
    "severity",
    "track_id",
    "label",
    "confidence",
    "source",
    "message",
    "metadata",
]


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    annotated_video: Path
    detections_csv: Path
    tracks_csv: Path
    anomalies_csv: Path
    events_csv: Path
    alerts_csv: Path
    run_manifest: Path


def _detection_row(detection: Detection, frame: FrameContext) -> dict[str, object]:
    x1, y1, x2, y2 = detection.xyxy
    return {
        "frame_number": detection.frame_number,
        "timestamp_s": frame.timestamp_s,
        "label": detection.label,
        "object_id": detection.object_id,
        "scenario_role": detection.scenario_role,
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


def _track_row(track: Track, frame: FrameContext) -> dict[str, object]:
    x1, y1, x2, y2 = track.xyxy
    return {
        "frame_number": track.frame_number,
        "timestamp_s": track.timestamp_s,
        "track_id": track.track_id,
        "label": track.label,
        "confidence": track.confidence,
        "hits": track.hits,
        "duration_s": track.duration_s,
        "first_frame_number": track.first_frame_number,
        "first_timestamp_s": track.first_timestamp_s,
        "scenario_role": track.scenario_role,
        "tracker": track.source,
        "sensor_id": frame.sensor_id,
        "modality": frame.modality,
        "source_path": str(frame.source_path) if frame.source_path else None,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def _anomaly_row(
    observation: AnomalyObservation,
    frame: FrameContext,
) -> dict[str, object]:
    return {
        "frame_number": observation.frame_number,
        "timestamp_s": observation.timestamp_s,
        "track_id": observation.track_id,
        "label": observation.label,
        "score": observation.score,
        "threshold": observation.threshold,
        "margin": observation.margin,
        "is_anomaly": observation.is_anomaly,
        "source": observation.source,
        "sensor_id": frame.sensor_id,
        "modality": frame.modality,
        "source_path": str(frame.source_path) if frame.source_path else None,
        "metadata": json.dumps(observation.metadata, sort_keys=True),
    }


def _event_row(event: Event) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "frame_number": event.frame_number,
        "timestamp_s": event.timestamp_s,
        "event_type": event.event_type,
        "severity": event.severity,
        "track_id": event.track_id,
        "label": event.label,
        "confidence": event.confidence,
        "source": event.source,
        "message": event.message,
        "metadata": json.dumps(event.metadata, sort_keys=True),
    }


def _alert_row(alert: Alert) -> dict[str, object]:
    return {
        "alert_id": alert.alert_id,
        "event_id": alert.event_id,
        "frame_number": alert.frame_number,
        "timestamp_s": alert.timestamp_s,
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "track_id": alert.track_id,
        "label": alert.label,
        "confidence": alert.confidence,
        "source": alert.source,
        "message": alert.message,
        "metadata": json.dumps(alert.metadata, sort_keys=True),
    }


class PipelineSentinel:
    """Orchestrate detector, tracker, anomaly, event, and alert components."""

    def __init__(
        self,
        detector: Detector,
        *,
        tracker: Tracker | None = None,
        anomaly_analyzer: AnomalyAnalyzer | None = None,
        event_detector: EventDetector | None = None,
        anomaly_event_detector: AnomalyEventDetector | None = None,
        alert_policy: AlertPolicy | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker or IoUTracker()
        self.anomaly_analyzer = anomaly_analyzer
        self.event_detector = event_detector
        self.anomaly_event_detector = anomaly_event_detector
        self.alert_policy = alert_policy or SeverityAlertPolicy()

    def run_frames(
        self,
        frames: Iterable[FrameContext],
        output_dir: Path,
        *,
        render_fps: float = 30.0,
        source_name: str | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> RunArtifacts:
        """Run the pipeline over any ordered stream of ``FrameContext`` objects."""

        if render_fps <= 0:
            raise ValueError("render_fps must be positive")

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        iterator = iter(frames)
        first = next(iterator, None)
        if first is None:
            raise ValueError("frame source yielded zero frames")

        self.tracker.reset()
        if self.anomaly_analyzer is not None:
            self.anomaly_analyzer.reset()
        if self.event_detector is not None:
            self.event_detector.reset()
        if self.anomaly_event_detector is not None:
            self.anomaly_event_detector.reset()

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
        track_rows: list[dict[str, object]] = []
        anomaly_rows: list[dict[str, object]] = []
        event_rows: list[dict[str, object]] = []
        alert_rows: list[dict[str, object]] = []
        unique_track_ids: set[int] = set()
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

                detections = list(self.detector.detect(frame))
                detection_rows.extend(_detection_row(detection, frame) for detection in detections)

                tracks = self.tracker.update(frame, detections)
                track_rows.extend(_track_row(track, frame) for track in tracks)
                unique_track_ids.update(track.track_id for track in tracks)

                anomalies = (
                    self.anomaly_analyzer.update(frame, tracks)
                    if self.anomaly_analyzer is not None
                    else []
                )
                anomaly_rows.extend(_anomaly_row(observation, frame) for observation in anomalies)

                events: list[Event] = []
                if self.event_detector is not None:
                    events.extend(self.event_detector.update(frame, tracks))
                if self.anomaly_event_detector is not None:
                    events.extend(self.anomaly_event_detector.update(frame, anomalies))
                event_rows.extend(_event_row(event) for event in events)

                frame_alerts: list[Alert] = []
                for event in events:
                    alert = self.alert_policy.evaluate(event)
                    if alert is not None:
                        frame_alerts.append(alert)
                        alert_rows.append(_alert_row(alert))
                alert_track_ids = {
                    alert.track_id for alert in frame_alerts if alert.track_id is not None
                }
                anomalous_track_ids = {
                    observation.track_id for observation in anomalies if observation.is_anomaly
                }

                rendered = frame.image.copy()
                for track in tracks:
                    x1, y1, x2, y2 = track.xyxy
                    is_alert = track.track_id in alert_track_ids
                    is_anomaly = track.track_id in anomalous_track_ids
                    box_value = (255, 255, 255) if is_alert else (180, 180, 180)
                    thickness = 3 if (is_alert or is_anomaly) else 2
                    cv2.rectangle(rendered, (x1, y1), (x2, y2), box_value, thickness)
                    suffix = " ANOM" if is_anomaly else ""
                    text = f"#{track.track_id} {track.label} {track.confidence:.2f}{suffix}"
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

                writer.write(rendered)
                frames_processed += 1
                last_frame_number = frame.frame_number
        finally:
            writer.release()

        detections_csv = output_dir / "detections.csv"
        pd.DataFrame(detection_rows, columns=DETECTION_COLUMNS).to_csv(detections_csv, index=False)

        tracks_csv = output_dir / "tracks.csv"
        pd.DataFrame(track_rows, columns=TRACK_COLUMNS).to_csv(tracks_csv, index=False)

        anomalies_csv = output_dir / "anomalies.csv"
        pd.DataFrame(anomaly_rows, columns=ANOMALY_COLUMNS).to_csv(anomalies_csv, index=False)

        events_csv = output_dir / "events.csv"
        pd.DataFrame(event_rows, columns=EVENT_COLUMNS).to_csv(events_csv, index=False)

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
                    "tracker_backend": self.tracker.name,
                    "anomaly_analyzer": (
                        self.anomaly_analyzer.name if self.anomaly_analyzer else None
                    ),
                    "event_detector": self.event_detector.name if self.event_detector else None,
                    "anomaly_event_detector": (
                        self.anomaly_event_detector.name if self.anomaly_event_detector else None
                    ),
                    "alert_policy": self.alert_policy.name,
                    "frames_processed": frames_processed,
                    "first_frame_number": first_frame_number,
                    "last_frame_number": last_frame_number,
                    "detections_emitted": len(detection_rows),
                    "track_observations_emitted": len(track_rows),
                    "unique_tracks": len(unique_track_ids),
                    "anomaly_observations_emitted": len(anomaly_rows),
                    "anomalies_flagged": sum(bool(row["is_anomaly"]) for row in anomaly_rows),
                    "events_emitted": len(event_rows),
                    "alerts_emitted": len(alert_rows),
                    "render_fps": float(render_fps),
                    "metadata": metadata,
                    "artifacts": {
                        "annotated_video": str(annotated_video),
                        "detections_csv": str(detections_csv),
                        "tracks_csv": str(tracks_csv),
                        "anomalies_csv": str(anomalies_csv),
                        "events_csv": str(events_csv),
                        "alerts_csv": str(alerts_csv),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return RunArtifacts(
            annotated_video=annotated_video,
            detections_csv=detections_csv,
            tracks_csv=tracks_csv,
            anomalies_csv=anomalies_csv,
            events_csv=events_csv,
            alerts_csv=alerts_csv,
            run_manifest=run_manifest,
        )

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
