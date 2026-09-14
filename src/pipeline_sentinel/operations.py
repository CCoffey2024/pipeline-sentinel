from __future__ import annotations

import json
import platform
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .anomaly import AnomalyReference, TrackCropAnomalyAnalyzer
from .config import ProductionConfig, load_production_config
from .embeddings import DinoV2Embedder
from .events import ConsecutiveAnomalyEventDetector, DwellEventDetector, SeverityAlertPolicy
from .pipeline import PipelineSentinel, RunArtifacts
from .tracking import IoUTracker
from .yolo import YoloDetector


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def make_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def append_run_log(path: Path, *, event: str, run_id: str, **fields: Any) -> None:
    payload = {"timestamp_utc": utc_now(), "event": event, "run_id": run_id, **fields}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ProductionRunArtifacts:
    run_id: str
    output_dir: Path
    effective_config_json: Path
    run_log_jsonl: Path
    run_status_json: Path
    pipeline: RunArtifacts


def _resolve_config_relative(path_value: str, *, config_source: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_source.parent / candidate).resolve()


def _build_pipeline(
    config: ProductionConfig,
    *,
    config_source: Path,
) -> PipelineSentinel:
    detector = YoloDetector(
        model_name=config.detector.model,
        confidence=config.detector.confidence,
        iou=config.detector.nms_iou,
        imgsz=config.detector.imgsz,
        device=config.detector.device,
        class_ids=list(config.detector.class_ids) if config.detector.class_ids else None,
    )
    tracker = IoUTracker(
        iou_threshold=config.tracker.iou_threshold,
        max_missed_updates=config.tracker.max_missed_updates,
    )

    anomaly_analyzer = None
    anomaly_event_detector = None
    if config.anomaly.enabled:
        assert config.anomaly.reference_path is not None
        reference_path = _resolve_config_relative(
            config.anomaly.reference_path,
            config_source=config_source,
        )
        reference = AnomalyReference.load(reference_path)
        embedder = DinoV2Embedder(
            model_name=config.anomaly.model,
            device=config.anomaly.device,
        )
        anomaly_analyzer = TrackCropAnomalyAnalyzer(
            embedder,
            reference,
            labels=set(config.anomaly.labels) if config.anomaly.labels else None,
            min_track_hits=config.anomaly.min_track_hits,
            pad_px=config.anomaly.pad_px,
            min_crop_size=config.anomaly.min_crop_size,
        )
        anomaly_event_detector = ConsecutiveAnomalyEventDetector(
            min_consecutive=config.anomaly.event_min_consecutive,
            severity=config.anomaly.event_severity,
        )

    event_detector = None
    if config.events.dwell.enabled:
        event_detector = DwellEventDetector(
            min_hits=config.events.dwell.min_hits,
            max_displacement_px=config.events.dwell.max_displacement_px,
            labels=set(config.events.dwell.labels) if config.events.dwell.labels else None,
        )
    return PipelineSentinel(
        detector,
        tracker=tracker,
        anomaly_analyzer=anomaly_analyzer,
        event_detector=event_detector,
        anomaly_event_detector=anomaly_event_detector,
        alert_policy=SeverityAlertPolicy(minimum_severity=config.alerts.minimum_severity),
    )


def _add_operational_provenance(
    artifacts: RunArtifacts,
    *,
    run_id: str,
    started_at: str,
    completed_at: str,
    config_source: Path,
    config_sha256: str,
    effective_config_json: Path,
    run_log_jsonl: Path,
    run_status_json: Path,
) -> None:
    manifest = json.loads(artifacts.run_manifest.read_text(encoding="utf-8"))
    metadata = dict(manifest.get("metadata") or {})
    metadata.update(
        {
            "run_id": run_id,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "config_source": str(config_source),
            "config_sha256": config_sha256,
            "effective_config_json": str(effective_config_json),
            "run_log_jsonl": str(run_log_jsonl),
            "run_status_json": str(run_status_json),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        }
    )
    manifest["metadata"] = metadata
    artifact_map = dict(manifest.get("artifacts") or {})
    artifact_map.update(
        {
            "effective_config_json": str(effective_config_json),
            "run_log_jsonl": str(run_log_jsonl),
            "run_status_json": str(run_status_json),
        }
    )
    manifest["artifacts"] = artifact_map
    artifacts.run_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_configured_video(
    video_path: Path,
    config_path: Path,
    *,
    output_dir: Path | None = None,
) -> ProductionRunArtifacts:
    """Execute one auditable config-driven production-style video run."""

    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)

    config, provenance = load_production_config(config_path)
    run_id = make_run_id()
    resolved_output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (Path("outputs") / "runs" / run_id).resolve()
    )
    if resolved_output.exists() and any(resolved_output.iterdir()):
        raise RuntimeError(f"Run output directory is not empty: {resolved_output}")
    resolved_output.mkdir(parents=True, exist_ok=True)

    effective_config_json = resolved_output / "effective_config.json"
    run_log_jsonl = resolved_output / "run_log.jsonl"
    run_status_json = resolved_output / "run_status.json"
    started_at = utc_now()

    write_json(
        effective_config_json,
        {
            "pipeline_sentinel_version": __version__,
            "config_source": str(provenance.path),
            "config_sha256": provenance.sha256,
            "effective_config": config.to_dict(),
        },
    )
    status: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "input_source": str(video),
        "output_dir": str(resolved_output),
        "config_source": str(provenance.path),
        "config_sha256": provenance.sha256,
        "error": None,
    }
    write_json(run_status_json, status)
    append_run_log(
        run_log_jsonl,
        event="run_started",
        run_id=run_id,
        input_source=str(video),
        output_dir=str(resolved_output),
        config_sha256=provenance.sha256,
        version=__version__,
    )

    try:
        pipeline = _build_pipeline(config, config_source=provenance.path)
        artifacts = pipeline.run_video(
            video,
            resolved_output,
            sensor_id=config.runtime.sensor_id,
            modality=config.runtime.modality,
        )
        completed_at = utc_now()
        _add_operational_provenance(
            artifacts,
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            config_source=provenance.path,
            config_sha256=provenance.sha256,
            effective_config_json=effective_config_json,
            run_log_jsonl=run_log_jsonl,
            run_status_json=run_status_json,
        )
    except Exception as exc:
        completed_at = utc_now()
        status.update(
            {
                "status": "failed",
                "completed_at_utc": completed_at,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(run_status_json, status)
        append_run_log(
            run_log_jsonl,
            event="run_failed",
            run_id=run_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise

    status.update(
        {
            "status": "completed",
            "completed_at_utc": completed_at,
            "artifacts": {
                "annotated_video": str(artifacts.annotated_video),
                "detections_csv": str(artifacts.detections_csv),
                "tracks_csv": str(artifacts.tracks_csv),
                "anomalies_csv": str(artifacts.anomalies_csv),
                "events_csv": str(artifacts.events_csv),
                "alerts_csv": str(artifacts.alerts_csv),
                "run_manifest": str(artifacts.run_manifest),
                "effective_config_json": str(effective_config_json),
                "run_log_jsonl": str(run_log_jsonl),
                "run_status_json": str(run_status_json),
            },
        }
    )
    write_json(run_status_json, status)
    append_run_log(
        run_log_jsonl,
        event="run_completed",
        run_id=run_id,
        completed_at_utc=completed_at,
        run_manifest=str(artifacts.run_manifest),
    )

    return ProductionRunArtifacts(
        run_id=run_id,
        output_dir=resolved_output,
        effective_config_json=effective_config_json,
        run_log_jsonl=run_log_jsonl,
        run_status_json=run_status_json,
        pipeline=artifacts,
    )
