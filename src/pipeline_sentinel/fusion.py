from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from . import __version__
from .events import SeverityAlertPolicy
from .types import Alert, Event, Modality, Severity


class FusionConfigError(ValueError):
    """Raised when a sensor-fusion configuration is malformed."""


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Configuration for auditable late fusion of completed sensor runs."""

    strategy: Literal["temporal_consensus"] = "temporal_consensus"
    max_time_delta_s: float = 0.50
    min_sensors: int = 2
    require_matching_label: bool = True
    spatial_iou_threshold: float | None = None
    minimum_severity: Severity = "warning"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FusionConfigProvenance:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class SensorEventEvidence:
    """One semantic event emitted by one completed sensor run."""

    run_dir: Path
    run_id: str | None
    sensor_id: str
    modality: Modality
    event_id: str
    frame_number: int
    timestamp_s: float
    event_type: str
    severity: Severity
    source: str
    track_id: int | None
    label: str | None
    confidence: float | None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class FusionArtifacts:
    fusion_id: str
    output_dir: Path
    events_csv: Path
    alerts_csv: Path
    contributors_csv: Path
    manifest_json: Path
    effective_config_json: Path


def default_fusion_config_path() -> Path:
    return Path(__file__).with_name("fusion.yaml").resolve()


def _severity(value: Any, name: str) -> Severity:
    if value not in {"info", "warning", "critical"}:
        raise FusionConfigError(f"{name} must be info, warning, or critical")
    return value


def _probability_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FusionConfigError(f"{name} must be numeric or null") from exc
    if not 0.0 <= result <= 1.0:
        raise FusionConfigError(f"{name} must be between 0 and 1")
    return result


def parse_fusion_config(data: dict[str, Any]) -> FusionConfig:
    allowed = {
        "strategy",
        "max_time_delta_s",
        "min_sensors",
        "require_matching_label",
        "spatial_iou_threshold",
        "alerts",
    }
    unknown = set(data) - allowed
    if unknown:
        raise FusionConfigError(f"Unknown fusion option(s): {sorted(unknown)}")

    strategy = data.get("strategy", "temporal_consensus")
    if strategy != "temporal_consensus":
        raise FusionConfigError("strategy currently supports only 'temporal_consensus'")

    try:
        max_time_delta_s = float(data.get("max_time_delta_s", 0.50))
    except (TypeError, ValueError) as exc:
        raise FusionConfigError("max_time_delta_s must be numeric") from exc
    if max_time_delta_s < 0:
        raise FusionConfigError("max_time_delta_s cannot be negative")

    min_sensors = data.get("min_sensors", 2)
    if isinstance(min_sensors, bool):
        raise FusionConfigError("min_sensors must be an integer")
    try:
        min_sensors = int(min_sensors)
    except (TypeError, ValueError) as exc:
        raise FusionConfigError("min_sensors must be an integer") from exc
    if min_sensors < 2:
        raise FusionConfigError("min_sensors must be at least 2")

    require_matching_label = data.get("require_matching_label", True)
    if not isinstance(require_matching_label, bool):
        raise FusionConfigError("require_matching_label must be true or false")

    alerts = data.get("alerts") or {}
    if not isinstance(alerts, dict):
        raise FusionConfigError("alerts must be a mapping")
    alert_unknown = set(alerts) - {"minimum_severity"}
    if alert_unknown:
        raise FusionConfigError(f"Unknown alerts option(s): {sorted(alert_unknown)}")

    return FusionConfig(
        strategy=strategy,
        max_time_delta_s=max_time_delta_s,
        min_sensors=min_sensors,
        require_matching_label=require_matching_label,
        spatial_iou_threshold=_probability_or_none(
            data.get("spatial_iou_threshold"),
            "spatial_iou_threshold",
        ),
        minimum_severity=_severity(
            alerts.get("minimum_severity", "warning"),
            "alerts.minimum_severity",
        ),
    )


def load_fusion_config(
    path: Path | None = None,
) -> tuple[FusionConfig, FusionConfigProvenance]:
    if path is None:
        resolved = default_fusion_config_path()
    else:
        requested = Path(path).expanduser()
        resolved = requested.resolve()
        if not resolved.is_file() and requested.as_posix().lower() == "config/fusion.yaml":
            resolved = default_fusion_config_path()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    raw = resolved.read_bytes()
    try:
        loaded = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise FusionConfigError(f"Could not parse fusion YAML: {resolved}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise FusionConfigError("Top-level fusion YAML document must be a mapping")
    return (
        parse_fusion_config(dict(loaded)),
        FusionConfigProvenance(path=resolved, sha256=hashlib.sha256(raw).hexdigest()),
    )


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _track_boxes(run_dir: Path) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    path = run_dir / "tracks.csv"
    if not path.is_file():
        return {}
    table = pd.read_csv(path)
    required = {"frame_number", "track_id", "x1", "y1", "x2", "y2"}
    if not required.issubset(table.columns):
        return {}
    boxes: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for row in table.itertuples(index=False):
        boxes[(int(row.frame_number), int(row.track_id))] = (
            float(row.x1),
            float(row.y1),
            float(row.x2),
            float(row.y2),
        )
    return boxes


def load_sensor_run_events(run_dir: Path) -> tuple[list[SensorEventEvidence], dict[str, Any]]:
    resolved = Path(run_dir).expanduser().resolve()
    manifest_path = resolved / "run_manifest.json"
    events_path = resolved / "events.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not events_path.is_file():
        raise FileNotFoundError(events_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sensor_id = str(manifest.get("sensor_id") or "").strip()
    modality = manifest.get("modality")
    if not sensor_id:
        raise ValueError(f"run manifest has no sensor_id: {manifest_path}")
    if modality not in {"EO", "IR", "OTHER"}:
        raise ValueError(f"run manifest has unsupported modality {modality!r}: {manifest_path}")

    events = pd.read_csv(events_path)
    required = {
        "event_id",
        "frame_number",
        "timestamp_s",
        "event_type",
        "severity",
        "source",
        "track_id",
        "label",
        "confidence",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events.csv is missing required columns: {sorted(missing)}")

    boxes = _track_boxes(resolved)
    run_id = (manifest.get("metadata") or {}).get("run_id")
    evidence: list[SensorEventEvidence] = []
    for row in events.itertuples(index=False):
        severity = str(row.severity)
        if severity not in {"info", "warning", "critical"}:
            raise ValueError(f"Unsupported event severity {severity!r} in {events_path}")
        frame_number = int(row.frame_number)
        track_id = _optional_int(row.track_id)
        bbox = boxes.get((frame_number, track_id)) if track_id is not None else None
        evidence.append(
            SensorEventEvidence(
                run_dir=resolved,
                run_id=str(run_id) if run_id else None,
                sensor_id=sensor_id,
                modality=modality,
                event_id=str(row.event_id),
                frame_number=frame_number,
                timestamp_s=float(row.timestamp_s),
                event_type=str(row.event_type),
                severity=severity,
                source=str(row.source),
                track_id=track_id,
                label=_optional_str(row.label),
                confidence=_optional_float(row.confidence),
                bbox=bbox,
            )
        )

    metadata = {
        "run_dir": str(resolved),
        "run_id": run_id,
        "sensor_id": sensor_id,
        "modality": modality,
        "pipeline_sentinel_version": manifest.get("pipeline_sentinel_version"),
        "manifest_sha256": _manifest_sha256(manifest_path),
        "events": len(evidence),
    }
    return evidence, metadata


_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


class TemporalConsensusFuser:
    """Late-fuse semantic events emitted by independent sensor pipelines.

    Matching always requires the same event type and distinct sensor IDs. It can additionally require
    equal labels and, when streams are explicitly registered into the same pixel geometry, a minimum
    track-box IoU. The fuser does not assume that EO and IR pixel coordinates align unless a spatial
    threshold is configured.
    """

    name = "temporal_consensus"

    def __init__(self, config: FusionConfig) -> None:
        self.config = config

    def _compatible(self, anchor: SensorEventEvidence, candidate: SensorEventEvidence) -> bool:
        if anchor.sensor_id == candidate.sensor_id:
            return False
        if anchor.event_type != candidate.event_type:
            return False
        if abs(anchor.timestamp_s - candidate.timestamp_s) > self.config.max_time_delta_s:
            return False
        if self.config.require_matching_label and anchor.label != candidate.label:
            return False
        threshold = self.config.spatial_iou_threshold
        if threshold is not None:
            if anchor.bbox is None or candidate.bbox is None:
                return False
            if box_iou(anchor.bbox, candidate.bbox) < threshold:
                return False
        return True

    def fuse(
        self,
        evidence: list[SensorEventEvidence],
    ) -> tuple[list[Event], list[dict[str, object]]]:
        ordered = sorted(
            evidence,
            key=lambda item: (item.timestamp_s, item.sensor_id, item.event_id),
        )
        consumed: set[int] = set()
        fused: list[Event] = []
        contributor_rows: list[dict[str, object]] = []

        for anchor_index, anchor in enumerate(ordered):
            if anchor_index in consumed:
                continue

            group: list[tuple[int, SensorEventEvidence]] = [(anchor_index, anchor)]
            used_sensors = {anchor.sensor_id}
            candidates: list[tuple[float, float, int, SensorEventEvidence]] = []
            for candidate_index, candidate in enumerate(ordered):
                if candidate_index == anchor_index or candidate_index in consumed:
                    continue
                if candidate.sensor_id in used_sensors:
                    continue
                if not self._compatible(anchor, candidate):
                    continue
                iou = (
                    box_iou(anchor.bbox, candidate.bbox)
                    if anchor.bbox is not None and candidate.bbox is not None
                    else -1.0
                )
                candidates.append(
                    (
                        abs(anchor.timestamp_s - candidate.timestamp_s),
                        -iou,
                        candidate_index,
                        candidate,
                    )
                )

            for _, _, candidate_index, candidate in sorted(candidates):
                if candidate.sensor_id in used_sensors:
                    continue
                group.append((candidate_index, candidate))
                used_sensors.add(candidate.sensor_id)

            if len(used_sensors) < self.config.min_sensors:
                continue

            for index, _ in group:
                consumed.add(index)

            members = [member for _, member in group]
            timestamps = [member.timestamp_s for member in members]
            fused_timestamp = sum(timestamps) / len(timestamps)
            severity = max(members, key=lambda member: _SEVERITY_RANK[member.severity]).severity
            labels = {member.label for member in members}
            fused_label = next(iter(labels)) if len(labels) == 1 else None
            fusion_event_id = f"fusion:{len(fused) + 1:06d}:{anchor.event_type}"
            spatial_ious = [
                box_iou(members[0].bbox, member.bbox)
                for member in members[1:]
                if members[0].bbox is not None and member.bbox is not None
            ]

            metadata = {
                "strategy": self.name,
                "support_count": len(members),
                "sensor_ids": [member.sensor_id for member in members],
                "modalities": [member.modality for member in members],
                "contributor_event_ids": [member.event_id for member in members],
                "contributor_run_ids": [member.run_id for member in members],
                "time_span_s": max(timestamps) - min(timestamps),
                "require_matching_label": self.config.require_matching_label,
                "spatial_iou_threshold": self.config.spatial_iou_threshold,
                "minimum_pair_iou": min(spatial_ious) if spatial_ious else None,
            }
            fused.append(
                Event(
                    event_id=fusion_event_id,
                    frame_number=anchor.frame_number,
                    timestamp_s=fused_timestamp,
                    event_type=anchor.event_type,
                    severity=severity,
                    source=self.name,
                    message=(
                        f"{len(members)} sensors corroborated '{anchor.event_type}' within "
                        f"{metadata['time_span_s']:.3f}s."
                    ),
                    track_id=None,
                    label=fused_label,
                    confidence=None,
                    metadata=metadata,
                )
            )

            for member in members:
                contributor_rows.append(
                    {
                        "fusion_event_id": fusion_event_id,
                        "run_dir": str(member.run_dir),
                        "run_id": member.run_id,
                        "sensor_id": member.sensor_id,
                        "modality": member.modality,
                        "event_id": member.event_id,
                        "frame_number": member.frame_number,
                        "timestamp_s": member.timestamp_s,
                        "event_type": member.event_type,
                        "severity": member.severity,
                        "track_id": member.track_id,
                        "label": member.label,
                        "confidence": member.confidence,
                        "x1": member.bbox[0] if member.bbox else None,
                        "y1": member.bbox[1] if member.bbox else None,
                        "x2": member.bbox[2] if member.bbox else None,
                        "y2": member.bbox[3] if member.bbox else None,
                        "time_delta_from_fused_s": member.timestamp_s - fused_timestamp,
                    }
                )

        return fused, contributor_rows


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


def _fusion_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"fusion-{stamp}-{uuid.uuid4().hex[:8]}"


def fuse_run_directories(
    run_dirs: list[Path],
    *,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> FusionArtifacts:
    """Late-fuse semantic events from two or more completed Pipeline Sentinel runs."""

    if len(run_dirs) < 2:
        raise ValueError("at least two sensor run directories are required")
    config, provenance = load_fusion_config(config_path)

    all_evidence: list[SensorEventEvidence] = []
    input_metadata: list[dict[str, Any]] = []
    seen_sensors: set[str] = set()
    for run_dir in run_dirs:
        evidence, metadata = load_sensor_run_events(run_dir)
        sensor_id = str(metadata["sensor_id"])
        if sensor_id in seen_sensors:
            raise ValueError(f"duplicate sensor_id across fusion inputs: {sensor_id}")
        seen_sensors.add(sensor_id)
        all_evidence.extend(evidence)
        input_metadata.append(metadata)

    if config.min_sensors > len(seen_sensors):
        raise ValueError(
            f"fusion min_sensors={config.min_sensors} exceeds distinct input sensors={len(seen_sensors)}"
        )

    fusion_id = _fusion_id()
    resolved_output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (Path("outputs") / "fusion" / fusion_id).resolve()
    )
    if resolved_output.exists() and any(resolved_output.iterdir()):
        raise RuntimeError(f"Fusion output directory is not empty: {resolved_output}")
    resolved_output.mkdir(parents=True, exist_ok=True)

    fuser = TemporalConsensusFuser(config)
    fused_events, contributor_rows = fuser.fuse(all_evidence)
    alert_policy = SeverityAlertPolicy(minimum_severity=config.minimum_severity)
    alerts = [
        alert
        for event in fused_events
        if (alert := alert_policy.evaluate(event)) is not None
    ]

    events_csv = resolved_output / "fusion_events.csv"
    alerts_csv = resolved_output / "fusion_alerts.csv"
    contributors_csv = resolved_output / "fusion_contributors.csv"
    manifest_json = resolved_output / "fusion_manifest.json"
    effective_config_json = resolved_output / "effective_fusion_config.json"

    event_columns = [
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
    alert_columns = [
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
    contributor_columns = [
        "fusion_event_id",
        "run_dir",
        "run_id",
        "sensor_id",
        "modality",
        "event_id",
        "frame_number",
        "timestamp_s",
        "event_type",
        "severity",
        "track_id",
        "label",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "time_delta_from_fused_s",
    ]
    pd.DataFrame([_event_row(event) for event in fused_events], columns=event_columns).to_csv(
        events_csv,
        index=False,
    )
    pd.DataFrame([_alert_row(alert) for alert in alerts], columns=alert_columns).to_csv(
        alerts_csv,
        index=False,
    )
    pd.DataFrame(contributor_rows, columns=contributor_columns).to_csv(
        contributors_csv,
        index=False,
    )

    effective_config_json.write_text(
        json.dumps(
            {
                "pipeline_sentinel_version": __version__,
                "config_source": str(provenance.path),
                "config_sha256": provenance.sha256,
                "effective_config": config.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_json.write_text(
        json.dumps(
            {
                "pipeline_sentinel_version": __version__,
                "fusion_id": fusion_id,
                "strategy": fuser.name,
                "inputs": input_metadata,
                "distinct_sensors": len(seen_sensors),
                "input_events": len(all_evidence),
                "fused_events": len(fused_events),
                "fused_alerts": len(alerts),
                "contributors": len(contributor_rows),
                "config_source": str(provenance.path),
                "config_sha256": provenance.sha256,
                "assumptions": {
                    "fusion_level": "semantic_event_late_fusion",
                    "raw_pixels_fused": False,
                    "feature_vectors_fused": False,
                    "pixel_registration_required": config.spatial_iou_threshold is not None,
                    "confidence_recalibrated": False,
                },
                "artifacts": {
                    "fusion_events_csv": str(events_csv),
                    "fusion_alerts_csv": str(alerts_csv),
                    "fusion_contributors_csv": str(contributors_csv),
                    "effective_fusion_config_json": str(effective_config_json),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return FusionArtifacts(
        fusion_id=fusion_id,
        output_dir=resolved_output,
        events_csv=events_csv,
        alerts_csv=alerts_csv,
        contributors_csv=contributors_csv,
        manifest_json=manifest_json,
        effective_config_json=effective_config_json,
    )
