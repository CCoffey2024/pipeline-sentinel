from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .types import Modality, Severity


class ConfigError(ValueError):
    """Raised when a production configuration is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    backend: Literal["yolo"] = "yolo"
    model: str = "yolo26n.pt"
    confidence: float = 0.25
    nms_iou: float = 0.70
    imgsz: int = 960
    device: str | int | None = None
    class_ids: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    backend: Literal["iou"] = "iou"
    iou_threshold: float = 0.30
    max_missed_updates: int = 2


@dataclass(frozen=True, slots=True)
class DwellConfig:
    enabled: bool = False
    min_hits: int = 30
    max_displacement_px: float = 40.0
    labels: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class EventConfig:
    dwell: DwellConfig = field(default_factory=DwellConfig)


@dataclass(frozen=True, slots=True)
class AlertConfig:
    minimum_severity: Severity = "warning"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    sensor_id: str = "EO_CAM_01"
    modality: Modality = "EO"


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    events: EventConfig = field(default_factory=EventConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConfigProvenance:
    path: Path
    sha256: str


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown(section: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"Unknown {name} option(s): {sorted(unknown)}")


def _probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be numeric") from exc
    if not 0.0 <= result <= 1.0:
        raise ConfigError(f"{name} must be between 0 and 1")
    return result


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return result


def _optional_int_tuple(value: Any, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a YAML list or null")
    values = tuple(_positive_int(item, name, minimum=0) for item in value)
    return values or None


def _optional_str_tuple(value: Any, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{name} must be a YAML list of non-empty strings or null")
    values = tuple(value)
    return values or None


def parse_production_config(data: dict[str, Any]) -> ProductionConfig:
    """Parse and validate one production runtime configuration mapping."""

    _reject_unknown(data, {"detector", "tracker", "events", "alerts", "runtime"}, "top-level")

    detector = _mapping(data.get("detector"), "detector")
    _reject_unknown(
        detector,
        {"backend", "model", "confidence", "nms_iou", "imgsz", "device", "class_ids"},
        "detector",
    )
    backend = detector.get("backend", "yolo")
    if backend != "yolo":
        raise ConfigError("detector.backend currently supports only 'yolo'")
    model = detector.get("model", "yolo26n.pt")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("detector.model must be a non-empty string")
    device = detector.get("device")
    if device is not None and not isinstance(device, (str, int)):
        raise ConfigError("detector.device must be a string, integer, or null")
    detector_config = DetectorConfig(
        model=model,
        confidence=_probability(detector.get("confidence", 0.25), "detector.confidence"),
        nms_iou=_probability(detector.get("nms_iou", 0.70), "detector.nms_iou"),
        imgsz=_positive_int(detector.get("imgsz", 960), "detector.imgsz"),
        device=device,
        class_ids=_optional_int_tuple(detector.get("class_ids"), "detector.class_ids"),
    )

    tracker = _mapping(data.get("tracker"), "tracker")
    _reject_unknown(tracker, {"backend", "iou_threshold", "max_missed_updates"}, "tracker")
    tracker_backend = tracker.get("backend", "iou")
    if tracker_backend != "iou":
        raise ConfigError("tracker.backend currently supports only 'iou'")
    tracker_config = TrackerConfig(
        iou_threshold=_probability(tracker.get("iou_threshold", 0.30), "tracker.iou_threshold"),
        max_missed_updates=_positive_int(
            tracker.get("max_missed_updates", 2),
            "tracker.max_missed_updates",
            minimum=0,
        ),
    )

    events = _mapping(data.get("events"), "events")
    _reject_unknown(events, {"dwell"}, "events")
    dwell = _mapping(events.get("dwell"), "events.dwell")
    _reject_unknown(
        dwell,
        {"enabled", "min_hits", "max_displacement_px", "labels"},
        "events.dwell",
    )
    enabled = dwell.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("events.dwell.enabled must be true or false")
    try:
        max_displacement_px = float(dwell.get("max_displacement_px", 40.0))
    except (TypeError, ValueError) as exc:
        raise ConfigError("events.dwell.max_displacement_px must be numeric") from exc
    if max_displacement_px < 0:
        raise ConfigError("events.dwell.max_displacement_px cannot be negative")
    event_config = EventConfig(
        dwell=DwellConfig(
            enabled=enabled,
            min_hits=_positive_int(dwell.get("min_hits", 30), "events.dwell.min_hits", minimum=2),
            max_displacement_px=max_displacement_px,
            labels=_optional_str_tuple(dwell.get("labels"), "events.dwell.labels"),
        )
    )

    alerts = _mapping(data.get("alerts"), "alerts")
    _reject_unknown(alerts, {"minimum_severity"}, "alerts")
    minimum_severity = alerts.get("minimum_severity", "warning")
    if minimum_severity not in {"info", "warning", "critical"}:
        raise ConfigError("alerts.minimum_severity must be info, warning, or critical")
    alert_config = AlertConfig(minimum_severity=minimum_severity)

    runtime = _mapping(data.get("runtime"), "runtime")
    _reject_unknown(runtime, {"sensor_id", "modality"}, "runtime")
    sensor_id = runtime.get("sensor_id", "EO_CAM_01")
    if not isinstance(sensor_id, str) or not sensor_id.strip():
        raise ConfigError("runtime.sensor_id must be a non-empty string")
    modality = runtime.get("modality", "EO")
    if modality not in {"EO", "IR", "OTHER"}:
        raise ConfigError("runtime.modality must be EO, IR, or OTHER")
    runtime_config = RuntimeConfig(sensor_id=sensor_id, modality=modality)

    return ProductionConfig(
        detector=detector_config,
        tracker=tracker_config,
        events=event_config,
        alerts=alert_config,
        runtime=runtime_config,
    )


def load_production_config(path: Path) -> tuple[ProductionConfig, ConfigProvenance]:
    """Load validated YAML plus a content hash for run provenance."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    raw = resolved.read_bytes()
    try:
        loaded = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not parse YAML configuration: {resolved}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError("Top-level YAML document must be a mapping")
    config = parse_production_config(dict(loaded))
    provenance = ConfigProvenance(path=resolved, sha256=hashlib.sha256(raw).hexdigest())
    return config, provenance
