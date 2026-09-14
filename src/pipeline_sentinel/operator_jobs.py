from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from .config import load_production_config
from .fusion import fuse_run_directories
from .operations import run_configured_video
from .types import Modality

JobKind = Literal["run", "fusion"]
JobStatus = Literal["queued", "running", "completed", "failed"]

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def make_operator_job_id(kind: JobKind) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{kind}-{stamp}-{uuid.uuid4().hex[:8]}"


def safe_upload_name(filename: str | None) -> str:
    """Normalize a browser-provided filename without trusting path components."""

    raw = Path(filename or "input.mp4").name.strip() or "input.mp4"
    normalized = _SAFE_FILENAME.sub("_", raw)
    return normalized[:180]


def validate_video_suffix(path: Path) -> None:
    if path.suffix.lower() not in _VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(_VIDEO_SUFFIXES))
        raise ValueError(f"unsupported video extension {path.suffix!r}; expected one of: {allowed}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(slots=True)
class OperatorJob:
    job_id: str
    kind: JobKind
    status: JobStatus
    created_at_utc: str
    started_at_utc: str | None
    completed_at_utc: str | None
    display_name: str
    output_dir: str
    input_path: str | None = None
    sensor_id: str | None = None
    modality: Modality | None = None
    source_job_ids: list[str] | None = None
    source_run_dirs: list[str] | None = None
    config_source: str | None = None
    pipeline_run_id: str | None = None
    fusion_id: str | None = None
    artifacts: dict[str, str] | None = None
    summary: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OperatorJob:
        return cls(**payload)


class OperatorJobManager:
    """Persist and execute local operator jobs with bounded concurrency.

    The first operator console intentionally defaults to one worker. Detector and representation
    runtimes can own substantial CPU/GPU memory, and allowing a browser click to start several model
    stacks concurrently is a poor default for a workstation application.
    """

    def __init__(self, workspace: Path, *, max_workers: int = 1) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.workspace = Path(workspace).expanduser().resolve()
        self.jobs_dir = self.workspace / "jobs"
        self.uploads_dir = self.workspace / "uploads"
        self.runs_dir = self.workspace / "runs"
        self.fusions_dir = self.workspace / "fusions"
        for directory in (self.jobs_dir, self.uploads_dir, self.runs_dir, self.fusions_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sentinel-job")
        self._jobs: dict[str, OperatorJob] = {}
        self._load_jobs()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _persist(self, job: OperatorJob) -> None:
        path = self._job_path(job.job_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _load_jobs(self) -> None:
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                job = OperatorJob.from_dict(payload)
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if job.status in {"queued", "running"}:
                job.status = "failed"
                job.completed_at_utc = utc_now()
                job.error = {
                    "type": "InterruptedJob",
                    "message": "Operator service restarted before this job completed.",
                }
                self._persist(job)
            self._jobs[job.job_id] = job

    def list_jobs(self, *, limit: int = 100) -> list[OperatorJob]:
        with self._lock:
            ordered = sorted(
                self._jobs.values(),
                key=lambda job: job.created_at_utc,
                reverse=True,
            )
            return [OperatorJob.from_dict(job.to_dict()) for job in ordered[: max(0, limit)]]

    def get_job(self, job_id: str) -> OperatorJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return OperatorJob.from_dict(job.to_dict())

    def _register(self, job: OperatorJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            self._persist(job)

    def _update(self, job_id: str, **fields: Any) -> OperatorJob:
        with self._lock:
            job = self._jobs[job_id]
            for name, value in fields.items():
                setattr(job, name, value)
            self._persist(job)
            return OperatorJob.from_dict(job.to_dict())

    def allocate_upload_path(self, filename: str | None) -> Path:
        safe_name = safe_upload_name(filename)
        path = self.uploads_dir / uuid.uuid4().hex / safe_name
        validate_video_suffix(path)
        path.parent.mkdir(parents=True, exist_ok=False)
        return path

    def discard_upload(self, path: Path) -> None:
        candidate = Path(path).resolve()
        try:
            candidate.relative_to(self.uploads_dir)
        except ValueError:
            return
        shutil.rmtree(candidate.parent, ignore_errors=True)

    def _runtime_config(
        self,
        *,
        sensor_id: str,
        modality: Modality,
        output_path: Path,
        config_path: Path | None,
    ) -> tuple[Path, str]:
        if not sensor_id.strip():
            raise ValueError("sensor_id must be non-empty")
        if modality not in {"EO", "IR", "OTHER"}:
            raise ValueError("modality must be EO, IR, or OTHER")

        config, provenance = load_production_config(config_path)
        payload = _json_safe(config.to_dict())
        anomaly = payload.get("anomaly") or {}
        reference = anomaly.get("reference_path")
        if reference:
            reference_path = Path(str(reference)).expanduser()
            if not reference_path.is_absolute():
                reference_path = (provenance.path.parent / reference_path).resolve()
            anomaly["reference_path"] = str(reference_path)
        payload["anomaly"] = anomaly
        payload["runtime"] = {"sensor_id": sensor_id.strip(), "modality": modality}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return output_path, str(provenance.path)

    def submit_run(
        self,
        video_path: Path,
        *,
        sensor_id: str,
        modality: Modality,
        config_path: Path | None = None,
    ) -> OperatorJob:
        video = Path(video_path).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        validate_video_suffix(video)
        if modality not in {"EO", "IR", "OTHER"}:
            raise ValueError("modality must be EO, IR, or OTHER")
        if not sensor_id.strip():
            raise ValueError("sensor_id must be non-empty")

        job_id = make_operator_job_id("run")
        output_dir = self.runs_dir / job_id
        job = OperatorJob(
            job_id=job_id,
            kind="run",
            status="queued",
            created_at_utc=utc_now(),
            started_at_utc=None,
            completed_at_utc=None,
            display_name=video.name,
            input_path=str(video),
            sensor_id=sensor_id.strip(),
            modality=modality,
            output_dir=str(output_dir),
            artifacts={},
            summary={},
        )
        self._register(job)
        self._executor.submit(
            self._execute_run,
            job_id,
            video,
            sensor_id.strip(),
            modality,
            Path(config_path).expanduser().resolve() if config_path else None,
        )
        return self.get_job(job_id)

    def _execute_run(
        self,
        job_id: str,
        video_path: Path,
        sensor_id: str,
        modality: Modality,
        config_path: Path | None,
    ) -> None:
        output_dir = Path(self._jobs[job_id].output_dir)
        self._update(job_id, status="running", started_at_utc=utc_now())
        try:
            generated_config, config_source = self._runtime_config(
                sensor_id=sensor_id,
                modality=modality,
                output_path=self.workspace / "runtime-configs" / f"{job_id}.yaml",
                config_path=config_path,
            )
            result = run_configured_video(
                video_path,
                generated_config,
                output_dir=output_dir,
            )
            manifest = json.loads(result.pipeline.run_manifest.read_text(encoding="utf-8"))
            artifacts = {
                "annotated_video": str(result.pipeline.annotated_video),
                "detections_csv": str(result.pipeline.detections_csv),
                "tracks_csv": str(result.pipeline.tracks_csv),
                "anomalies_csv": str(result.pipeline.anomalies_csv),
                "events_csv": str(result.pipeline.events_csv),
                "alerts_csv": str(result.pipeline.alerts_csv),
                "run_manifest": str(result.pipeline.run_manifest),
                "effective_config_json": str(result.effective_config_json),
                "run_log_jsonl": str(result.run_log_jsonl),
                "run_status_json": str(result.run_status_json),
            }
            summary = {
                "frames": manifest.get("frames_processed", 0),
                "detections": manifest.get("detections_emitted", 0),
                "tracks": manifest.get("unique_tracks", 0),
                "anomalies": manifest.get("anomalies_flagged", 0),
                "events": manifest.get("events_emitted", 0),
                "alerts": manifest.get("alerts_emitted", 0),
            }
            self._update(
                job_id,
                status="completed",
                completed_at_utc=utc_now(),
                config_source=config_source,
                pipeline_run_id=result.run_id,
                artifacts=artifacts,
                summary=summary,
                error=None,
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                completed_at_utc=utc_now(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )

    def submit_fusion(
        self,
        source_job_ids: list[str],
        *,
        config_path: Path | None = None,
    ) -> OperatorJob:
        if len(source_job_ids) < 2:
            raise ValueError("select at least two completed sensor runs")
        jobs = [self.get_job(job_id) for job_id in source_job_ids]
        if any(job.kind != "run" for job in jobs):
            raise ValueError("fusion inputs must be sensor run jobs")
        if any(job.status != "completed" for job in jobs):
            raise ValueError("all fusion inputs must be completed")
        sensors = [job.sensor_id for job in jobs]
        if len(set(sensors)) != len(sensors):
            raise ValueError("fusion inputs must have distinct sensor IDs")

        job_id = make_operator_job_id("fusion")
        output_dir = self.fusions_dir / job_id
        run_dirs = [job.output_dir for job in jobs]
        job = OperatorJob(
            job_id=job_id,
            kind="fusion",
            status="queued",
            created_at_utc=utc_now(),
            started_at_utc=None,
            completed_at_utc=None,
            display_name=" + ".join(job.display_name for job in jobs),
            output_dir=str(output_dir),
            source_job_ids=list(source_job_ids),
            source_run_dirs=list(run_dirs),
            artifacts={},
            summary={},
        )
        self._register(job)
        self._executor.submit(
            self._execute_fusion,
            job_id,
            [Path(path) for path in run_dirs],
            Path(config_path).expanduser().resolve() if config_path else None,
        )
        return self.get_job(job_id)

    def _execute_fusion(
        self,
        job_id: str,
        run_dirs: list[Path],
        config_path: Path | None,
    ) -> None:
        output_dir = Path(self._jobs[job_id].output_dir)
        self._update(job_id, status="running", started_at_utc=utc_now())
        try:
            result = fuse_run_directories(
                run_dirs,
                config_path=config_path,
                output_dir=output_dir,
            )
            manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
            artifacts = {
                "fusion_events_csv": str(result.events_csv),
                "fusion_alerts_csv": str(result.alerts_csv),
                "fusion_contributors_csv": str(result.contributors_csv),
                "fusion_manifest_json": str(result.manifest_json),
                "effective_fusion_config_json": str(result.effective_config_json),
            }
            summary = {
                "input_events": manifest.get("input_events", 0),
                "events": manifest.get("fused_events", 0),
                "alerts": manifest.get("fused_alerts", 0),
                "sensors": manifest.get("distinct_sensors", 0),
            }
            self._update(
                job_id,
                status="completed",
                completed_at_utc=utc_now(),
                fusion_id=result.fusion_id,
                config_source=manifest.get("config_source"),
                artifacts=artifacts,
                summary=summary,
                error=None,
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                completed_at_utc=utc_now(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )

    def artifact_path(self, job_id: str, artifact_name: str) -> Path:
        job = self.get_job(job_id)
        artifacts = job.artifacts or {}
        value = artifacts.get(artifact_name)
        if value is None:
            raise KeyError(artifact_name)
        path = Path(value).resolve()
        output_dir = Path(job.output_dir).resolve()
        try:
            path.relative_to(output_dir)
        except ValueError as exc:
            raise ValueError("artifact escaped job output directory") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def tabular_records(
        self,
        job_id: str,
        artifact_name: str,
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        path = self.artifact_path(job_id, artifact_name)
        table = pd.read_csv(path)
        if table.empty:
            return []
        rows = table.tail(max(0, min(limit, 1000))).where(pd.notna(table), None)
        return [_json_safe(record) for record in rows.to_dict(orient="records")]
