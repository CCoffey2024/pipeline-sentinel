from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from .operator_local_sources import LocalSourceOperatorJobManager


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc.args[0]))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, RuntimeError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


def _table(manager: LocalSourceOperatorJobManager, job_id: str, artifact_name: str) -> pd.DataFrame:
    path = manager.artifact_path(job_id, artifact_name)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _optional_table(
    manager: LocalSourceOperatorJobManager,
    job_id: str,
    artifact_name: str,
) -> pd.DataFrame:
    try:
        return _table(manager, job_id, artifact_name)
    except (KeyError, FileNotFoundError):
        return pd.DataFrame()


def _optional_json(
    manager: LocalSourceOperatorJobManager,
    job_id: str,
    artifact_name: str,
) -> dict[str, Any]:
    try:
        path = manager.artifact_path(job_id, artifact_name)
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (KeyError, FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return {}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _managed_upload_root(
    manager: LocalSourceOperatorJobManager,
    input_path: str | None,
) -> Path | None:
    if not input_path:
        return None
    candidate = Path(input_path).expanduser().resolve()
    try:
        relative = candidate.relative_to(manager.uploads_dir)
    except ValueError:
        return None
    if not relative.parts:
        return None
    return manager.uploads_dir / relative.parts[0]


def _job_storage(manager: LocalSourceOperatorJobManager, job_id: str) -> dict[str, object]:
    job = manager.get_job(job_id)
    output_path = Path(job.output_dir).expanduser().resolve()
    output_bytes = manager._path_size(output_path)
    managed_input = _managed_upload_root(manager, job.input_path)
    input_bytes = manager._path_size(managed_input) if managed_input else 0
    runtime_config = manager.workspace / "runtime-configs" / f"{job_id}.yaml"
    registry_file = manager.jobs_dir / f"{job_id}.json"
    metadata_bytes = manager._path_size(runtime_config) + manager._path_size(registry_file)
    return {
        "output_bytes": output_bytes,
        "managed_input_bytes": input_bytes,
        "metadata_bytes": metadata_bytes,
        "total_bytes": output_bytes + input_bytes + metadata_bytes,
        "input_is_managed": managed_input is not None,
    }


def _run_result_summary(
    manager: LocalSourceOperatorJobManager,
    job_id: str,
) -> dict[str, object]:
    detections = _table(manager, job_id, "detections_csv")
    tracks = _table(manager, job_id, "tracks_csv")
    representations = _optional_table(manager, job_id, "representations_csv")
    representation_manifest = _optional_json(
        manager,
        job_id,
        "representation_manifest_json",
    )

    class_rows: list[dict[str, object]] = []
    if not detections.empty and "label" in detections.columns:
        labels = detections["label"].fillna("unknown").astype(str)
        counts = labels.value_counts()
        confidence = (
            pd.to_numeric(detections.get("confidence"), errors="coerce")
            if "confidence" in detections.columns
            else None
        )
        track_counts: dict[str, int] = {}
        if not tracks.empty and {"label", "track_id"}.issubset(tracks.columns):
            track_counts = {
                str(label): int(count)
                for label, count in tracks.groupby("label", dropna=False)["track_id"]
                .nunique()
                .items()
            }

        for label, count in counts.head(30).items():
            mask = labels == label
            mean_confidence = None
            if confidence is not None:
                mean_confidence = _finite(confidence[mask].mean())
            class_rows.append(
                {
                    "label": str(label),
                    "detections": int(count),
                    "tracks": track_counts.get(str(label), 0),
                    "mean_confidence": mean_confidence,
                }
            )

    frame_series: list[dict[str, int]] = []
    frame_stats: dict[str, float | int | None] = {
        "min": None,
        "max": None,
        "mean": None,
    }
    if not detections.empty and "frame_number" in detections.columns:
        counts_by_frame = detections.groupby("frame_number").size().sort_index()
        if not counts_by_frame.empty:
            frame_stats = {
                "min": int(counts_by_frame.min()),
                "max": int(counts_by_frame.max()),
                "mean": float(counts_by_frame.mean()),
            }
            stride = max(1, math.ceil(len(counts_by_frame) / 240))
            sampled = counts_by_frame.iloc[::stride]
            frame_series = [
                {"frame": int(frame), "count": int(count)} for frame, count in sampled.items()
            ]

    return {
        "kind": "run",
        "classes": class_rows,
        "detections_by_frame": frame_series,
        "detection_frame_stats": frame_stats,
        "representations": {
            "enabled": bool(representation_manifest.get("enabled", False)),
            "observations": int(len(representations)),
            "tracks": (
                int(representations["track_id"].nunique())
                if not representations.empty and "track_id" in representations.columns
                else 0
            ),
            "embedding_dimension": representation_manifest.get("embedding_dimension"),
            "analyzer": representation_manifest.get("analyzer"),
            "embedder": (representation_manifest.get("configuration") or {}).get("embedder"),
        },
        "storage": _job_storage(manager, job_id),
    }


def _fusion_result_summary(
    manager: LocalSourceOperatorJobManager,
    job_id: str,
) -> dict[str, object]:
    events = _table(manager, job_id, "fusion_events_csv")
    event_rows: list[dict[str, object]] = []
    if not events.empty and "event_type" in events.columns:
        counts = events["event_type"].fillna("unknown").astype(str).value_counts()
        event_rows = [
            {"label": str(label), "detections": int(count), "tracks": 0, "mean_confidence": None}
            for label, count in counts.head(30).items()
        ]
    return {
        "kind": "fusion",
        "classes": event_rows,
        "detections_by_frame": [],
        "detection_frame_stats": {"min": None, "max": None, "mean": None},
        "storage": _job_storage(manager, job_id),
    }


def build_operator_results_router(manager: LocalSourceOperatorJobManager) -> APIRouter:
    """Build lightweight result-inspection and disposal routes for the local operator console."""

    router = APIRouter()
    result_cache: dict[str, dict[str, object]] = {}

    @router.get("/api/jobs/{job_id}/results")
    def job_results(job_id: str) -> dict[str, object]:
        try:
            job = manager.get_job(job_id)
            if job.status != "completed":
                raise ValueError("results are available only for completed jobs")
            cached = result_cache.get(job_id)
            if cached is not None:
                return cached
            summary = (
                _fusion_result_summary(manager, job_id)
                if job.kind == "fusion"
                else _run_result_summary(manager, job_id)
            )
            result_cache[job_id] = summary
            return summary
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/api/jobs/{job_id}/detections")
    def job_detections(
        job_id: str,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        try:
            job = manager.get_job(job_id)
            if job.kind != "run":
                return []
            return manager.tabular_records(job_id, "detections_csv", limit=limit)
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/api/jobs/{job_id}/tracks")
    def job_tracks(
        job_id: str,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        try:
            job = manager.get_job(job_id)
            if job.kind != "run":
                return []
            return manager.tabular_records(job_id, "tracks_csv", limit=limit)
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/api/jobs/{job_id}/anomalies")
    def job_anomalies(
        job_id: str,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        try:
            job = manager.get_job(job_id)
            if job.kind != "run":
                return []
            return manager.tabular_records(job_id, "anomalies_csv", limit=limit)
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/api/jobs/{job_id}/representations")
    def job_representations(
        job_id: str,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        try:
            job = manager.get_job(job_id)
            if job.kind != "run" or not (job.artifacts or {}).get("representations_csv"):
                return []
            return manager.tabular_records(job_id, "representations_csv", limit=limit)
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, object]:
        try:
            job = manager.get_job(job_id)
            if job.status in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="active jobs cannot be deleted")
            deleted = manager.delete_job(job_id)
            result_cache.pop(job_id, None)
            return deleted
        except HTTPException:
            raise
        except Exception as exc:
            raise _http_error(exc) from exc

    return router
