from __future__ import annotations

import mimetypes
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .operator_jobs import OperatorJobManager

DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class FusionRequest(BaseModel):
    job_ids: list[str] = Field(min_length=2)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc.args[0]))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, RuntimeError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


def create_app(
    *,
    workspace: Path = Path("outputs/operator"),
    max_workers: int = 1,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> FastAPI:
    """Create the local Pipeline Sentinel operator API and bundled web console."""

    if max_upload_bytes < 1:
        raise ValueError("max_upload_bytes must be positive")
    resolved_workspace = Path(workspace).expanduser().resolve()
    manager = OperatorJobManager(resolved_workspace, max_workers=max_workers)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            manager.close()

    app = FastAPI(
        title="Pipeline Sentinel Operator API",
        version=__version__,
        description="Local operator service for starting sensor runs, inspecting evidence, and fusing completed runs.",
        lifespan=lifespan,
    )
    app.state.job_manager = manager
    app.state.workspace = resolved_workspace

    @app.get("/", response_class=HTMLResponse)
    def operator_console() -> HTMLResponse:
        page = Path(__file__).with_name("web") / "operator.html"
        if not page.is_file():
            raise HTTPException(status_code=500, detail="bundled operator console is missing")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, object]:
        jobs = manager.list_jobs(limit=1000)
        active = sum(job.status in {"queued", "running"} for job in jobs)
        return {
            "status": "ok",
            "version": __version__,
            "workspace": str(resolved_workspace),
            "active_jobs": active,
            "jobs_known": len(jobs),
            "max_workers": max_workers,
            "max_upload_bytes": max_upload_bytes,
        }

    @app.get("/api/jobs")
    def list_jobs(limit: int = 100) -> list[dict[str, object]]:
        limit = max(1, min(limit, 500))
        return [job.to_dict() for job in manager.list_jobs(limit=limit)]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, object]:
        try:
            return manager.get_job(job_id).to_dict()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/jobs/run", status_code=202)
    async def submit_run(
        file: Annotated[UploadFile, File(description="EO/IR video to analyze")],
        sensor_id: Annotated[str, Form()] = "EO_CAM_01",
        modality: Annotated[Literal["EO", "IR", "OTHER"], Form()] = "EO",
    ) -> dict[str, object]:
        try:
            upload_path = manager.allocate_upload_path(file.filename)
        except Exception as exc:
            raise _http_error(exc) from exc

        total = 0
        try:
            with upload_path.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"upload exceeds configured limit of {max_upload_bytes} bytes",
                        )
                    handle.write(chunk)
            if total == 0:
                raise HTTPException(status_code=400, detail="uploaded video is empty")
            job = manager.submit_run(
                upload_path,
                sensor_id=sensor_id,
                modality=modality,
            )
            return job.to_dict()
        except HTTPException:
            manager.discard_upload(upload_path)
            raise
        except Exception as exc:
            manager.discard_upload(upload_path)
            raise _http_error(exc) from exc
        finally:
            await file.close()

    @app.post("/api/jobs/fusion", status_code=202)
    def submit_fusion(request: FusionRequest) -> dict[str, object]:
        try:
            job = manager.submit_fusion(request.job_ids)
            return job.to_dict()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/jobs/{job_id}/alerts")
    def job_alerts(job_id: str, limit: int = 200) -> list[dict[str, object]]:
        try:
            job = manager.get_job(job_id)
            name = "alerts_csv" if job.kind == "run" else "fusion_alerts_csv"
            return manager.tabular_records(job_id, name, limit=limit)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str, limit: int = 200) -> list[dict[str, object]]:
        try:
            job = manager.get_job(job_id)
            name = "events_csv" if job.kind == "run" else "fusion_events_csv"
            return manager.tabular_records(job_id, name, limit=limit)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/jobs/{job_id}/artifacts")
    def job_artifacts(job_id: str) -> dict[str, object]:
        try:
            job = manager.get_job(job_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        artifacts = job.artifacts or {}
        return {
            "job_id": job.job_id,
            "artifacts": {
                name: {
                    "filename": Path(path).name,
                    "url": f"/api/jobs/{job.job_id}/artifacts/{name}",
                }
                for name, path in artifacts.items()
            },
        }

    @app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
    def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
        try:
            path = manager.artifact_path(job_id, artifact_name)
        except Exception as exc:
            raise _http_error(exc) from exc
        media_type, _ = mimetypes.guess_type(path.name)
        if path.suffix.lower() == ".mp4":
            media_type = "video/mp4"
        return FileResponse(path, media_type=media_type, filename=path.name)

    return app


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace: Path = Path("outputs/operator"),
    max_workers: int = 1,
    max_upload_gib: float = 4.0,
    open_browser: bool = False,
    allow_remote: bool = False,
) -> None:
    """Run the operator console with safe workstation-oriented defaults."""

    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if max_upload_gib <= 0:
        raise ValueError("max_upload_gib must be positive")
    if host not in _LOCAL_HOSTS and not allow_remote:
        raise ValueError(
            "refusing non-local bind without --allow-remote; the v0.10 operator console has no authentication"
        )

    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Operator service dependencies are not installed. Install the 'operator' extra."
        ) from exc

    max_upload_bytes = int(max_upload_gib * 1024**3)
    app = create_app(
        workspace=workspace,
        max_workers=max_workers,
        max_upload_bytes=max_upload_bytes,
    )
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{port}/"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
