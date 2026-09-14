from __future__ import annotations

import mimetypes
import shutil
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .image_sources import IMAGE_SUFFIXES, LocalSourceType, inspect_local_source
from .operator_jobs import safe_upload_name, validate_video_suffix
from .operator_local_sources import LocalSourceOperatorJobManager
from .operator_results import build_operator_results_router

DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class FusionRequest(BaseModel):
    job_ids: list[str] = Field(min_length=2)


class LocalSourceInspectRequest(BaseModel):
    source_type: LocalSourceType
    path: str = Field(min_length=1)


class LocalSequenceRunRequest(BaseModel):
    source_type: LocalSourceType
    path: str = Field(min_length=1)
    sequence_id: str | None = None
    sensor_id: str = Field(min_length=1)
    modality: Literal["EO", "IR", "OTHER"] = "EO"
    frame_step: int = Field(default=1, ge=1)
    max_frames: int | None = Field(default=None, ge=1)
    fps: float = Field(default=30.0, gt=0)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc.args[0]))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, RuntimeError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


def _media_kind(filename: str | None) -> Literal["video", "image"]:
    """Classify one browser upload using the runtime's supported media suffixes."""

    candidate = Path(filename or "")
    if candidate.suffix.lower() in IMAGE_SUFFIXES:
        return "image"
    try:
        validate_video_suffix(candidate)
    except ValueError as exc:
        allowed_images = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(
            f"unsupported media extension {candidate.suffix!r}; "
            f"supported image extensions: {allowed_images}; "
            "or provide a supported encoded video"
        ) from exc
    return "video"


async def _write_upload(
    upload: UploadFile,
    destination: Path,
    *,
    total_bytes: int,
    max_upload_bytes: int,
) -> int:
    """Stream one browser upload to disk while enforcing the request-wide byte ceiling."""

    wrote = 0
    with destination.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            wrote += len(chunk)
            total_bytes += len(chunk)
            if total_bytes > max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"upload exceeds configured limit of {max_upload_bytes} bytes",
                )
            handle.write(chunk)
    if wrote == 0:
        raise HTTPException(status_code=400, detail=f"uploaded media is empty: {destination.name}")
    return total_bytes


def create_app(
    *,
    workspace: Path = Path("outputs/operator"),
    max_workers: int = 1,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    allow_local_sources: bool = True,
) -> FastAPI:
    """Create the local Pipeline Sentinel operator API and bundled web console."""

    if max_upload_bytes < 1:
        raise ValueError("max_upload_bytes must be positive")
    resolved_workspace = Path(workspace).expanduser().resolve()
    manager = LocalSourceOperatorJobManager(resolved_workspace, max_workers=max_workers)

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
    app.state.allow_local_sources = allow_local_sources
    app.include_router(build_operator_results_router(manager))

    @app.get("/", response_class=HTMLResponse)
    def operator_console() -> HTMLResponse:
        page = Path(__file__).with_name("web") / "operator.html"
        if not page.is_file():
            raise HTTPException(status_code=500, detail="bundled operator console is missing")
        html = page.read_text(encoding="utf-8")
        html = html.replace("</body>", '<script src="/local-sources.js"></script>\n</body>')
        return HTMLResponse(html)

    @app.get("/local-sources.js")
    def local_sources_javascript() -> FileResponse:
        script = Path(__file__).with_name("web") / "local-sources.js"
        if not script.is_file():
            raise HTTPException(status_code=500, detail="bundled local-source UI is missing")
        return FileResponse(script, media_type="text/javascript")

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
            "local_sources_enabled": allow_local_sources,
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

    @app.post("/api/jobs/media", status_code=202)
    async def submit_media(
        files: Annotated[
            list[UploadFile],
            File(description="One encoded video or one or more still-image frames"),
        ],
        sensor_id: Annotated[str, Form()] = "EO_CAM_01",
        modality: Annotated[Literal["EO", "IR", "OTHER"], Form()] = "EO",
        fps: Annotated[float, Form()] = 30.0,
    ) -> dict[str, object]:
        """Submit one operator run from a video or an ordered group of still images."""

        if not files:
            raise HTTPException(status_code=400, detail="select at least one media file")
        if fps <= 0:
            raise HTTPException(status_code=400, detail="fps must be positive")

        try:
            kinds = [_media_kind(upload.filename) for upload in files]
        except Exception as exc:
            for upload in files:
                await upload.close()
            raise _http_error(exc) from exc

        try:
            if "video" in kinds:
                if len(files) != 1 or kinds != ["video"]:
                    raise HTTPException(
                        status_code=400,
                        detail="select one video, or select one or more still images; mixed media runs are not supported",
                    )
                upload = files[0]
                try:
                    upload_path = manager.allocate_upload_path(upload.filename)
                except Exception as exc:
                    raise _http_error(exc) from exc
                try:
                    await _write_upload(
                        upload,
                        upload_path,
                        total_bytes=0,
                        max_upload_bytes=max_upload_bytes,
                    )
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

            upload_dir = manager.uploads_dir / uuid4().hex
            upload_dir.mkdir(parents=True, exist_ok=False)
            total_bytes = 0
            used_names: set[str] = set()
            try:
                for index, upload in enumerate(files, start=1):
                    name = safe_upload_name(upload.filename or f"frame-{index:06d}.jpg")
                    if name in used_names:
                        stem = Path(name).stem
                        suffix = Path(name).suffix
                        name = f"{stem}-{index:06d}{suffix}"
                    used_names.add(name)
                    target = upload_dir / name
                    total_bytes = await _write_upload(
                        upload,
                        target,
                        total_bytes=total_bytes,
                        max_upload_bytes=max_upload_bytes,
                    )

                job = manager.submit_local_sequence(
                    source_type="image_folder",
                    source_root=upload_dir,
                    sequence_id=f"uploaded-images-{upload_dir.name[:8]}",
                    sensor_id=sensor_id,
                    modality=modality,
                    render_fps=fps,
                )
                return job.to_dict()
            except HTTPException:
                shutil.rmtree(upload_dir, ignore_errors=True)
                raise
            except Exception as exc:
                shutil.rmtree(upload_dir, ignore_errors=True)
                raise _http_error(exc) from exc
        finally:
            for upload in files:
                await upload.close()

    @app.post("/api/jobs/run", status_code=202)
    async def submit_run(
        file: Annotated[UploadFile, File(description="EO/IR video to analyze")],
        sensor_id: Annotated[str, Form()] = "EO_CAM_01",
        modality: Annotated[Literal["EO", "IR", "OTHER"], Form()] = "EO",
    ) -> dict[str, object]:
        """Backward-compatible encoded-video upload endpoint."""

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

    @app.post("/api/local-sources/inspect")
    def inspect_source(request: LocalSourceInspectRequest) -> dict[str, object]:
        if not allow_local_sources:
            raise HTTPException(
                status_code=403,
                detail="read-in-place local sources are disabled when the service is remotely bound",
            )
        try:
            return inspect_local_source(request.source_type, Path(request.path))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/jobs/local-sequence", status_code=202)
    def submit_local_sequence(request: LocalSequenceRunRequest) -> dict[str, object]:
        if not allow_local_sources:
            raise HTTPException(
                status_code=403,
                detail="read-in-place local sources are disabled when the service is remotely bound",
            )
        try:
            job = manager.submit_local_sequence(
                source_type=request.source_type,
                source_root=Path(request.path),
                sequence_id=request.sequence_id,
                sensor_id=request.sensor_id,
                modality=request.modality,
                frame_step=request.frame_step,
                max_frames=request.max_frames,
                render_fps=request.fps,
            )
            return job.to_dict()
        except Exception as exc:
            raise _http_error(exc) from exc

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
            "refusing non-local bind without --allow-remote; the operator console has no authentication"
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
        allow_local_sources=host in _LOCAL_HOSTS,
    )
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{port}/"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
