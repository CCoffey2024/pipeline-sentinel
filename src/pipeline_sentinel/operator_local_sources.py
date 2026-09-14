from __future__ import annotations

import json
from pathlib import Path

from .image_sources import LocalSourceType, open_local_sequence
from .operator_jobs import OperatorJob, OperatorJobManager, make_operator_job_id, utc_now
from .operations import run_configured_frames
from .types import Modality


class LocalSourceOperatorJobManager(OperatorJobManager):
    """Extend the operator queue with read-in-place local image sources."""

    def submit_local_sequence(
        self,
        *,
        source_type: LocalSourceType,
        source_root: Path,
        sequence_id: str | None,
        sensor_id: str,
        modality: Modality,
        frame_step: int = 1,
        max_frames: int | None = None,
        render_fps: float = 30.0,
        config_path: Path | None = None,
    ) -> OperatorJob:
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if render_fps <= 0:
            raise ValueError("render_fps must be positive")
        if modality not in {"EO", "IR", "OTHER"}:
            raise ValueError("modality must be EO, IR, or OTHER")
        if not sensor_id.strip():
            raise ValueError("sensor_id must be non-empty")

        sequence = open_local_sequence(source_type, source_root, sequence_id=sequence_id)
        job_id = make_operator_job_id("run")
        output_dir = self.runs_dir / job_id
        display_name = f"{source_type}:{sequence.sequence_id}"
        job = OperatorJob(
            job_id=job_id,
            kind="run",
            status="queued",
            created_at_utc=utc_now(),
            started_at_utc=None,
            completed_at_utc=None,
            display_name=display_name,
            input_path=str(sequence.source_root),
            sensor_id=sensor_id.strip(),
            modality=modality,
            output_dir=str(output_dir),
            artifacts={},
            summary={"source_type": source_type, "sequence_id": sequence.sequence_id},
        )
        self._register(job)
        self._executor.submit(
            self._execute_local_sequence,
            job_id,
            source_type,
            Path(source_root).expanduser().resolve(),
            sequence.sequence_id,
            sensor_id.strip(),
            modality,
            frame_step,
            max_frames,
            render_fps,
            Path(config_path).expanduser().resolve() if config_path else None,
        )
        return self.get_job(job_id)

    def _execute_local_sequence(
        self,
        job_id: str,
        source_type: LocalSourceType,
        source_root: Path,
        sequence_id: str,
        sensor_id: str,
        modality: Modality,
        frame_step: int,
        max_frames: int | None,
        render_fps: float,
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
            sequence = open_local_sequence(source_type, source_root, sequence_id=sequence_id)
            frames = sequence.iter_frames(
                frame_step=frame_step,
                max_frames=max_frames,
                render_fps=render_fps,
                sensor_id=sensor_id,
                modality=modality,
            )
            source_name = f"{source_type}:{sequence.sequence_id}@{sequence.source_root}"
            result = run_configured_frames(
                frames,
                generated_config,
                source_name=source_name,
                render_fps=render_fps,
                output_dir=output_dir,
                run_metadata={
                    "source_type": source_type,
                    "source_root": str(sequence.source_root),
                    "sequence_id": sequence.sequence_id,
                    "frame_step": frame_step,
                    "max_frames": max_frames,
                    "imagery_access": "read_in_place",
                    "imagery_copied": False,
                },
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
                "source_type": source_type,
                "sequence_id": sequence.sequence_id,
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
