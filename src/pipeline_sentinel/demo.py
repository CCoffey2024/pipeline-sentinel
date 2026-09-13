from __future__ import annotations

from pathlib import Path

from .detectors import GroundTruthDetector
from .events import ScenarioRoleEventDetector, SeverityAlertPolicy
from .ingest import OpenCVVideoIngestAdapter
from .pipeline import PipelineSentinel, RunArtifacts
from .synthetic import generate_demo_video


def run_demo(
    output_dir: Path,
    *,
    frame_count: int = 120,
    fps: float = 10.0,
    size: tuple[int, int] = (640, 360),
) -> RunArtifacts:
    output_dir = Path(output_dir).resolve()
    raw_dir = output_dir / "data" / "raw"
    interim_dir = output_dir / "data" / "interim"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    eo_video = raw_dir / "pipeline_demo_eo.mp4"
    eo_gt = raw_dir / "pipeline_demo_eo_gt.csv"
    ir_video = raw_dir / "pipeline_demo_ir.mp4"
    ir_gt = raw_dir / "pipeline_demo_ir_gt.csv"

    generate_demo_video(eo_video, eo_gt, modality="EO", frame_count=frame_count, fps=fps, size=size)
    generate_demo_video(ir_video, ir_gt, modality="IR", frame_count=frame_count, fps=fps, size=size)

    ingest = OpenCVVideoIngestAdapter(sample_every_n=2)
    eo_manifest = ingest.extract(
        eo_video,
        interim_dir / "frames" / "eo",
        sensor_id="EO_CAM_01",
        modality="EO",
    )
    ir_manifest = ingest.extract(
        ir_video,
        interim_dir / "frames" / "ir",
        sensor_id="IR_CAM_01",
        modality="IR",
    )
    eo_manifest.to_csv(interim_dir / "eo_manifest.csv", index=False)
    ir_manifest.to_csv(interim_dir / "ir_manifest.csv", index=False)

    pipeline = PipelineSentinel(
        GroundTruthDetector(eo_gt),
        event_detector=ScenarioRoleEventDetector(),
        alert_policy=SeverityAlertPolicy(minimum_severity="warning"),
    )
    return pipeline.run_video(eo_video, output_dir / "run")
