from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, load_production_config
from .demo import run_demo
from .detectors import GroundTruthDetector
from .evaluation import evaluate_visdrone_run
from .events import DwellEventDetector, ScenarioRoleEventDetector, SeverityAlertPolicy
from .fusion import FusionConfigError, fuse_run_directories, load_fusion_config
from .ingest import OpenCVVideoIngestAdapter
from .operations import ProductionRunArtifacts, run_configured_video
from .pipeline import PipelineSentinel, RunArtifacts
from .tracking import IoUTracker
from .visdrone import VisDroneDataset
from .yolo import YoloDependencyError, YoloDetector


def _add_yolo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--conf", dest="confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu, cuda:0, or 0")
    parser.add_argument(
        "--class-id",
        dest="class_ids",
        action="append",
        type=int,
        default=None,
        help="Restrict inference to a COCO class ID; repeat for multiple classes",
    )


def _add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracker-iou",
        type=float,
        default=0.30,
        help="Minimum same-class IoU for baseline track association",
    )
    parser.add_argument(
        "--tracker-max-missed",
        type=int,
        default=2,
        help="Number of missed frame updates before an unmatched track expires",
    )
    parser.add_argument(
        "--enable-dwell-events",
        action="store_true",
        help="Enable baseline persistence/low-displacement events for learned detections",
    )
    parser.add_argument("--dwell-min-hits", type=int, default=30)
    parser.add_argument("--dwell-max-displacement-px", type=float, default=40.0)
    parser.add_argument(
        "--dwell-label",
        dest="dwell_labels",
        action="append",
        default=None,
        help="Restrict dwell events to one label; repeat for multiple labels",
    )


def _make_yolo_detector(args: argparse.Namespace) -> YoloDetector:
    return YoloDetector(
        model_name=args.model,
        confidence=args.confidence,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        class_ids=args.class_ids,
    )


def _make_learned_pipeline(detector: YoloDetector, args: argparse.Namespace) -> PipelineSentinel:
    tracker = IoUTracker(
        iou_threshold=args.tracker_iou,
        max_missed_updates=args.tracker_max_missed,
    )
    event_detector = None
    if args.enable_dwell_events:
        event_detector = DwellEventDetector(
            min_hits=args.dwell_min_hits,
            max_displacement_px=args.dwell_max_displacement_px,
            labels=set(args.dwell_labels) if args.dwell_labels else None,
        )
    return PipelineSentinel(
        detector,
        tracker=tracker,
        event_detector=event_detector,
        alert_policy=SeverityAlertPolicy(minimum_severity="warning"),
    )


def _print_artifacts(artifacts: RunArtifacts) -> None:
    print(f"Annotated video: {artifacts.annotated_video}")
    print(f"Detections:      {artifacts.detections_csv}")
    print(f"Tracks:          {artifacts.tracks_csv}")
    print(f"Anomalies:       {artifacts.anomalies_csv}")
    print(f"Events:          {artifacts.events_csv}")
    print(f"Alerts:          {artifacts.alerts_csv}")
    print(f"Run manifest:    {artifacts.run_manifest}")


def _print_production_artifacts(artifacts: ProductionRunArtifacts) -> None:
    print(f"Run ID:           {artifacts.run_id}")
    print(f"Output directory: {artifacts.output_dir}")
    _print_artifacts(artifacts.pipeline)
    print(f"Effective config: {artifacts.effective_config_json}")
    print(f"Run log:          {artifacts.run_log_jsonl}")
    print(f"Run status:       {artifacts.run_status_json}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline-sentinel",
        description="Pipeline Sentinel modular computer-vision application",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    production = sub.add_parser(
        "run",
        help="Run the config-driven production video pipeline",
    )
    production.add_argument("video", type=Path)
    production.add_argument(
        "--config",
        type=Path,
        default=Path("config/production.yaml"),
        help="Validated YAML runtime profile",
    )
    production.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Exact run output directory; default is outputs/runs/<run-id>",
    )

    validate = sub.add_parser(
        "validate-config",
        help="Validate and print an effective production YAML configuration",
    )
    validate.add_argument("config", type=Path)

    validate_fusion = sub.add_parser(
        "validate-fusion-config",
        help="Validate and print an effective late-fusion YAML configuration",
    )
    validate_fusion.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("config/fusion.yaml"),
    )

    fuse = sub.add_parser(
        "fuse-runs",
        help="Late-fuse events from two or more completed sensor run directories",
    )
    fuse.add_argument("run_dirs", type=Path, nargs="+")
    fuse.add_argument(
        "--config",
        type=Path,
        default=Path("config/fusion.yaml"),
        help="Validated fusion profile",
    )
    fuse.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Exact fusion output directory; default is outputs/fusion/<fusion-id>",
    )

    demo = sub.add_parser("demo", help="Run deterministic synthetic end-to-end smoke test")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--frames", type=int, default=120)
    demo.add_argument("--fps", type=float, default=10.0)
    demo.add_argument("--width", type=int, default=640)
    demo.add_argument("--height", type=int, default=360)

    extract = sub.add_parser("extract", help="Extract video into the canonical frame manifest")
    extract.add_argument("video", type=Path)
    extract.add_argument("--frames-dir", type=Path, required=True)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--sensor-id", default="CAM_01")
    extract.add_argument("--modality", choices=["EO", "IR", "OTHER"], default="EO")
    extract.add_argument("--sample-every", type=int, default=1)

    ref = sub.add_parser("run-reference", help="Run the deterministic GT-backed pipeline")
    ref.add_argument("video", type=Path)
    ref.add_argument("ground_truth", type=Path)
    ref.add_argument("--output", type=Path, default=Path("outputs/reference"))

    yolo = sub.add_parser("run-yolo", help="Run YOLO on one encoded video")
    yolo.add_argument("video", type=Path)
    yolo.add_argument("--output", type=Path, default=Path("outputs/yolo"))
    _add_yolo_arguments(yolo)
    _add_tracking_arguments(yolo)
    yolo.add_argument("--sensor-id", default="EO_CAM_01")
    yolo.add_argument("--modality", choices=["EO", "IR", "OTHER"], default="EO")

    vd_info = sub.add_parser(
        "visdrone-info",
        help="Validate a VisDrone2019-VID train/val root and list image sequences",
    )
    vd_info.add_argument("dataset_root", type=Path)
    vd_info.add_argument("--limit", type=int, default=20)

    vd_run = sub.add_parser(
        "run-visdrone",
        help="Run YOLO directly over a VisDrone2019-VID image sequence",
    )
    vd_run.add_argument("dataset_root", type=Path)
    vd_run.add_argument(
        "--sequence",
        default=None,
        help="VisDrone sequence ID; defaults to the first sequence in sorted order",
    )
    vd_run.add_argument("--output", type=Path, default=Path("outputs/visdrone"))
    vd_run.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for a quick acceptance run; default processes the whole sequence",
    )
    vd_run.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth source frame",
    )
    vd_run.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Working cadence for timestamps and rendered MP4; VisDrone VID is stored as JPEG frames",
    )
    _add_yolo_arguments(vd_run)
    _add_tracking_arguments(vd_run)

    vd_eval = sub.add_parser(
        "evaluate-visdrone",
        help="Evaluate one saved VisDrone run with the shared COCO/VisDrone ontology",
    )
    vd_eval.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing detections.csv, ground_truth.csv, and run_manifest.json",
    )
    vd_eval.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold used for fixed-threshold TP/FP/FN matching",
    )
    vd_eval.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Benchmark output directory; defaults to <run_dir>/benchmark",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.command == "run":
        try:
            production_artifacts = run_configured_video(
                args.video,
                args.config,
                output_dir=args.output,
            )
        except (ConfigError, FileNotFoundError, YoloDependencyError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        _print_production_artifacts(production_artifacts)
        return 0

    if args.command == "validate-config":
        try:
            config, provenance = load_production_config(args.config)
        except (ConfigError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "valid": True,
                    "config_source": str(provenance.path),
                    "config_sha256": provenance.sha256,
                    "effective_config": config.to_dict(),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate-fusion-config":
        try:
            config, provenance = load_fusion_config(args.config)
        except (FusionConfigError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "valid": True,
                    "config_source": str(provenance.path),
                    "config_sha256": provenance.sha256,
                    "effective_config": config.to_dict(),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "fuse-runs":
        try:
            fusion = fuse_run_directories(
                args.run_dirs,
                config_path=args.config,
                output_dir=args.output,
            )
        except (FusionConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"Fusion ID:       {fusion.fusion_id}")
        print(f"Output directory:{' '}{fusion.output_dir}")
        print(f"Fused events:    {fusion.events_csv}")
        print(f"Fused alerts:    {fusion.alerts_csv}")
        print(f"Contributors:    {fusion.contributors_csv}")
        print(f"Fusion manifest: {fusion.manifest_json}")
        print(f"Effective config:{' '}{fusion.effective_config_json}")
        return 0

    if args.command == "demo":
        artifacts = run_demo(
            args.output,
            frame_count=args.frames,
            fps=args.fps,
            size=(args.width, args.height),
        )
    elif args.command == "extract":
        adapter = OpenCVVideoIngestAdapter(sample_every_n=args.sample_every)
        manifest = adapter.extract(
            args.video,
            args.frames_dir,
            sensor_id=args.sensor_id,
            modality=args.modality,
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.manifest, index=False)
        print(f"Extracted {len(manifest)} sampled frames -> {args.manifest}")
        return 0
    elif args.command == "run-reference":
        pipeline = PipelineSentinel(
            GroundTruthDetector(args.ground_truth),
            event_detector=ScenarioRoleEventDetector(),
            alert_policy=SeverityAlertPolicy(minimum_severity="warning"),
        )
        artifacts = pipeline.run_video(args.video, args.output)
    elif args.command == "run-yolo":
        try:
            detector = _make_yolo_detector(args)
        except YoloDependencyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        pipeline = _make_learned_pipeline(detector, args)
        artifacts = pipeline.run_video(
            args.video,
            args.output,
            sensor_id=args.sensor_id,
            modality=args.modality,
        )
    elif args.command == "visdrone-info":
        dataset = VisDroneDataset(args.dataset_root)
        summary = dataset.summary()
        print(json.dumps(summary, indent=2))
        limit = max(0, args.limit)
        for sequence_id in dataset.sequence_ids()[:limit]:
            sequence = dataset.sequence(sequence_id)
            annotation_state = "annotations" if sequence.annotation_path else "no annotations"
            print(f"{sequence_id}: {sequence.frame_count} frames, {annotation_state}")
        return 0
    elif args.command == "run-visdrone":
        dataset = VisDroneDataset(args.dataset_root)
        sequence = dataset.sequence(args.sequence) if args.sequence else dataset.first_sequence()
        output_dir = Path(args.output).resolve() / dataset.split_name / sequence.sequence_id
        output_dir.mkdir(parents=True, exist_ok=True)

        selected_frame_numbers = sequence.selected_frame_numbers(
            frame_step=args.frame_step,
            max_frames=args.max_frames,
        )
        ground_truth_path: Path | None = None
        ground_truth_rows: int | None = None
        if sequence.annotation_path is not None:
            ground_truth_path = output_dir / "ground_truth.csv"
            ground_truth = sequence.normalized_ground_truth(frame_numbers=selected_frame_numbers)
            ground_truth.to_csv(ground_truth_path, index=False)
            ground_truth_rows = len(ground_truth)

        try:
            detector = _make_yolo_detector(args)
        except YoloDependencyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        print(
            f"VisDrone {dataset.split_name}: sequence={sequence.sequence_id}, "
            f"frames={sequence.frame_count}, step={args.frame_step}, max={args.max_frames or 'all'}"
        )
        pipeline = _make_learned_pipeline(detector, args)
        artifacts = pipeline.run_frames(
            sequence.iter_frames(
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                render_fps=args.fps,
            ),
            output_dir,
            render_fps=args.fps,
            source_name=f"VisDrone2019-VID:{dataset.split_name}/{sequence.sequence_id}",
            run_metadata={
                "source_type": "visdrone_image_sequence",
                "dataset": "VisDrone2019-VID",
                "dataset_root": str(dataset.root),
                "split": dataset.split_name,
                "sequence_id": sequence.sequence_id,
                "annotation_path": str(sequence.annotation_path) if sequence.annotation_path else None,
                "ground_truth_csv": str(ground_truth_path) if ground_truth_path else None,
                "ground_truth_scope": "processed_frames" if ground_truth_path else None,
                "ground_truth_rows": ground_truth_rows,
                "frame_step": args.frame_step,
                "max_frames": args.max_frames,
                "detector_model": args.model,
                "detector_confidence": args.confidence,
                "detector_nms_iou": args.iou,
                "detector_imgsz": args.imgsz,
                "detector_device": args.device,
                "detector_class_ids": args.class_ids,
                "tracker_iou": args.tracker_iou,
                "tracker_max_missed": args.tracker_max_missed,
                "dwell_events_enabled": args.enable_dwell_events,
                "dwell_min_hits": args.dwell_min_hits if args.enable_dwell_events else None,
                "dwell_max_displacement_px": (
                    args.dwell_max_displacement_px if args.enable_dwell_events else None
                ),
                "dwell_labels": args.dwell_labels if args.enable_dwell_events else None,
                "timing_note": (
                    "VisDrone VID is distributed as image sequences; --fps is a declared working "
                    "cadence for timestamps/rendering, not recovered acquisition timing."
                ),
            },
        )
    elif args.command == "evaluate-visdrone":
        try:
            result, benchmark_artifacts = evaluate_visdrone_run(
                args.run_dir,
                iou_threshold=args.iou_threshold,
                output_dir=args.output,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        overall = result.summary["overall"]
        print(
            "Shared-class benchmark: "
            f"TP={overall['true_positive']} FP={overall['false_positive']} "
            f"FN={overall['false_negative']} precision={overall['precision']} "
            f"recall={overall['recall']} f1={overall['f1']}"
        )
        print(result.class_metrics.to_string(index=False))
        print(f"Benchmark summary: {benchmark_artifacts.summary_json}")
        print(f"Class metrics:     {benchmark_artifacts.class_metrics_csv}")
        print(f"Matches:           {benchmark_artifacts.matches_csv}")
        print(f"Excluded preds:    {benchmark_artifacts.excluded_predictions_csv}")
        print(f"Excluded GT:       {benchmark_artifacts.excluded_ground_truth_csv}")
        return 0
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")

    _print_artifacts(artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
