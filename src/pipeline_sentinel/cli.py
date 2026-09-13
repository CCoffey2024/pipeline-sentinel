from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .demo import run_demo
from .detectors import GroundTruthDetector
from .ingest import OpenCVVideoIngestAdapter
from .pipeline import PipelineSentinel


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline-sentinel",
        description="Pipeline Sentinel modular computer-vision prototype",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

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
    return parser


def main() -> int:
    args = _build_parser().parse_args()

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
        pipeline = PipelineSentinel(GroundTruthDetector(args.ground_truth))
        artifacts = pipeline.run_video(args.video, args.output)
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")

    print(f"Annotated video: {artifacts.annotated_video}")
    print(f"Alerts:          {artifacts.alerts_csv}")
    print(f"Run manifest:    {artifacts.run_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
