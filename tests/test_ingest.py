from pathlib import Path

from pipeline_sentinel.ingest import OpenCVVideoIngestAdapter
from pipeline_sentinel.manifest import validate_frame_manifest
from pipeline_sentinel.synthetic import generate_demo_video


def test_opencv_ingest_emits_valid_manifest(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    ground_truth = tmp_path / "gt.csv"
    generate_demo_video(video, ground_truth, frame_count=20, fps=10.0, size=(160, 96))

    adapter = OpenCVVideoIngestAdapter(sample_every_n=2)
    manifest = adapter.extract(
        video,
        tmp_path / "frames",
        sensor_id="EO_TEST",
        modality="EO",
    )

    assert len(manifest) == 10
    assert manifest["frame_number"].tolist() == list(range(0, 20, 2))
    assert manifest["timestamp_s"].tolist()[1] == 0.2
    assert manifest["sensor_id"].eq("EO_TEST").all()
    assert manifest["modality"].eq("EO").all()
    validate_frame_manifest(manifest)
