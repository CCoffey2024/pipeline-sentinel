from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import pandas as pd

from .types import Detection, FrameContext

VISDRONE_CATEGORY_NAMES = {
    0: "ignored_region",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning_tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

VISDRONE_COLUMNS = [
    "frame_index",
    "target_id",
    "bbox_left",
    "bbox_top",
    "bbox_width",
    "bbox_height",
    "score",
    "object_category",
    "truncation",
    "occlusion",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _frame_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"VisDrone frame name must have a numeric stem: {path.name}") from exc


@dataclass(frozen=True, slots=True)
class VisDroneSequence:
    """One VisDrone VID image sequence and its optional annotation file."""

    dataset_root: Path
    sequence_id: str
    frames_dir: Path
    annotation_path: Path | None

    @property
    def frame_paths(self) -> list[Path]:
        paths = [
            path
            for path in self.frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        paths.sort(key=_frame_index)
        if not paths:
            raise ValueError(f"VisDrone sequence contains no image frames: {self.frames_dir}")
        return paths

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)

    def iter_frames(
        self,
        *,
        frame_step: int = 1,
        max_frames: int | None = None,
        render_fps: float = 30.0,
    ) -> Iterator[FrameContext]:
        """Yield decoded VisDrone frames as Pipeline Sentinel runtime contexts.

        VisDrone VID is distributed as JPEG image sequences rather than encoded MP4 clips. The
        dataset does not provide acquisition timing through the image files, so ``render_fps`` is a
        declared working cadence used for timestamps and the annotated output video.
        """

        if frame_step <= 0:
            raise ValueError("frame_step must be positive")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if render_fps <= 0:
            raise ValueError("render_fps must be positive")

        emitted = 0
        for ordinal, image_path in enumerate(self.frame_paths):
            if ordinal % frame_step != 0:
                continue
            if max_frames is not None and emitted >= max_frames:
                break

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise RuntimeError(f"OpenCV could not decode VisDrone frame: {image_path}")

            frame_number = _frame_index(image_path)
            yield FrameContext(
                frame_number=frame_number,
                timestamp_s=ordinal / render_fps,
                image=image,
                source_path=image_path,
                sensor_id=f"VISDRONE:{self.sequence_id}",
                modality="EO",
            )
            emitted += 1

    def annotations(self) -> pd.DataFrame:
        """Return native VisDrone VID ground truth plus normalized XYXY columns."""

        if self.annotation_path is None or not self.annotation_path.exists():
            raise FileNotFoundError(f"No VisDrone annotations found for sequence {self.sequence_id}")

        frame = pd.read_csv(self.annotation_path, header=None, names=VISDRONE_COLUMNS)
        if frame.shape[1] != len(VISDRONE_COLUMNS):
            raise ValueError(
                f"Unexpected VisDrone annotation schema in {self.annotation_path}: "
                f"expected {len(VISDRONE_COLUMNS)} columns"
            )

        numeric_columns = VISDRONE_COLUMNS
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")

        frame["label"] = frame["object_category"].map(VISDRONE_CATEGORY_NAMES).fillna("unknown")
        frame["x1"] = frame["bbox_left"].astype(int)
        frame["y1"] = frame["bbox_top"].astype(int)
        frame["x2"] = (frame["bbox_left"] + frame["bbox_width"]).astype(int)
        frame["y2"] = (frame["bbox_top"] + frame["bbox_height"]).astype(int)
        frame["ignored"] = frame["object_category"].astype(int).eq(0) | frame["score"].le(0)
        return frame

    def normalized_ground_truth(self) -> pd.DataFrame:
        columns = [
            "frame_index",
            "target_id",
            "label",
            "object_category",
            "score",
            "truncation",
            "occlusion",
            "ignored",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
        return self.annotations()[columns].copy()

    def detections_for_frame(self, frame_number: int, *, include_ignored: bool = False) -> list[Detection]:
        rows = self.annotations()
        rows = rows.loc[rows["frame_index"].astype(int) == int(frame_number)]
        if not include_ignored:
            rows = rows.loc[~rows["ignored"]]

        detections: list[Detection] = []
        for row in rows.itertuples(index=False):
            detections.append(
                Detection(
                    frame_number=int(row.frame_index),
                    label=str(row.label),
                    confidence=1.0,
                    x1=int(row.x1),
                    y1=int(row.y1),
                    x2=int(row.x2),
                    y2=int(row.y2),
                    source="visdrone_ground_truth",
                    object_id=int(row.target_id),
                    metadata={
                        "visdrone_category_id": int(row.object_category),
                        "score": float(row.score),
                        "truncation": int(row.truncation),
                        "occlusion": int(row.occlusion),
                        "ignored": bool(row.ignored),
                    },
                )
            )
        return detections


class VisDroneDataset:
    """Discovery/validation adapter for a VisDrone2019-VID train/val root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.sequences_dir = self.root / "sequences"
        self.annotations_dir = self.root / "annotations"

        if not self.root.exists():
            raise FileNotFoundError(self.root)
        if not self.sequences_dir.is_dir():
            raise ValueError(
                f"Expected VisDrone VID directory '{self.sequences_dir}'. "
                "Pass the VisDrone2019-VID-train or VisDrone2019-VID-val root."
            )

    @property
    def split_name(self) -> str:
        lower = self.root.name.lower()
        if lower.endswith("-train"):
            return "train"
        if lower.endswith("-val"):
            return "val"
        if "test" in lower:
            return "test"
        return self.root.name

    def sequence_ids(self) -> list[str]:
        return sorted(path.name for path in self.sequences_dir.iterdir() if path.is_dir())

    def sequence(self, sequence_id: str) -> VisDroneSequence:
        frames_dir = self.sequences_dir / sequence_id
        if not frames_dir.is_dir():
            available = self.sequence_ids()
            preview = ", ".join(available[:5])
            raise KeyError(
                f"Unknown VisDrone sequence '{sequence_id}'. "
                f"Found {len(available)} sequences; first entries: {preview}"
            )

        annotation_path = self.annotations_dir / f"{sequence_id}.txt"
        if not annotation_path.exists():
            annotation_path = None

        return VisDroneSequence(
            dataset_root=self.root,
            sequence_id=sequence_id,
            frames_dir=frames_dir,
            annotation_path=annotation_path,
        )

    def first_sequence(self) -> VisDroneSequence:
        sequence_ids = self.sequence_ids()
        if not sequence_ids:
            raise ValueError(f"No VisDrone sequences found under {self.sequences_dir}")
        return self.sequence(sequence_ids[0])

    def summary(self) -> dict[str, object]:
        sequence_ids = self.sequence_ids()
        annotation_count = 0
        if self.annotations_dir.is_dir():
            annotation_count = len(list(self.annotations_dir.glob("*.txt")))
        return {
            "root": str(self.root),
            "split": self.split_name,
            "sequence_count": len(sequence_ids),
            "annotation_file_count": annotation_count,
            "first_sequences": sequence_ids[:10],
        }
