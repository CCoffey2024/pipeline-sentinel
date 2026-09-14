from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import cv2

from .types import FrameContext, Modality
from .visdrone import VisDroneDataset, VisDroneSequence

LocalSourceType = Literal["image_folder", "visdrone", "uavdt"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_NATURAL_PARTS = re.compile(r"(\d+)")
_UAVDT_FRAME = re.compile(r"^img(?P<index>\d+)$", re.IGNORECASE)


def _natural_key(name: str) -> tuple[object, ...]:
    parts = _NATURAL_PARTS.split(name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _image_paths(directory: Path) -> list[Path]:
    """Return sorted frame paths without decoding any imagery.

    Directory entries must be materialized briefly so filenames can be sorted deterministically, but
    image bytes are never loaded here. Runtime decoding remains one-frame-at-a-time.
    """

    resolved = Path(directory).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    names = [
        entry.name
        for entry in os.scandir(resolved)
        if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES
    ]
    names.sort(key=_natural_key)
    if not names:
        raise ValueError(f"image sequence contains no supported image files: {resolved}")
    return [resolved / name for name in names]


def _validate_sampling(frame_step: int, max_frames: int | None, render_fps: float) -> None:
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")
    if render_fps <= 0:
        raise ValueError("render_fps must be positive")


class LocalFrameSequence(Protocol):
    source_type: LocalSourceType
    sequence_id: str
    source_root: Path

    @property
    def frame_count(self) -> int: ...

    def iter_frames(
        self,
        *,
        frame_step: int = 1,
        max_frames: int | None = None,
        render_fps: float = 30.0,
        sensor_id: str,
        modality: Modality,
    ) -> Iterator[FrameContext]: ...


@dataclass(frozen=True, slots=True)
class ImageFolderSequence:
    """Read an arbitrary local image folder as an ordered frame stream."""

    source_root: Path
    sequence_id: str
    frames_dir: Path
    source_type: LocalSourceType = "image_folder"

    @property
    def frame_paths(self) -> list[Path]:
        return _image_paths(self.frames_dir)

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)

    def iter_frames(
        self,
        *,
        frame_step: int = 1,
        max_frames: int | None = None,
        render_fps: float = 30.0,
        sensor_id: str,
        modality: Modality,
    ) -> Iterator[FrameContext]:
        _validate_sampling(frame_step, max_frames, render_fps)
        emitted = 0
        for ordinal, image_path in enumerate(self.frame_paths):
            if ordinal % frame_step != 0:
                continue
            if max_frames is not None and emitted >= max_frames:
                break
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise RuntimeError(f"OpenCV could not decode image frame: {image_path}")
            yield FrameContext(
                frame_number=ordinal,
                timestamp_s=ordinal / render_fps,
                image=image,
                source_path=image_path,
                sensor_id=sensor_id,
                modality=modality,
            )
            emitted += 1


@dataclass(frozen=True, slots=True)
class VisDroneLocalSequence:
    """Adapt the existing VisDrone sequence contract to operator-selected sensor metadata."""

    sequence: VisDroneSequence
    source_type: LocalSourceType = "visdrone"

    @property
    def sequence_id(self) -> str:
        return self.sequence.sequence_id

    @property
    def source_root(self) -> Path:
        return self.sequence.dataset_root

    @property
    def frame_count(self) -> int:
        return self.sequence.frame_count

    def iter_frames(
        self,
        *,
        frame_step: int = 1,
        max_frames: int | None = None,
        render_fps: float = 30.0,
        sensor_id: str,
        modality: Modality,
    ) -> Iterator[FrameContext]:
        for frame in self.sequence.iter_frames(
            frame_step=frame_step,
            max_frames=max_frames,
            render_fps=render_fps,
        ):
            yield FrameContext(
                frame_number=frame.frame_number,
                timestamp_s=frame.timestamp_s,
                image=frame.image,
                source_path=frame.source_path,
                sensor_id=sensor_id,
                modality=modality,
            )


def _resolve_uavdt_root(root: Path) -> tuple[Path, Path]:
    requested = Path(root).expanduser().resolve()
    if not requested.exists():
        raise FileNotFoundError(requested)

    if requested.is_dir() and requested.name.lower() == "uav-benchmark-m":
        return requested.parent, requested
    if (requested / "UAV-benchmark-M").is_dir():
        return requested, requested / "UAV-benchmark-M"

    candidates: list[Path] = []
    for child in requested.iterdir() if requested.is_dir() else []:
        if not child.is_dir():
            continue
        if child.name.lower() == "uav-benchmark-m":
            candidates.append(child)
        if (child / "UAV-benchmark-M").is_dir():
            candidates.append(child / "UAV-benchmark-M")
        for grandchild in child.iterdir():
            if grandchild.is_dir() and grandchild.name.lower() == "uav-benchmark-m":
                candidates.append(grandchild)

    unique = sorted({path.resolve() for path in candidates}, key=str)
    if len(unique) == 1:
        return unique[0].parent, unique[0]
    if len(unique) > 1:
        formatted = "\n  - ".join(str(path) for path in unique)
        raise ValueError(
            "Multiple UAVDT UAV-benchmark-M roots were found. Pass one explicitly:\n  - "
            + formatted
        )
    raise ValueError(
        f"Could not locate UAVDT 'UAV-benchmark-M' beneath '{requested}'. "
        "Pass the UAVDT dataset root or UAV-benchmark-M directory."
    )


def _uavdt_frame_number(path: Path) -> int:
    match = _UAVDT_FRAME.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"UAVDT frame name must look like img000001.jpg: {path.name}")
    return int(match.group("index"))


@dataclass(frozen=True, slots=True)
class UavdtSequence:
    dataset_root: Path
    benchmark_root: Path
    sequence_id: str
    frames_dir: Path
    gt_dir: Path | None
    source_type: LocalSourceType = "uavdt"

    @property
    def source_root(self) -> Path:
        return self.dataset_root

    @property
    def frame_paths(self) -> list[Path]:
        paths = _image_paths(self.frames_dir)
        paths.sort(key=_uavdt_frame_number)
        return paths

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)

    @property
    def annotation_candidates(self) -> list[Path]:
        if self.gt_dir is None:
            return []
        names = [
            f"{self.sequence_id}_gt.txt",
            f"{self.sequence_id}_whole.txt",
            f"{self.sequence_id}_gt_whole.txt",
            f"{self.sequence_id}_gt_merge.txt",
        ]
        return [path for name in names if (path := self.gt_dir / name).is_file()]

    def iter_frames(
        self,
        *,
        frame_step: int = 1,
        max_frames: int | None = None,
        render_fps: float = 30.0,
        sensor_id: str,
        modality: Modality,
    ) -> Iterator[FrameContext]:
        _validate_sampling(frame_step, max_frames, render_fps)
        emitted = 0
        for ordinal, image_path in enumerate(self.frame_paths):
            if ordinal % frame_step != 0:
                continue
            if max_frames is not None and emitted >= max_frames:
                break
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise RuntimeError(f"OpenCV could not decode UAVDT frame: {image_path}")
            yield FrameContext(
                frame_number=_uavdt_frame_number(image_path),
                timestamp_s=ordinal / render_fps,
                image=image,
                source_path=image_path,
                sensor_id=sensor_id,
                modality=modality,
            )
            emitted += 1


class UavdtDataset:
    """Discovery adapter for the original UAVDT UAV-benchmark-M layout."""

    def __init__(self, root: Path) -> None:
        self.requested_root = Path(root).expanduser().resolve()
        self.root, self.benchmark_root = _resolve_uavdt_root(self.requested_root)
        candidate_gt = self.root / "UAV-benchmark-MOTD_v1.0" / "GT"
        self.gt_dir = candidate_gt if candidate_gt.is_dir() else None

    def sequence_ids(self) -> list[str]:
        return sorted(
            (path.name for path in self.benchmark_root.iterdir() if path.is_dir()),
            key=_natural_key,
        )

    def sequence(self, sequence_id: str) -> UavdtSequence:
        frames_dir = self.benchmark_root / sequence_id
        if not frames_dir.is_dir():
            available = self.sequence_ids()
            raise KeyError(
                f"Unknown UAVDT sequence '{sequence_id}'. Found {len(available)} sequences; "
                f"first entries: {', '.join(available[:8])}"
            )
        return UavdtSequence(
            dataset_root=self.root,
            benchmark_root=self.benchmark_root,
            sequence_id=sequence_id,
            frames_dir=frames_dir,
            gt_dir=self.gt_dir,
        )

    def summary(self) -> dict[str, object]:
        sequence_ids = self.sequence_ids()
        return {
            "requested_root": str(self.requested_root),
            "root": str(self.root),
            "benchmark_root": str(self.benchmark_root),
            "sequence_count": len(sequence_ids),
            "gt_dir": str(self.gt_dir) if self.gt_dir else None,
            "first_sequences": sequence_ids[:10],
        }


def inspect_local_source(
    source_type: LocalSourceType,
    root: Path,
    *,
    sequence_limit: int = 500,
) -> dict[str, object]:
    """Inspect local metadata only; never decode or copy source imagery."""

    if source_type == "image_folder":
        resolved = Path(root).expanduser().resolve()
        sequence = ImageFolderSequence(resolved, resolved.name, resolved)
        return {
            "source_type": source_type,
            "requested_root": str(resolved),
            "resolved_root": str(resolved),
            "sequence_count": 1,
            "sequences": [
                {
                    "sequence_id": sequence.sequence_id,
                    "frame_count": sequence.frame_count,
                }
            ],
            "sequence_list_truncated": False,
            "imagery_access": "read_in_place",
            "imagery_copied": False,
        }

    if source_type == "visdrone":
        dataset = VisDroneDataset(root)
        sequence_ids = dataset.sequence_ids()
        return {
            "source_type": source_type,
            "requested_root": str(dataset.requested_root),
            "resolved_root": str(dataset.root),
            "sequence_count": len(sequence_ids),
            "sequences": [{"sequence_id": value} for value in sequence_ids[:sequence_limit]],
            "sequence_list_truncated": len(sequence_ids) > sequence_limit,
            "imagery_access": "read_in_place",
            "imagery_copied": False,
        }

    if source_type == "uavdt":
        dataset = UavdtDataset(root)
        sequence_ids = dataset.sequence_ids()
        return {
            "source_type": source_type,
            "requested_root": str(dataset.requested_root),
            "resolved_root": str(dataset.root),
            "benchmark_root": str(dataset.benchmark_root),
            "sequence_count": len(sequence_ids),
            "sequences": [{"sequence_id": value} for value in sequence_ids[:sequence_limit]],
            "sequence_list_truncated": len(sequence_ids) > sequence_limit,
            "gt_dir": str(dataset.gt_dir) if dataset.gt_dir else None,
            "imagery_access": "read_in_place",
            "imagery_copied": False,
        }

    raise ValueError(f"unsupported local source type: {source_type}")


def open_local_sequence(
    source_type: LocalSourceType,
    root: Path,
    *,
    sequence_id: str | None = None,
) -> LocalFrameSequence:
    """Resolve one local image sequence without copying its source files."""

    if source_type == "image_folder":
        resolved = Path(root).expanduser().resolve()
        return ImageFolderSequence(resolved, sequence_id or resolved.name, resolved)
    if source_type == "visdrone":
        dataset = VisDroneDataset(root)
        sequence = dataset.sequence(sequence_id) if sequence_id else dataset.first_sequence()
        return VisDroneLocalSequence(sequence)
    if source_type == "uavdt":
        dataset = UavdtDataset(root)
        ids = dataset.sequence_ids()
        if not ids:
            raise ValueError(f"No UAVDT sequences found under {dataset.benchmark_root}")
        return dataset.sequence(sequence_id or ids[0])
    raise ValueError(f"unsupported local source type: {source_type}")
