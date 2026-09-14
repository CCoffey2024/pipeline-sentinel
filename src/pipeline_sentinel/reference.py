from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .anomaly import AnomalyReference, encode_in_batches, fit_normal_reference
from .embeddings import DinoV2DependencyError, DinoV2Embedder, Embedder

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def discover_reference_images(directory: Path) -> list[Path]:
    """Return a stable, recursive list of supported normal-reference images."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported reference images found under: {root}")
    return paths


def _source_digest(paths: list[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_reference_images(paths: list[Path]) -> list[np.ndarray]:
    """Load reference crops and fail loudly on unreadable input rather than silently skipping it."""

    images: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Could not decode reference image: {path}")
        images.append(image)
    return images


def build_reference_from_directory(
    directory: Path,
    output_path: Path,
    *,
    embedder: Embedder,
    quantile: float = 0.95,
    batch_size: int = 16,
) -> AnomalyReference:
    """Fit and persist the normal-centroid reference extracted from Notebook 06."""

    root = Path(directory).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("Anomaly reference output must use the .npz extension")
    paths = discover_reference_images(root)
    images = load_reference_images(paths)
    embeddings = encode_in_batches(embedder, images, batch_size=batch_size)
    reference = fit_normal_reference(
        embeddings,
        embedder_name=embedder.name,
        quantile=quantile,
        metadata={
            "source_directory": str(root),
            "source_image_count": len(paths),
            "source_sha256": _source_digest(paths, root=root),
        },
    )
    reference.save(output)
    return reference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline-sentinel-fit-reference",
        description=(
            "Build a normal-activity anomaly reference from curated image crops using DINOv2"
        ),
    )
    parser.add_argument("normal_crops", type=Path, help="Directory of curated normal image crops")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz reference artifact")
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument("--device", default=None, help="Torch device such as cpu or cuda")
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        embedder = DinoV2Embedder(model_name=args.model, device=args.device)
        reference = build_reference_from_directory(
            args.normal_crops,
            args.output,
            embedder=embedder,
            quantile=args.quantile,
            batch_size=args.batch_size,
        )
    except (DinoV2DependencyError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "embedder": reference.embedder_name,
                "embedding_dim": reference.embedding_dim,
                "normal_samples": reference.sample_count,
                "quantile": reference.quantile,
                "threshold": reference.threshold,
                "metadata": reference.metadata,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
