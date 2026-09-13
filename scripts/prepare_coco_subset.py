"""Create a deterministic smaller COCO-format benchmark subset.

The script is provider-neutral: obtain a COCO-format export from Roboflow, the official COCO
release, or another source, then point this utility at the instances JSON and image directory.
Dataset bytes are not committed to Git.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

DEFAULT_CATEGORIES = ["person", "car", "truck", "bus", "motorcycle", "bicycle", "airplane"]


def _load_coco(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"images", "annotations", "categories"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"COCO JSON is missing required top-level keys: {sorted(missing)}")
    return payload


def build_subset(
    annotations: Path,
    images: Path,
    output: Path,
    *,
    categories: list[str] | None = None,
    max_images: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a deterministic COCO subset and return its manifest.

    Only annotations for the requested categories are retained. An image is eligible when it
    contains at least one requested category. Selection is deterministic for the same source JSON,
    category list, seed, and maximum image count.
    """

    if max_images <= 0:
        raise ValueError("max_images must be greater than zero")
    if not images.is_dir():
        raise NotADirectoryError(images)

    requested_categories = list(categories or DEFAULT_CATEGORIES)
    if not requested_categories:
        raise ValueError("At least one category is required")
    if len(requested_categories) != len(set(requested_categories)):
        raise ValueError("Categories must be unique")

    payload = _load_coco(annotations)
    name_to_id = {str(c["name"]): int(c["id"]) for c in payload["categories"]}
    missing_categories = sorted(set(requested_categories) - set(name_to_id))
    if missing_categories:
        raise ValueError(
            f"Requested categories absent from COCO annotations: {missing_categories}"
        )

    selected_category_ids = {name_to_id[name] for name in requested_categories}
    candidate_image_ids = {
        int(annotation["image_id"])
        for annotation in payload["annotations"]
        if int(annotation["category_id"]) in selected_category_ids
    }

    image_by_id = {
        int(image["id"]): image
        for image in payload["images"]
        if int(image["id"]) in candidate_image_ids
    }

    # Sort before shuffling so Python set/hash ordering cannot affect repeatability.
    selected_ids = sorted(image_by_id)
    random.Random(seed).shuffle(selected_ids)
    selected_ids = selected_ids[:max_images]
    selected_id_set = set(selected_ids)

    selected_images = [image_by_id[image_id] for image_id in selected_ids]
    selected_annotations = [
        annotation
        for annotation in payload["annotations"]
        if int(annotation["image_id"]) in selected_id_set
        and int(annotation["category_id"]) in selected_category_ids
    ]
    selected_categories = [
        category
        for category in payload["categories"]
        if int(category["id"]) in selected_category_ids
    ]

    output_images = output / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    for image in selected_images:
        relative_name = Path(str(image["file_name"]))
        source = images / relative_name
        destination = output_images / relative_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    subset = {
        "info": payload.get("info", {}),
        "licenses": payload.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": selected_categories,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotations.json").write_text(
        json.dumps(subset, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest = {
        "source_annotations": str(annotations),
        "source_images": str(images),
        "categories": requested_categories,
        "seed": seed,
        "max_images": max_images,
        "selected_images": len(selected_images),
        "selected_annotations": len(selected_annotations),
    }
    (output / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic subset from an existing COCO-format export."
    )
    parser.add_argument("annotations", type=Path, help="COCO instances JSON")
    parser.add_argument("images", type=Path, help="Directory containing source images")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_subset(
        args.annotations,
        args.images,
        args.output,
        categories=args.categories,
        max_images=args.max_images,
        seed=args.seed,
    )
    print(
        "Wrote "
        f"{manifest['selected_images']} images and "
        f"{manifest['selected_annotations']} annotations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
