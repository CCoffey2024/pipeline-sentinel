import json
import subprocess
import sys
from pathlib import Path


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        (images / name).write_bytes(name.encode("utf-8"))

    payload = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "c.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10]},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [2, 2, 10, 10]},
            {"id": 12, "image_id": 2, "category_id": 2, "bbox": [3, 3, 10, 10]},
            {"id": 13, "image_id": 3, "category_id": 3, "bbox": [4, 4, 10, 10]},
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"},
            {"id": 3, "name": "dog"},
        ],
    }
    annotations = tmp_path / "instances.json"
    annotations.write_text(json.dumps(payload), encoding="utf-8")
    return annotations, images


def _run_subset_script(
    annotations: Path,
    images: Path,
    output: Path,
) -> tuple[dict, dict]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_coco_subset.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(annotations),
            str(images),
            str(output),
            "--categories",
            "person",
            "car",
            "--max-images",
            "2",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subset = json.loads((output / "annotations.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "subset_manifest.json").read_text(encoding="utf-8"))
    return subset, manifest


def test_coco_subset_cli_filters_categories_and_is_deterministic(tmp_path: Path) -> None:
    annotations, images = _write_fixture(tmp_path)

    first, first_manifest = _run_subset_script(
        annotations,
        images,
        tmp_path / "first",
    )
    second, second_manifest = _run_subset_script(
        annotations,
        images,
        tmp_path / "second",
    )

    assert first == second
    assert {category["name"] for category in first["categories"]} == {"person", "car"}
    assert all(annotation["category_id"] in {1, 2} for annotation in first["annotations"])
    assert first_manifest["selected_images"] == 2
    assert first_manifest == second_manifest
